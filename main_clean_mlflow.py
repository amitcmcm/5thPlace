import torch
print(torch.cuda.is_available())
print(torch.version.cuda,torch.__version__)
import torch.nn as nn
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.distributions import Beta
from torchvision.ops import sigmoid_focal_loss
import os
import torch.cuda.amp as amp
import random
import gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
import mlflow


GPU = torch.cuda.is_available()
WAV_SIZE=2000
STEP_SIZE=500
TIMES_TRAIN=8
IS_MIXED_PRECISION = True
SEED=42
SHUFFLE_BEFORE_SPLITTING = False
DROPOUT_RATE = 0

INPUT_PATH_NP = r'N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\data_np' # folder of preprocessed data (output of data_creation)
INPUT_PATH = r'N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl' # folder containing meta files
OUTPUT_PATH = r'D:\non-dataset-spec-w-dl'

hparams = {
    # Optional hparams
    "backbone": 'wavenet_4096',
    "learning_rate": [5e-4],
    "max_epochs": 30, #
    "batch_size": 16,
    "num_workers": 0,
    "val_sanity_checks": 0,
    "fast_dev_run": False,
    "output_path": f"",
    "gpu": torch.cuda.is_available(),
    'div_factor':5,
    'final_div_factor':10,
    'scaler': amp.GradScaler(),
    'wav_size':WAV_SIZE,
    'step_size':STEP_SIZE,
    'times_train':TIMES_TRAIN,
    'is_mixed_precision':IS_MIXED_PRECISION,
    'seed': SEED,
    'shuffle_before_splitting': SHUFFLE_BEFORE_SPLITTING,
    'dropout_rate': DROPOUT_RATE



}



def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms = True
    random.seed(0)
    np.random.seed(0)


class FOGDataset(torch.utils.data.Dataset):

    def __init__(self, df, is_train=False,transforms=None):
        self.is_train = is_train
        self.data = df

    def __len__(self):
        if self.is_train:
            return len(self.data)*TIMES_TRAIN
        else:
            return len(self.data)


    def __getitem__(self, idx):
        if self.is_train:
            idx = np.random.randint(0,len(self.data))

        row = self.data.iloc[idx]
        wav = np.load(INPUT_PATH_NP + fr'\{row.Id}_sig.npy')
        tgt = np.load(INPUT_PATH_NP + fr'\{row.Id}_tgt.npy')

        wav = wav/40.

        label = tgt
        wav_df = pd.DataFrame(wav)
        tgt_df = pd.DataFrame(label)

        wavs = []
        tgts = []
        if self.is_train:
            for w in wav_df.rolling(WAV_SIZE,step=STEP_SIZE):
                if w.shape[0] == WAV_SIZE:
                    wavs.append(w.values)

            if len(wavs) ==0:
                wavs = [wav]

            for w in tgt_df.rolling(WAV_SIZE,step=STEP_SIZE):
                if w.shape[0] == WAV_SIZE:
                    tgts.append(w.values)

            if len(tgts) ==0:
                tgts = [label]

            wav = np.stack(wavs,axis=0)
            label = np.stack(tgts,axis=0)
            actual_len=-1

        else:
            actual_len = len(wav)
            nchunk = (len(wav)//WAV_SIZE)+1
            wav = wav.reshape(-1,len(wav),3)
            label = label.reshape(-1,len(label),3)


        if self.is_train and len(wav)>1:
            rix = np.random.randint(0,len(wav))
            wav = wav[rix:rix+1]
            label = label[rix:rix+1]

        #print('wav',wav.shape, label.shape)

        sample = {"wav": wav, "label":label, "actual_len":actual_len}

        #print('label',label.shape,tgt.shape)
        #print('wav',wav.shape)

        return sample

def collate_wrapper(batch):
    out = {}
    wavs = []
    labels = []
    for item in batch:
        wavs.append(item['wav'])
        labels.append(item['label'])
    out['wav'] = torch.from_numpy(np.concatenate(wavs,axis=0))
    out['label'] = torch.from_numpy(np.concatenate(labels,axis=0))

    return out

def getDataLoader(params,train_x,val_x,train_transforms=None,val_transforms=None):

    train_dataset = FOGDataset(
            df=train_x, is_train=True, transforms=train_transforms
        )
    val_dataset = FOGDataset(df=val_x, transforms=val_transforms)

    trainDataLoader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=params['batch_size'],
                            num_workers=params['num_workers'],
                            shuffle=True,collate_fn = collate_wrapper,
                            pin_memory=False,
                            worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id)
                        )
    valDataLoader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=1,
                        num_workers=params['num_workers'],
                        shuffle=False,
                        pin_memory=False,
                    )

    return trainDataLoader,valDataLoader




class Mixup(nn.Module):
    def __init__(self, mix_beta=1):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1,1) * Y + (1 - coeffs.view(-1, 1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight




class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res


class Classifier(nn.Module):
    def __init__(self, inch=3, kernel_size=3, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.LSTM = nn.GRU(input_size=128, hidden_size=128, num_layers=4,
                           batch_first=True, bidirectional=True)

        # self.wave_block1 = Wave_Block(inch, 16, 12, kernel_size)
        self.wave_block2 = Wave_Block(inch, 32, 8, kernel_size)
        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)

        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(256, 3)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)

        x = self.wave_block4(x)
        x = x.permute(0, 2, 1)
        x, h = self.LSTM(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return x



class AmpNet(Classifier):

    def __init__(self, params):
        super(AmpNet, self).__init__()

    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(AmpNet, self).forward(*args)



def getOptimzersScheduler(model, params, steps_in_epoch=25, pct_start=0.1):
    mdl_parameters = [
        {'params': model.parameters(), 'lr': 1e-4},
    ]

    optimizer = torch.optim.Adam(mdl_parameters, lr=params['learning_rate'][0])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=1,
                                                    pct_start=pct_start,
                                                    max_lr=params['learning_rate'],
                                                    epochs=params['max_epochs'],
                                                    div_factor=params['div_factor'],
                                                    final_div_factor=params['final_div_factor'],
                                                    verbose=True)

    return optimizer, scheduler, False


def save_model(epoch, model, ckpt_path='./', name='', val_rmse=0):
    path = os.path.join(ckpt_path, '{}_{}.pth'.format(name, epoch))
    torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)


def load_model(model, ckpt_path):
    state = torch.load(ckpt_path)
    print(model.load_state_dict(state, strict=False))
    return model


def focal_loss(pred,target):
    return 32*sigmoid_focal_loss(pred,target,reduction='mean')


def training_step(model, batch, batch_idx,optimizer,scaler,scheduler,isStepScheduler=False):
    # Load images and labels
    x = batch["wav"].float()
    y = batch["label"].float()


    # Mixup Augmentation
    mixup = Mixup()

    if np.random.uniform(0,1) < 0:
        x,y = mixup(x,y)

    #print('x',x.shape,ys1.shape,ye1.shape)

    if GPU:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

    criterion = focal_loss

    iters_to_accumulate=2

    with amp.autocast():
        preds = model(x)
        b,s,c = y.shape
        y = y.reshape(b*s,c)
        preds = preds.reshape(b*s,-1)
        loss = criterion(preds,y)/ iters_to_accumulate

        scaler.scale(loss).backward()

        if (batch_idx + 1) % iters_to_accumulate == 0: # the optimization step is performed only for batches where batch_idx+1 is a multiple of iters_to_accumulate, meaning optimization every iters_to_accumulate (currently 2) mini-batches
        # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
            scaler.unscale_(optimizer) # In mixed precision training, gradients are scaled up to avoid underflow. Before updating the weights, these scaled gradients need to be unscaled back to their original range in preparation for the optimization step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss = loss.item()


    if isStepScheduler:
        scheduler.step()

    return loss


def validation_step(model, batch, batch_idx):
    # Load images and labels
    x = batch["wav"].float()
    y = batch["label"].float()
    actual_len = batch['actual_len'].long()
    iters_to_accumulate = 2

    if GPU:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

    criterion = focal_loss

    x = x[0]
    y = y[0]
    actual_len = actual_len[0]

    BS = 20

    preds_list = []
    tgt_list = []

    # Forward pass & softmax
    with torch.no_grad():
        with amp.autocast():
            num_iter =  x.shape[0]//BS
            if num_iter == 0:
                num_iter = 1
            for b in range(num_iter):
                preds = model(x[b*BS:(b+1)*BS]) # prediction in batches
                yb = y[b*BS:(b+1)*BS]
                b,s,c = yb.shape
                yb = yb.reshape(b*s,c)
                preds = preds.reshape(b*s,-1)
                preds_list.append(preds)
                tgt_list.append(yb)

            preds = torch.cat(preds_list,dim=0)
            y = torch.cat(tgt_list,dim=0)

            #print('preds',preds.shape,y.shape)

            y = y[0:actual_len]
            preds = preds[0:actual_len]
            loss = criterion(preds, y) / iters_to_accumulate

    preds = torch.sigmoid(preds)

    loss = loss.item()
    return loss,preds.detach().cpu().numpy(),y.long().detach().cpu().numpy()




def train_epoch(model, trainDataLoader, optimizer, scaler, scheduler, isStepScheduler=True):

    total_loss = 0
    model.train()
    torch.set_grad_enabled(True)
    total_step = 0

    pbar = tqdm(enumerate(trainDataLoader), total=len(trainDataLoader))
    try: #delete later
        for bi, data in pbar:
            loss = training_step(model, data, bi, optimizer, scaler, scheduler)
            total_loss += loss
            total_step += 1
            pbar.set_postfix({'loss': total_loss / total_step})
    except:
        import traceback
        traceback.print_exc()


    if not isStepScheduler:  # in case of epoch based scheduler
        scheduler.step()

    total_loss /= total_step
    return total_loss


def val_epoch(model, valDataLoader):
    total_loss = 0

    total_step = 0
    model.eval()
    preds = []
    targets = []
    pbar = tqdm(enumerate(valDataLoader), total=len(valDataLoader))
    for bi, data in pbar:
        loss, pred, tgt = validation_step(model, data, bi)
        total_loss += loss
        total_step += 1
        preds.extend(pred)
        targets.extend(tgt)

        pbar.set_postfix({'loss': total_loss / total_step})

    preds = np.stack(preds)
    preds = np.clip(preds, 0, 1)
    targets = np.stack(targets)

    print('targets', targets.shape, preds.shape)
    aps = []
    for i in range(3):
        score = average_precision_score(targets[:, i], preds[:, i])
        aps.append(score)

    APx = average_precision_score(targets, preds, average='macro')
    AP = np.mean(aps)

    del targets, preds
    gc.collect()

    print('AP', AP, APx)
    total_loss /= total_step
    return total_loss, AP


def training_loop(params, train_x, val_x, savedir='./', mdl_name='resnet34'):
    # create model
    model = AmpNet(params).cuda()

    # get loaders
    trainDataLoader, valDataLoader = getDataLoader(params, train_x, val_x)

    scaler = params['scaler']
    optimizer, scheduler, isStepScheduler = getOptimzersScheduler(model, params,
                                                                  steps_in_epoch=len(trainDataLoader),
                                                                  pct_start=0.1)
    best_ap = 0
    # control loop
    for e in range(params['max_epochs']):
        train_loss = train_epoch(model, trainDataLoader, optimizer, scaler, scheduler, isStepScheduler)
        loss, AP = val_epoch(model, valDataLoader)

        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("val_loss", loss)
        mlflow.log_metric("AP", AP)

        print(e, 'Val Result', f'AP={AP} ')
        if AP > best_ap:
            print(f'Saving for AP {AP}')
            save_model(e, model, ckpt_path=savedir, name=mdl_name, val_rmse=best_ap)

            model_info = mlflow.sklearn.log_model(
                model,
                artifact_path=mdl_name,
                # registered_model_name=mdl_name,
            )
            best_ap = AP
        else:
            try:
                print(f'Not Saving for AP {AP}')
            except:
                print('not saving')







def main():

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment("temp_lab_fogathome_30e_bs16_test10per_excsub_2d57c2")

    with mlflow.start_run():

        mlflow.log_params(hparams)
        mlflow.set_tag("Training Info", "same settings and data as fortunate-cod-99, plus added dropout (0.5). with selecting at random (seed=42) 10% of the lab subjects for test. subject 2d57c2 is excluded. this could potentially have an effect on the internal test set, so need to check that if we want to recreate the exact conditions of the previous runs with the previous internal test set.")

        set_seed(SEED)

        metadata = pd.read_csv(f'{INPUT_PATH}/meta_data.csv')
        if SHUFFLE_BEFORE_SPLITTING:
            # metadata = metadata.sample(frac=1, random_state=0).reset_index(drop=True)
            unique_subjects = metadata.Subject.unique()
            #np.random.seed(0)
            np.random.seed(1)
            np.random.shuffle(unique_subjects)
            # Re-map the shuffled subjects back to the metadata
            subject_mapping = {old: new for new, old in enumerate(unique_subjects)}
            metadata['Shuffled_Subject'] = metadata['Subject'].map(subject_mapping)
        else:
            metadata['Shuffled_Subject'] = metadata['Subject']


        kf = GroupKFold(n_splits=5)
        for i, (train_index, test_index) in enumerate(kf.split(metadata.Id, metadata.Medication, groups=metadata.Shuffled_Subject)):
            metadata.loc[test_index, 'fold'] = i

        version = '6'
        for fn in [1, 4, 0, 2, 3]: #[1, 4, 0, 2, 3]:
            set_seed()

            mdl_name = hparams['backbone']
            savedir = os.path.join(OUTPUT_PATH, f'trained-models-clean-{mdl_name}-v{version}')
            Path(savedir).mkdir(exist_ok=True, parents=True)

            val = metadata[metadata.fold == fn]
            tr = metadata[metadata.fold != fn]

            print('FOLD', fn, 'Train', tr.shape, 'Val', val.shape)

            training_loop(hparams, tr, val, savedir=savedir, mdl_name=f'{mdl_name}-fold{fn}')
            gc.collect()




if __name__ == "__main__":
    main()