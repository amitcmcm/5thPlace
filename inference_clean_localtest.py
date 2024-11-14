import torch
torch.cuda.is_available()
import torch.cuda.amp as amp
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import librosa
import random
import glob
import os
import pandas as pd

# parameters
WAV_SIZE = 20000
TIMES_TRAIN = 8
IS_MIXED_PRECISION = True
TARGET_COLS = ['StartHesitation', 'Turn', 'Walking']
GPU = torch.cuda.is_available()
SEED = 42

MLFLOW_RUN_NAME = 'fortunate-cod-99'
INFERENCE_DATA = r'N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\data_test' # path of inference data
OUTPUT_PATH = r'N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl'

# mostly related to training
hparams = {
    # Optional hparams
    "backbone": 'wavenet_4096',
    "learning_rate": [5e-4],
    "max_epochs": 121,
    "batch_size": 8,
    "num_workers": 0,
    "val_sanity_checks": 0,
    "fast_dev_run": False,
    "output_path": f"",
    "gpu": torch.cuda.is_available(),
    'div_factor': 10,
    'final_div_factor': 20,
}



class FOGDataset(torch.utils.data.Dataset):

    def __init__(self, df, is_train=False, transforms=None):
        self.is_train = is_train
        self.data = df

    def __len__(self):
        if self.is_train:
            return len(self.data) * TIMES_TRAIN
        else:
            return len(self.data)


    def __getitem__(self, idx):
        g0 = 9.80665
        row = self.data.iloc[idx]
        data = pd.read_csv(row.filename)

        print(row.Id, data.shape)

        sig = data[['AccV', 'AccML', 'AccAP']].values

        if row.fs != 100:
            sigs = []
            for c in range(3):
                sigs.append(librosa.resample(sig[:, c], orig_sr=row.fs, target_sr=100))
            wav = np.stack(sigs, axis=1)
        if row.scale_to_g:
            wav = sig * g0

        print('after resampling', wav.shape)
        wav = wav / 40.
        act_len = len(wav)
        nchunk = len(wav) // WAV_SIZE
        rem_size = len(wav) - nchunk * WAV_SIZE
        arrs = []
        for chk in range(nchunk):
            arrs.append(wav[chk * WAV_SIZE:(chk + 1) * WAV_SIZE])

        if rem_size > 0:
            last_arr = wav[-WAV_SIZE:]
            arrs.append(last_arr)

        wav = np.stack(arrs, axis=0)

        print('wav', wav.shape, rem_size)

        sample = {"wav": wav, "Id": row.Id, 'df_length': len(data),
                  'act_len': act_len, 'nchunk': nchunk, 'rem_size': rem_size, 'fs': row.fs}

        return sample


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms = True
    random.seed(0)
    np.random.seed(0)


def getDataLoader(params, val_x):
    val_dataset = FOGDataset(df=val_x, transforms=None)

    valDataLoader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=params['num_workers'],
        shuffle=False,
        pin_memory=False,
    )

    return valDataLoader


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
    def __init__(self, inch=3, kernel_size=3):
        super().__init__()
        self.LSTM = nn.GRU(input_size=128, hidden_size=128, num_layers=4,
                           batch_first=True, bidirectional=True)

        self.wave_block2 = Wave_Block(inch, 32, 8, kernel_size)
        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)
        self.fc1 = nn.Linear(256, 3)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)

        x = self.wave_block4(x)
        x = x.permute(0, 2, 1)
        x, h = self.LSTM(x)
        x = self.fc1(x)

        return x, x, x



class AmpNet(Classifier):

    def __init__(self, params):
        super(AmpNet, self).__init__()

    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(AmpNet, self).forward(*args)



def load_model(model, ckpt_path):
    state = torch.load(ckpt_path)
    print(model.load_state_dict(state, strict=False))
    return model


def validation_step(model, batch):
    # Load images and labels
    x = batch["wav"].float()
    if GPU:
        x = x.cuda(non_blocking=True)
    x = x[0]

    # print('x', x.shape)

    # Forward pass & softmax
    flat_pred = np.zeros((batch['act_len'][0], 3))
    with torch.no_grad(): # disabling gradient calculation to reduce memory consumption during inference
        if IS_MIXED_PRECISION:
            with amp.autocast():
                preds, _, _ = model(x)
                preds = preds

                print('preds', preds.shape)

    for i in range(batch['nchunk'][0]): # iterate over the data chunks
        flat_pred[i * WAV_SIZE:(i + 1) * WAV_SIZE] = torch.sigmoid(preds[i]).detach().cpu().numpy()

    rem_sz = batch['rem_size'][0]
    if rem_sz > 0:
        flat_pred[-rem_sz:] = torch.sigmoid(preds[-1]).detach().cpu().numpy()[-rem_sz:]

    return flat_pred




def test_epoch(model, valDataLoader):
    model.eval()
    pred_dfs = []

    pbar = tqdm(enumerate(valDataLoader), total=len(valDataLoader))
    for bi, data in pbar:
        pred = validation_step(model, data)

        if data['fs'][0] != 100:
            preds = []
            for clss in range(3):
                preds.append(librosa.resample(pred[:, clss].astype(np.float32), orig_sr=100, target_sr=data['fs'][0]))

            pred = np.stack(preds, axis=1)
            pred = np.clip(pred, 0, 1)

            pred1 = np.zeros((data['df_length'][0], 3))
            pred1[0:data['df_length'][0]] = pred[0:data['df_length'][0]]
            pred = pred1

        preds_df = pd.DataFrame(pred)
        print('preds_df', preds_df.shape)
        preds_df.columns = TARGET_COLS
        preds_df['Id'] = data['Id'][0]
        preds_df['Id'] = preds_df['Id'] + '_' + preds_df.index.values.astype(str)

        pred_dfs.append(preds_df)

    print('len preds_df', len(preds))
    preds = pd.concat(pred_dfs)

    print('preds', preds.shape)
    return preds[TARGET_COLS], preds[['Id']]




def inference_loop(params, test_x, ckpt_paths):
    # create model
    models = []

    for c in ckpt_paths:
        model = AmpNet(params).cuda()
        # load model
        model = load_model(model, c)
        model.eval()
        models.append(model)
    # get loaders
    valDataLoader = getDataLoader(params, test_x)

    preds_dfs = []
    id_df = None
    for m in models:
        df, id_df = test_epoch(m, valDataLoader)
        preds_dfs.append(df)

    preds = preds_dfs[0].copy()
    for pred_df in preds_dfs:
        for c in TARGET_COLS:
            preds[c] += pred_df[c]

    for c in TARGET_COLS:
        preds[c] /= len(models)

    preds['Id'] = id_df
    return preds




def main():

    set_seed(SEED)

    fog_data = pd.read_csv(os.path.join(INFERENCE_DATA,'test_meta.csv'))

    print(fog_data)

    # select model checkpoints to use. recommended: the last checkpoint saved for each fold. it averages the predictions done by all of the models.
    ckpt_paths = [os.path.join(r"D:\non-dataset-spec-w-dl",MLFLOW_RUN_NAME,"wavenet_4096-fold0_8.pth"),
                  os.path.join(r"D:\non-dataset-spec-w-dl",MLFLOW_RUN_NAME,"wavenet_4096-fold1_10.pth"),
                  os.path.join(r"D:\non-dataset-spec-w-dl",MLFLOW_RUN_NAME,"wavenet_4096-fold2_11.pth"),
                  os.path.join(r"D:\non-dataset-spec-w-dl",MLFLOW_RUN_NAME,"wavenet_4096-fold3_19.pth"),
                  os.path.join(r"D:\non-dataset-spec-w-dl",MLFLOW_RUN_NAME,"wavenet_4096-fold4_22.pth")]

    submission = inference_loop(hparams, fog_data, ckpt_paths)

    print(submission[-4000:])

    submission[['Id', 'StartHesitation', 'Turn', 'Walking']].to_csv(os.path.join(OUTPUT_PATH, 'submission_'+MLFLOW_RUN_NAME+'.csv'), index=False)


if __name__ == "__main__":
    main()