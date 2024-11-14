import torch
print(torch.cuda.is_available())
print(torch.version.cuda,torch.__version__)
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import torch.cuda.amp as amp
import random
from sklearn.model_selection import GroupKFold
import re
import librosa



MLFLOW_RUN_NAME = 'fortunate-cod-99'

GPU = torch.cuda.is_available()
WAV_SIZE=2000
TIMES_TRAIN=8
IS_MIXED_PRECISION = True
TARGET_COLS = ['StartHesitation', 'Turn', 'Walking']
SEED=42
SHUFFLE_BEFORE_SPLITTING = False


INPUT_PATH_NP = r'N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\data_np' # folder of preprocessed data (output of data_creation)
INPUT_PATH = r'N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl' # folder containing meta files
OUTPUT_PATH = r'N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl'

meta_data_all = {"defog": r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\defog_metadata.csv",
              "tdcs": r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\tdcsfog_metadata.csv",
              "fog_at_home": r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\fog@home_metadata.csv"}

subjects_all = {key: set(pd.read_csv(file)["Subject"].tolist()) for key, file in meta_data_all.items()}


hparams = {
    # Optional hparams
    "backbone": 'wavenet_4096',
    "learning_rate": [5e-4],
    "batch_size": 8,
    "num_workers": 0,
    "val_sanity_checks": 0,
    "fast_dev_run": False,
    "output_path": f"",
    "gpu": torch.cuda.is_available(),
    'div_factor':10,
    'final_div_factor':20,
    'wav_size':WAV_SIZE,
    'is_mixed_precision':IS_MIXED_PRECISION,
    'seed': SEED,
    'shuffle_before_splitting': SHUFFLE_BEFORE_SPLITTING,



}



def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms = True
    random.seed(0)
    np.random.seed(0)


def count_csv_rows(file_path):
    with open(file_path, 'r') as f:
        # Subtract 1 to exclude the header row
        return sum(1 for line in f) - 1


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
        row = self.data.iloc[idx]
        wav = np.load(INPUT_PATH_NP + fr'\{row.Id}_sig.npy')

        wav = wav/40.

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

        sample = {"wav": wav, "Id": row.Id, "Subject": row.Subject, 'act_len': act_len, 'nchunk': nchunk, 'rem_size': rem_size}

        return sample


def getDataLoader(params, val_x):
    val_dataset = FOGDataset(df=val_x, transforms=None) #is_train here means the type of data is originally training data (already resampled, scaled to g, etc.).

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

        pred_dataset = [dataset for dataset, id_list in subjects_all.items() if data['Subject'][0] in id_list]
        assert len(pred_dataset) == 1

        if pred_dataset[0] == 'tdcs':
            preds = []
            for clss in range(3):
                preds.append(librosa.resample(pred[:, clss].astype(np.float32), orig_sr=100, target_sr=128))

            pred = np.stack(preds, axis=1)
            pred = np.clip(pred, 0, 1)

            df_length = count_csv_rows(os.path.join(r"E:/kaggle dataset/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog", data['Id'][0]+'.csv'))

            pred1 = np.zeros((df_length, 3))
            pred1[0:df_length] = pred[0:df_length]
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





set_seed(SEED)

metadata = pd.read_csv(f'{INPUT_PATH}/meta_data.csv')
if SHUFFLE_BEFORE_SPLITTING:
    # metadata = metadata.sample(frac=1, random_state=0).reset_index(drop=True)
    unique_subjects = metadata.Subject.unique()
    # np.random.seed(0)  # For reproducibility
    np.random.seed(1)
    np.random.shuffle(unique_subjects)
    # Re-map the shuffled subjects back to the metadata
    subject_mapping = {old: new for new, old in enumerate(unique_subjects)}
    metadata['Shuffled_Subject'] = metadata['Subject'].map(subject_mapping)
else:
    metadata['Shuffled_Subject'] = metadata['Subject']

kf = GroupKFold(n_splits=5)
for i, (train_index, test_index) in enumerate(
        kf.split(metadata.Id, metadata.Medication, groups=metadata.Shuffled_Subject)):
    metadata.loc[test_index, 'fold'] = i


version = '6'
oof_preds = []
for fn in [1, 4, 0, 2, 3]:

    set_seed()

    val = metadata[metadata.fold == fn]
    tr = metadata[metadata.fold != fn]

    print('FOLD', fn, 'Train', tr.shape, 'Val', val.shape)

    # check fold model name
    directory_path = r"D:\non-dataset-spec-w-dl\{}".format(MLFLOW_RUN_NAME)
    file_pattern = "wavenet_4096-fold{}_.+\.pth".format(fn)
    model_files = os.listdir(os.path.join(directory_path, 'selected_for_oof_preds'))
    # Loop through the files and match the pattern with fn
    for file_name in model_files:
        if re.match(file_pattern, file_name):
            # Extract the integer that replaces the "8" (or any number in that place)
            match = re.search(r"wavenet_4096-fold{}_(\d+)\.pth".format(fn), file_name)
            if match:
                fold_epoch = int(match.group(1))
                break

    ckpt_paths = [os.path.join(r"D:\non-dataset-spec-w-dl",MLFLOW_RUN_NAME, 'selected_for_oof_preds',"wavenet_4096-fold"+str(fn)+"_"+str(fold_epoch)+".pth")]

    prediction = inference_loop(hparams, val, ckpt_paths)
    oof_preds.append(prediction)
oof_preds_all = pd.concat(oof_preds, axis=0, ignore_index=True)
oof_preds_all[['id_without_sample', 'sample']] = oof_preds_all['Id'].str.split('_', expand=True)
oof_preds_all['sample'] = oof_preds_all['sample'].astype(int)
oof_preds_all = oof_preds_all.sort_values(by=['id_without_sample', 'sample'])
oof_preds_all = oof_preds_all.set_index('Id').drop(columns=['id_without_sample', 'sample'])

oof_preds_all.to_csv(os.path.join(OUTPUT_PATH, MLFLOW_RUN_NAME + '_oof_predictions.csv'))

