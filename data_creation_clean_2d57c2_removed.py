import pandas as pd
import librosa
import numpy as np
import os
from tqdm import tqdm


np.random.seed(42)
test_per = 0.1 # define test split percent

INPUT_PATH = r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction'
OUTPUT_PATH = r'N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl'
EVENTS_FILE_PATH = r'N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl'

INPUT_DATA = {"defog": {"path": r"E:/kaggle dataset/tlvmc-parkinsons-freezing-gait-prediction/train/defog/",
                         "meta": r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\defog_metadata.csv", #must have "Id" and "Subject" columns
                         "fs": 100,
                         "scale_to_g": True},
              "tdcs": {"path": r"E:/kaggle dataset/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/",
                       "meta": r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\tdcsfog_metadata.csv", #must have "Id" and "Subject" columns
                        "fs": 128,
                        "scale_to_g": False},
              # "daily_living": {"path": r"N:/Projects/ML competition project/winner uploads/5th InnerVoice/local/non-dataset-spec-w-dl/dl_preprocessed_3/",
              #        "meta": r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\daily_living_metadata.csv", #must have "Id" and "Subject" columns
              #        "fs": 100,
              #        "scale_to_g": True},
              "fog_at_home": {"path": r"N:/Projects/ML competition project/winner uploads/5th InnerVoice/local/non-dataset-spec-w-dl/fog@home_preprocessed/",
                    "meta": r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\fog@home_metadata.csv",
                    "fs": 100,
                    "scale_to_g": True}
 }





g0=9.80665
if not os.path.exists(os.path.join(OUTPUT_PATH, f'data_np')):
    os.makedirs(os.path.join(OUTPUT_PATH, f'data_np'))


events = pd.read_csv(f'{EVENTS_FILE_PATH}/events.csv')
events = events[~events.Type.isnull()]

missing_signals = []
final_meta = []
final_test_meta = []
for j, data_source in enumerate(INPUT_DATA):
    meta = pd.read_csv(INPUT_DATA[data_source]["meta"])
    if data_source=='tdcs':
        meta = meta[meta.Subject != '2d57c2']
    if data_source != "daily_living" and data_source != "fog_at_home":
        has_event = meta[meta['Id'].isin(events['Id'])]
        subjects = pd.unique(has_event.Subject)
        subs_for_test = np.random.choice(subjects, int(np.round(len(subjects) * test_per)), replace=False)
    for i,r in tqdm(meta.iterrows(), total=len(meta)):
        if r.Id == 'c261b476e8':
            continue
        if r.Id in(events.Id.values):
            try:
                try:
                    data = pd.read_csv(f'{INPUT_DATA[data_source]["path"]}{r.Id}.csv')
                except:
                    data = pd.read_parquet(f'{INPUT_DATA[data_source]["path"]}{r.Id}.parquet', engine='pyarrow')
                signal = data[['AccV', 'AccML', 'AccAP']].values
                target = data[['StartHesitation', 'Turn', 'Walking']].astype(np.float32).values

                # scaling to g
                if INPUT_DATA[data_source]["scale_to_g"]:
                    signal = signal * g0

                # resampling to 100Hz
                if INPUT_DATA[data_source]["fs"] != 100:
                    signal_resample = []
                    target_resample = []
                    for i in range(3):
                        signal_resample.append(librosa.resample(signal[:, i], orig_sr=INPUT_DATA[data_source]["fs"], target_sr=100))
                    signal = np.stack(signal_resample, axis=1)
                    for i in range(3):
                        target_resample.append(librosa.resample(target[:, i], orig_sr=INPUT_DATA[data_source]["fs"], target_sr=100))
                    target = np.stack(target_resample, axis=1)

                if r.upsample: #to sample defog x4
                    signal = np.tile(signal, (4, 1))
                    target = np.tile(target, (4, 1))

                if r.Subject not in subs_for_test:
                    np.save(os.path.join(OUTPUT_PATH, f'./data_np/{r.Id}_sig.npy'), signal)
                    np.save(os.path.join(OUTPUT_PATH, f'./data_np/{r.Id}_tgt.npy'), target)

                    final_meta.append(r)
                else:
                    #no need to save test np files, but its possible to save them here using similar commands to the train ones (but different folder if needed)
                    final_test_meta.append(r)
            except:
                missing_signals.append(r.Id)


final_meta = pd.DataFrame(final_meta)
final_meta.to_csv(os.path.join(OUTPUT_PATH, 'meta_data.csv'), index=False)
final_test_meta = pd.DataFrame(final_test_meta)
final_test_meta.to_csv(os.path.join(OUTPUT_PATH, 'meta_data_test.csv'), index=False)