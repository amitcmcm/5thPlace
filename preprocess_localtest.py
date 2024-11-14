import pandas as pd
from tqdm import tqdm
import os
import glob


META_PATH = r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\meta_data_test.csv"
DATA_PATH = r"E:/kaggle dataset/tlvmc-parkinsons-freezing-gait-prediction/train"
OUTPUT_PATH = r'N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl'


def find_file(folder_path, file_name):
    # Use glob to search for files matching the pattern
    search_pattern = os.path.join(folder_path, '**', file_name + '.*')
    matched_files = glob.glob(search_pattern, recursive=True)

    # Test to ensure there is only one such file
    if len(matched_files) == 0:
        raise FileNotFoundError(f"No file named '{file_name}' found in the subfolders.")
    elif len(matched_files) > 1:
        raise FileExistsError(f"More than one file named '{file_name}' found: {matched_files}")

    # Return the full path of the matched file
    return matched_files[0]


if not os.path.exists(os.path.join(OUTPUT_PATH, f'data_test')):
    os.makedirs(os.path.join(OUTPUT_PATH, f'data_test'))
    os.makedirs(os.path.join(OUTPUT_PATH, f'data_test','acc'))
    os.makedirs(os.path.join(OUTPUT_PATH, f'data_test','labels'))


test_meta = pd.read_csv(META_PATH)

for i,r in tqdm(test_meta.iterrows(), total=len(test_meta)):
    id = r.Id
    file_path = find_file(DATA_PATH,id)
    data = pd.read_csv(file_path)
    if 'defog' in file_path:
        data = data[(data.Valid==True) & (data.Task==True)]
        data = data.reset_index(drop=True)
        test_meta.loc[i, 'fs'] = 100
        test_meta.loc[i, 'scale_to_g'] = True
    else:
        test_meta.loc[i, 'fs'] = 128
        test_meta.loc[i, 'scale_to_g'] = False
    acc = data[['Time', 'AccV', 'AccML', 'AccAP']]
    labels = data[['Time', 'StartHesitation', 'Turn', 'Walking']]

    acc.to_csv(os.path.join(OUTPUT_PATH, f'data_test','acc',id+'.csv'), index=False)
    labels.to_csv(os.path.join(OUTPUT_PATH, f'data_test','labels',id+'.csv'), index=False)

    test_meta.loc[i,'filename'] = os.path.join(OUTPUT_PATH, f'data_test','acc',id+'.csv')


test_meta.to_csv(os.path.join(OUTPUT_PATH, f'data_test', 'test_meta.csv'), index=False)
