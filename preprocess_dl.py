import pandas as pd
import os
from tqdm import tqdm


# pseudo labels are based on predictions by 1st, 3rd and 5th place models, with decision based on the higher ranked model in case of "tie".
plabels_path = r"N:\Projects\ML competition project\winner uploads\Daily living analysis\Output_Merged"
data_path = r"N:\Projects\ML competition project\Kaggle dataset (from ryan with test sets)\archive\Daily Living"
output_path = r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl"
events_path = r"E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\events.csv"

def one_hot_classes(labels):
    one_hot = pd.DataFrame(0, index=labels.index, columns=['StartHesitation', 'Turn', 'Walking'])
    one_hot.loc[labels == 1, 'StartHesitation'] = 1
    one_hot.loc[labels == 2, 'Turn'] = 1
    one_hot.loc[labels == 3, 'Walking'] = 1
    return one_hot


events = pd.read_csv(events_path)

mapping_file = pd.read_excel(r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\AxivityFileMapping.xlsx")
mapping_data = mapping_file[['KaggleFileName', 'NFOGQ']].copy()
mapping_data.rename(columns={'KaggleFileName': 'Id'}, inplace=True)
mapping_data['Id'] = mapping_data['Id'].str.replace('.mat', '')

dl_files = os.listdir(plabels_path)


# ####temp#######
# already_done = os.listdir(r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\dl_preprocessed")
# already_done = [s.split('.')[0] for s in already_done]
# dl_files = [s.split('_')[1].split('.')[0] for s in dl_files]
# set_a = set(already_done)
# set_b = set(dl_files)
# # Find missing items in list_b that are not in list_a
# missing_items = list(set_b - set_a)
# modified_strings = ['FoGByClass_' + s + '.csv' for s in missing_items]
# #############

faulty_files =[]
events_rows = []
for file in tqdm(dl_files):
    id = file.split('_')[1]
    id = id.split('.')[0]
    data = pd.read_parquet(os.path.join(data_path, id+".parquet"), engine='pyarrow')

    if mapping_data[mapping_data['Id']==id]['NFOGQ'].values>0:
        labels = pd.read_csv(os.path.join(plabels_path, file))
        labels = labels["Class_1StartH_2Turn_3Walk"]
        labels = one_hot_classes(labels)
        if labels.shape[0] < data.shape[0]:
            data = data.head(labels.shape[0])
        elif labels.shape[0] > data.shape[0]:
            labels = labels.head(data.shape[0])
    else:
        labels = pd.DataFrame(0, index=range(data.shape[0]), columns=['StartHesitation', 'Turn', 'Walking'])

    try:
        assert(labels.shape[0]==data.shape[0])
        preprocessed_file = pd.concat([data, labels], axis=1)
        preprocessed_file.to_parquet(os.path.join(output_path, "dl_preprocessed", id+".parquet"), index=False, compression='snappy')
        events_rows.append({'Id': id, 'Init': 999, 'Completion': 999, 'Type': 'Class', 'Kinetic': 999})
    except:
        faulty_files.append(id)

new_events_file = pd.concat([events, pd.DataFrame(events_rows)], ignore_index=True)
new_events_file.to_csv(os.path.join(output_path, "events.csv"), index=False)

