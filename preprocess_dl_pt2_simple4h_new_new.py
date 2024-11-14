import pandas as pd
import os
import numpy as np
import scipy.io
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback



# script for reducing data #
# takes 4 hours per subject (morning, afternoon, evening, night). makes sure labels are 0 for lying segments>5min.

skip_onpar = 0

input_path = r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\dl_preprocessed"
output_path = r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl"
meta_info = pd.read_excel(r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\daily_living_metadata_moreinfo2.xlsx")
defog_axivity_path = {"1": r"N:\Projects\DeFOG\AX6\TLV", "2": r"N:\Projects\DeFOG\AX6\KUL"}


hour_conv_factor = 100*60*60
file_list = os.listdir(input_path)




def find_windows_info(arr, window_size):
    # Split the array into non-overlapping windows of size x
    windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window_size)[::window_size]

    # Calculate the sum of each window
    window_sums = windows.sum(axis=1)

    highest_sum_index = np.argsort(window_sums)[-1]
    second_highest_sum_index = np.argsort(window_sums)[-2]
    lowest_sum_index = np.argmin(window_sums)
    median_sum_index = np.argsort(window_sums)[len(window_sums) // 2] # median

    return {
        "highest_sum_index": (highest_sum_index*window_size,(highest_sum_index+1)*window_size), #highest %fog
        "second_highest_sum_index": (second_highest_sum_index*window_size,(second_highest_sum_index+1)*window_size),
        "lowest_sum_index": (lowest_sum_index*window_size,(lowest_sum_index+1)*window_size), #lowest %fog
        "median_sum_index": (median_sum_index*window_size,(median_sum_index+1)*window_size)
    }





for file in tqdm(file_list):
    lying_zeroing = 1
    if skip_onpar:
        project = meta_info[meta_info["Id"] == file.split(".")[0]]["project"].values[0]
        if project == "onpar":
            continue

    data = pd.read_parquet(os.path.join(input_path, file), engine="pyarrow")

    # get lying vector
    try:

        project = meta_info[meta_info["Id"]==file.split(".")[0]]["project"].values[0]

        if project=="defog":

            axivity_id = meta_info[meta_info["Id"] == file.split(".")[0]]["id_in_project"].values[0]
            if "c1" in axivity_id.lower():
                lying_vec_path = [f for f in os.listdir(os.path.join(defog_axivity_path["1"], "Posture\Lying")) if axivity_id.lower() in f.lower() and 'lyingvec' in f.lower()]
                lying_vec = scipy.io.loadmat(os.path.join(defog_axivity_path["1"], "Posture\Lying", lying_vec_path[0]))["Lying_Vec"]
            elif "c2" in axivity_id.lower():
                lying_vec_path = [f for f in os.listdir(os.path.join(defog_axivity_path["2"], "Posture\Lying")) if axivity_id.lower() in f.lower() and 'lyingvec' in f.lower()]
                lying_vec = scipy.io.loadmat(os.path.join(defog_axivity_path["2"], "Posture\Lying", lying_vec_path[0]))["Lying_Vec"]
            else:
                raise("defog id error")


            lying_vec = np.squeeze(lying_vec)

            if len(lying_vec) > len(data.index):
                if len(data)!= 60480000:
                    a=4
                data = data.head(len(lying_vec))
                # keep only 1st day
                lying_vec = lying_vec[:hour_conv_factor * 24]

            elif len(lying_vec) < len(data.index):
                lying_zeroing = 0
                print(file+': lying vec sync problem.')

        elif project=="onpar": #not pd
            data[['StartHesitation', 'Turn', 'Walking']] = 0
            lying_zeroing = 0



        # keep only 1st day
        data = data.head(hour_conv_factor*24)

        if lying_zeroing:
            ### zero labels for lying segments of 5 minutes or longer
            samples_5min = 100*60*5
            # lying segments
            padded_lying = np.pad(lying_vec.astype(np.int32), (1, 1), mode='constant', constant_values=0)
            diff = np.diff(padded_lying)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            durations = ends - starts
            starts = starts[durations>=samples_5min]
            ends = ends[durations>=samples_5min]
            assert len(starts) == len(ends)
            if starts.size > 0:
                # fig, ax = plt.subplots(figsize=(10, 6))
                # ax.plot(data["AccV"], label='accv')
                # for index in starts:
                #     ax.axvline(x=index, color='r', linewidth=0.8, label='starts' if index == starts[0] else "")
                # for index in ends:
                #     ax.axvline(x=index, color='b', linewidth=0.8, label='ends' if index == ends[0] else "")
                # plt.savefig('lying_marked_'+file.split('.')[0]+'.pdf', format='pdf')
                # plt.close(fig)
                for idx in range(len(starts)):
                    data.loc[starts[idx]:ends[idx], ['StartHesitation', 'Turn', 'Walking']] = 0


    except:
        traceback.print_exc()

    # keep 1 fog hour per day
    keep_indices = find_windows_info(data[['StartHesitation', 'Turn', 'Walking']].max(axis=1),
                                     hour_conv_factor)  # select based on fog incidence
    data_4h = pd.DataFrame()
    for start, end in keep_indices.values():
        data_4h = pd.concat([data_4h, data.iloc[start:end]])
    data_4h.reset_index(drop=True, inplace=True)
    # print(project)
    # print("turn fog: "+str(sum(data_4h["Turn"])))

    data_4h.to_parquet(os.path.join(output_path, "dl_preprocessed_3", file), index=False, compression='snappy')
