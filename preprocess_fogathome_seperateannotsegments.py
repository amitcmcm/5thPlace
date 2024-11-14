import pandas as pd
import glob
import os
import mat73
from tqdm import tqdm
import scipy.io
import numpy as np
import datetime
import random
import string
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1000000000
mpl.use('TkAgg')
import matplotlib.pyplot

'''
this script is similar to preprocess_fogathome, 
but files are saved separately for different segments of annotation



*for fog at home, the original annotation files are not always consistent and tend to have issues, 
which is why i suggest using the already-preprocessed files (they are inside the folder fog@home_preprocessed). 
it is possible that more labeled fog@home data has been added, 
but i don't know if the amount of new data justifies dealing with the inconsistent label files and issues that could arise. 
if you decide to do it anyway, you will have to look at the fog at home preprocessing code carefully and 
look at the different cases and the data files they are meant to handle. 
you will also need access to the label files in the path listed below (ask eran). 

'''


path_axivity = r"N:\Projects\FoG@Home\Data\Axivity\axivity analysis\Mat files"
path_annotations = r"N:\Projects\FoG@Home\Data\Annotations\Annotations\Daily living"
output_path = r"N:/Projects/ML competition project/winner uploads/5th InnerVoice/local/non-dataset-spec-w-dl"

weartime_check = False



def convert_time(time):
    if "-" in time:
        mon_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        year = int(time.split(' ')[0].split('-')[-1])
        month = mon_map[time.split(' ')[0].split('-')[1]]
        day = int(time.split(' ')[0].split('-')[0])
        hour = int(time.split(' ')[1].split(':')[0])
        minute = int(time.split(' ')[1].split(':')[1])
        second = int(time.split(' ')[1].split(':')[-1])
    else:
        year = int(time[:4])
        month = int(time[4:6])
        day = int(time[6:8])
        hour = int(time[8:10])
        minute = int(time[10:12])
        second = int(time[12:14])
    conv_time = datetime.datetime(year, month, day, hour, minute, second)
    return conv_time


def check_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))




if not os.path.exists(os.path.join(output_path, "fog@home_preprocessed")):
    os.makedirs(os.path.join(output_path, "fog@home_preprocessed"))


meta_exc_fogathome = pd.read_csv(r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\meta_data.csv")
existing_ids = meta_exc_fogathome.Id
existing_subs = meta_exc_fogathome.Subject

meta_fogathome = []
characters = string.ascii_lowercase + string.digits


for subject in tqdm(os.listdir(path_annotations), total=len(os.listdir(path_annotations))):
    if subject=='Subject 1' or subject=='Subject 3':
        continue
    # load axivity acceleration data
    subject_filename = 'back_0' + str(subject.split(' ')[1] if len(subject.split(' ')[1])==1 else 'back_' + subject.split(' ')[1])
    axivity = mat73.loadmat(os.path.join(path_axivity, 'acc', subject_filename + '.mat'))

    if weartime_check:
        weartime_vec = scipy.io.loadmat(os.path.join(r"N:\Projects\FoG@Home\Data\Axivity\axivity analysis\Wear Time","WearTime_"+subject_filename+".mat"))
        # over 5% nonwear time
        if np.sum(weartime_vec['WearTime'])<0.95*len(axivity['New_Data']): # todo check this test
            print('weartime issue for '+subject)
            continue
    else:
        print('weartime check option is off (assuming manual check was ok, or that first two days are full)')

    data = pd.DataFrame(axivity['New_Data'][:,:3])
    data.columns = ['AccV', 'AccML', 'AccAP']

    # load axivity record time data
    info = scipy.io.loadmat(os.path.join(path_axivity, 'info', subject_filename + '_info.mat'))['fileinfo']
    start_rec, end_rec = convert_time(info[0][0][5][0][0][1][0]), convert_time(info[0][0][6][0][0][1][0]) #info[0][0][5][0][0][0][0][0], info[0][0][6][0][0][0][0][0]
    time_offsets = np.arange(len(data)) * 0.01
    time_vec = pd.DatetimeIndex(start_rec + pd.to_timedelta(time_offsets, unit='s'))#[start_rec + pd.Timedelta(seconds=i * 0.01) for i in range(len(data))] #np.arange(start_rec, start_rec + len(data) * 0.01, 0.01)
    data.insert(0, 'Time', time_vec)

    # add label columns
    data[['sit-to-stand', 'stand-to-sit', 'turning1-l', 'turning2-l', 'turning1-r', 'turning2-r', 'walking','label']] = np.zeros((data.shape[0], 8)) # todo handle starthesi and walking (might not appear in motor_situtation and will need to be found by looking at which event(=activity) the fog event happened in. also, see if there are situations where fog continues to the next event (ask valerie?)
    del axivity

    # add fog annotations
    # sanity check (day order)
    assert check_increasing(os.listdir(os.path.join(path_annotations, subject)))

    for day in ['Day 1', 'Day 2', 'Day1', 'Day2']: #os.listdir(os.path.join(path_annotations, subject)): # todo start only with first two days. ask valeri to add subjects.

        # get daily annotation files
        annotation_files = glob.glob(os.path.join(path_annotations, subject, day, '*.txt'))
        annotation_files = [s for s in annotation_files if "provoking" not in s.lower()]

        for file in annotation_files:

            # get start time of annotations
            annot_start_time = file.split('\\')[-1].split('_')[2]
            annot_start_time = convert_time(annot_start_time)

            # get end time
            annot_end_time = file.split('\\')[-1].split('_')[3].split('.')[0]
            if any(char.isalpha() for char in annot_end_time):
                annot_end_time = annot_end_time.split('-')[0]
            annot_end_time = convert_time(annot_end_time)

            # update label flag
            assert (annot_end_time-annot_start_time > datetime.timedelta(minutes=5))
            data.loc[(data.Time >= annot_start_time) & (data.Time <= annot_end_time), 'label'] = 1

            # load current annotation file
            annotations = pd.read_csv(file, delimiter='\t')

            # include only fog annotations
            fog_annotations = annotations[(annotations.FOG.notna()) & (annotations.FOG != 'Interrupt')]

            # add fog events to data array
            for i, r in fog_annotations.iterrows():
                start_split = [float(j) for j in r["Begin Time - hh:mm:ss.ms"].split(':')]
                end_split = [float(j) for j in r["End Time - hh:mm:ss.ms"].split(':')]
                duration_split = [float(j) for j in r["Duration - hh:mm:ss.ms"].split(":")]
                fog_start = annot_start_time + datetime.timedelta(milliseconds=start_split[2]*1000, minutes=start_split[1], hours=start_split[0])

                # sanity check
                fog_end = annot_start_time + datetime.timedelta(milliseconds=end_split[2]*1000, minutes=end_split[1], hours=end_split[0])
                assert datetime.timedelta(milliseconds=duration_split[2]*1000, minutes=duration_split[1], hours=duration_split[0])==fog_end-fog_start

                # if motor situation is mentioned in the fog label itself
                if str(r.Motor_situation).lower() in ['sit-to-stand', 'stand-to-sit', 'turning1-l', 'turning2-l', 'turning1-r', 'turning2-r', 'walking']:
                    data.loc[(data.Time >= fog_start) & (data.Time <= fog_end), r.Motor_situation.lower()] = 1
                # else find inside which motor situation(s) the fog is happening
                else:
                    if pd.isna(r.Motor_situation):
                        fog_bt = r['Begin Time - ss.msec']
                        fog_et = r['End Time - ss.msec']
                        annotations_exc_fog = annotations[pd.isna(annotations.FOG)]
                        annotations_exc_fog.reset_index(drop=True, inplace=True)
                        # start
                        if len(annotations_exc_fog.index[annotations_exc_fog['Begin Time - ss.msec'] == fog_bt])>0:
                            start_of_segment_of_fog = annotations_exc_fog.index[annotations_exc_fog['Begin Time - ss.msec'] == fog_bt][0]
                        else:
                            try:
                                idx_after_fogstart = annotations_exc_fog.index[annotations_exc_fog['Begin Time - ss.msec'] > fog_bt][0]
                                start_of_segment_of_fog = idx_after_fogstart - 1
                            except: # if fog happens at the last task annotated
                                if fog_bt>annotations_exc_fog.loc[len(annotations_exc_fog)-1,'Begin Time - ss.msec']:
                                    start_of_segment_of_fog = len(annotations_exc_fog) - 1

                        # end
                        if len(annotations_exc_fog.index[annotations_exc_fog['End Time - ss.msec'] == fog_et])>0:
                            end_of_segment_of_fog = annotations_exc_fog.index[annotations_exc_fog['End Time - ss.msec'] == fog_et][0]
                        else:
                            try:
                                end_of_segment_of_fog = annotations_exc_fog.index[annotations_exc_fog['End Time - ss.msec'] > fog_et][0]
                            except: # if fog happens at the last task annotated
                                if fog_et>annotations_exc_fog.loc[len(annotations_exc_fog)-1,'End Time - ss.msec']:
                                    end_of_segment_of_fog = len(annotations_exc_fog) - 1

                        # check if its more than one row in annotations (=more than one motor situation ovarlapping with the fog event)
                        if start_of_segment_of_fog==end_of_segment_of_fog:
                            fog_motor_sit = annotations_exc_fog.Motor_situation[start_of_segment_of_fog].lower()
                            if fog_motor_sit in ['sit-to-stand', 'stand-to-sit', 'turning1-l', 'turning2-l', 'turning1-r', 'turning2-r', 'walking']:
                                data.loc[(data.Time >= fog_start) & (data.Time <= fog_end), fog_motor_sit] = 1
                            else:
                                print(subject+' '+day+' '+fog_motor_sit)
                                raise ('unknown motor sit.')
                        else:
                            fog_segments = annotations_exc_fog[start_of_segment_of_fog:end_of_segment_of_fog+1] #take rows relevant to the fog event
                            fog_segments.reset_index(drop=True, inplace=True)

                            for j, fog_segment in fog_segments.iterrows(): #make sure that idx is not reset
                                if fog_segment.Motor_situation.lower() in ['sit-to-stand', 'stand-to-sit', 'turning1-l', 'turning2-l', 'turning1-r', 'turning2-r', 'walking']:
                                    if j==0:
                                        if fog_segment['End Time - ss.msec']<fog_bt:
                                            continue
                                        seg_end_split = [float(x) for x in fog_segment["End Time - hh:mm:ss.ms"].split(':')]
                                        seg_end_time = annot_start_time + datetime.timedelta(milliseconds=seg_end_split[2] * 1000, minutes=seg_end_split[1], hours=seg_end_split[0])
                                        data.loc[(data.Time >= fog_start) & (data.Time <= seg_end_time), fog_segment.Motor_situation.lower()] = 1 #check
                                    elif j<len(fog_segments)-1:
                                        seg_start_split = [float(x) for x in fog_segment["Begin Time - hh:mm:ss.ms"].split(':')]
                                        seg_end_split = [float(x) for x in fog_segment["End Time - hh:mm:ss.ms"].split(':')]
                                        seg_start_time = annot_start_time + datetime.timedelta(milliseconds=seg_start_split[2] * 1000, minutes=seg_start_split[1], hours=seg_start_split[0])
                                        seg_end_time = annot_start_time + datetime.timedelta(milliseconds=seg_end_split[2] * 1000, minutes=seg_end_split[1], hours=seg_end_split[0])
                                        data.loc[(data.Time >= seg_start_time) & (data.Time <= seg_end_time), fog_segment.Motor_situation.lower()] = 1 #check
                                    else:
                                        seg_start_split = [float(x) for x in fog_segment["Begin Time - hh:mm:ss.ms"].split(':')]
                                        seg_start_time = annot_start_time + datetime.timedelta(milliseconds=seg_start_split[2] * 1000, minutes=seg_start_split[1], hours=seg_start_split[0])
                                        data.loc[(data.Time >= seg_start_time) & (data.Time <= fog_end), fog_segment.Motor_situation.lower()] = 1 #check
                                else:
                                    print(subject + ' ' + day + ' ' + fog_segment.Motor_situation.lower())
                    else:
                        print(subject+' '+day+' '+r.Motor_situation.lower())
                        #raise ('unknown motor sit.')

    # arrange label columns
    data[['turning1-l', 'turning2-l', 'turning1-r', 'turning2-r', 'stand-to-sit']] = data[['turning1-l', 'turning2-l', 'turning1-r', 'turning2-r', 'stand-to-sit']].astype(bool)
    turn_col = (data['turning1-l'] | data['turning2-l'] | data['turning1-r'] | data['turning2-r'] | data['stand-to-sit'])
    data.insert(data.columns.get_loc('turning1-l') , 'Turn', turn_col.astype('int64'))
    # # sanity check
    # data[['turning1-l', 'turning2-l', 'turning1-r', 'turning2-r', 'stand-to-sit']].astype(float).plot()
    # mpl.pyplot.legend(title='Columns')
    # mpl.pyplot.xlabel('Index')
    # mpl.pyplot.ylabel('Values')
    # mpl.pyplot.savefig('plot2.png')
    # mpl.pyplot.close()
    # #
    data = data.rename(columns={'sit-to-stand': 'StartHesitation'}) #todo eran said i can check if it's at the start. and to check with valerie how she tagged in general.
    data['StartHesitation'] = data['StartHesitation'].astype('int64')
    data['Walking'] = data['walking'].astype('int64')
    data = data.drop(columns=['turning1-l', 'turning2-l', 'turning1-r', 'turning2-r', 'stand-to-sit', 'walking'])



    # # sanity check
    # binary_label = data[['StartHesitation', 'Turn', 'Walking']].max(axis=1)
    # fog_indices = data['Turn'].loc[data['Turn'] == 1].index.tolist()
    # fig, (ax1, ax2) = mpl.pyplot.subplots(2, 1, figsize=(10, 8))
    # # Plot the 3 columns in the first subplot
    # data.loc[fog_indices[2000]-5000:fog_indices[2000]+5000, ['AccV', 'AccML', 'AccAP']].plot(ax=ax1)
    # # data.loc[0:13000000, ['AccV', 'AccML', 'AccAP']].plot(ax=ax1)
    # # data.loc[:, ['AccV', 'AccML', 'AccAP']].plot(ax=ax1)
    # ax1.set_xlabel('Index')
    # ax1.set_ylabel('acc')
    # ax1.legend()
    # # Plot the Series in the second subplot
    # ax2.plot(binary_label.loc[fog_indices[2000]-5000:fog_indices[2000]+5000], marker='o')
    # # ax2.plot(binary_label.loc[0:13000000], marker='o')
    # # ax2.plot(binary_label, marker='o')
    # ax2.set_xlabel('Index')
    # ax2.set_ylabel('fog')
    # # Display the plot
    # mpl.pyplot.tight_layout()
    # mpl.pyplot.savefig(r'N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\fog@home_preprocessed\sub4.pdf',format='pdf')
    # mpl.pyplot.close(fig)
    # #mpl.pyplot.show()
        ####



    # save annotation segments separately:

    data['annot_segment'] = (data['label'] != data['label'].shift()).cumsum()
    consecutive_segments = data[data['label'] == 1]
    grouped_segments = consecutive_segments.groupby('annot_segment')
    # generate random subject id
    sub_id = ''.join(random.choices(characters, k=6))
    while sub_id in existing_subs:
        sub_id = ''.join(random.choices(characters, k=6))
    # save the files
    for segment, group in grouped_segments:
        # generate random file id
        id = ''.join(random.choices(characters, k=10))
        while id in existing_ids:
            id = ''.join(random.choices(characters, k=10))
        # add to meta data
        sub_meta = pd.Series(data=[id, sub_id, 0], index=['Id', 'Subject', 'upsample'])
        meta_fogathome.append(sub_meta)
        group.drop(columns=['annot_segment', 'label']).to_parquet(os.path.join(output_path, "fog@home_preprocessed", id + ".parquet"), index=False, compression='snappy')


meta_fogathome = pd.DataFrame(meta_fogathome)
meta_fogathome.to_csv(os.path.join(output_path, 'fog@home_metadata.csv'), index=False)

a=3

#todo plot a few examples of the annotations i created to see if they make sense
# todo check that acc is in the right direction in all axes

