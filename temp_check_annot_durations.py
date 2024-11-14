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



path_annotations = r"N:\Projects\FoG@Home\Data\Annotations\Annotations\Daily living"


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


annot_durations = []
short_annots = []
dur_sum = 0

min_dur = datetime.timedelta(seconds=10000000)

for subject in tqdm(os.listdir(path_annotations), total=len(os.listdir(path_annotations))):
    if subject=='Subject 1' or subject=='Subject 3':
        continue

    sub_annot_durations = []

    for day in os.listdir(os.path.join(path_annotations, subject)): # todo start only with first two days. ask valeri to add subjects.

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

            annot_duration = annot_end_time - annot_start_time

            if annot_duration < datetime.timedelta(minutes=2):
                short_annots.append((subject, day))

            sub_annot_durations.append(annot_duration)
            # dur_sum+=annot_duration
            if min(sub_annot_durations)<min_dur:
                min_dur = min(sub_annot_durations)


    annot_durations.append(sub_annot_durations)

a=3