import os
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score
from tqdm import tqdm
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1000000000
mpl.use('TkAgg')
import matplotlib.pyplot


mlflow_run_name = "fortunate-cod-99"
save_results_file = False
test_path = r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\data_test"


def sort_by_id_and_timepoint(df):

    df[['Id_part1', 'Id_part2']] = df['Id'].str.split('_', expand=True)
    df['Id_part2'] = df['Id_part2'].astype(int)
    df = df.sort_values(by=['Id_part1', 'Id_part2']).reset_index(drop=True)
    df = df.drop(columns=['Id_part1', 'Id_part2'])

    return df





if not os.path.exists(os.path.join(test_path, 'results', mlflow_run_name)):
    os.makedirs(os.path.join(test_path, 'results', mlflow_run_name))

submission = pd.read_csv(r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\submission_"+mlflow_run_name+".csv")
submission = sort_by_id_and_timepoint(submission)

# load and concat label files
label_files = os.listdir(os.path.join(test_path, "labels"))

all_labels = pd.DataFrame()
for j, label_file in tqdm(enumerate(label_files), total=len(label_files)):
    id = label_file.split(".")[0]
    file = pd.read_csv(os.path.join(test_path, "labels", label_file))
    file.reset_index(drop=True, inplace=True)
    file = file.drop(columns=['Time'])
    file.insert(0, 'Id', id + '_' + file.index.astype(str))
    all_labels = pd.concat([all_labels, file], ignore_index=True)

all_labels = sort_by_id_and_timepoint(all_labels)

assert all_labels['Id'].equals(submission['Id'])

# ###
# #look at the samples of one specific id
# all_labels['id'] = all_labels['Id'].str.split('_', expand=True)[0]
# all_labels = all_labels[all_labels.id=='bdcff4be3a']
# all_labels = all_labels.drop(columns=['id'])
# submission['id'] = submission['Id'].str.split('_', expand=True)[0]
# submission = submission[submission.id=='bdcff4be3a']
# submission = submission.drop(columns=['id'])
# ###

y_true = all_labels.iloc[:,1:]
y_pred = submission.iloc[:,1:]

y_true["FOG"] = y_true.max(axis=1)
y_pred["FOG"] = y_pred.max(axis=1)

# #######################
# # plot pred and labels together
# label = y_true["FOG"]
# pred = y_pred["FOG"]
#
# mpl.pyplot.figure(figsize=(10, 6))
# mpl.pyplot.plot(label, label='Label')
# mpl.pyplot.plot(pred, label='Pred')
#
# # Adding legend
# mpl.pyplot.legend()
#
# # Adding labels and title (optional)
# mpl.pyplot.xlabel('Index')
# mpl.pyplot.ylabel('Values')
#
# # Show the plot
# mpl.pyplot.show()
# #######################

average_precisions=[]
results = pd.DataFrame(columns=["type","F1 score","Accuracy","Precision","Recall","Specificity"])

fig1, ax1 = mpl.pyplot.subplots()
fig2, ax2 = mpl.pyplot.subplots()


for i, type in enumerate(y_true.columns):
    precision, recall, thresholds = precision_recall_curve(y_true[type], y_pred[type])
    fpr, tpr, roc_thresholds = roc_curve(y_true[type], y_pred[type])

    roc_auc = roc_auc_score(y_true[type], y_pred[type])

    average_precision = np.mean(precision)
    if type!='FOG':
        average_precisions.append(average_precision)

    # find best threshold
    pr_array = np.column_stack((precision,recall))
    distances = np.linalg.norm(pr_array - np.array([1, 1]), axis=1)
    closest_index = np.argmin(distances)
    best_precision = precision[closest_index]
    best_recall = recall[closest_index]
    best_threshold = thresholds[closest_index]
    best_f1 = (2*best_precision*best_recall)/(best_precision+best_recall)

    # ##from ryan's code
    # f_scores = [2*p * r / ((p + r) if p + r > 0 else 1) for p, r in zip(precision, recall)]
    # maxf = np.argmax(f_scores)
    # maxf_val=f_scores[maxf]

    # find specificity and accuracy @ best threshold
    pred_binary = (y_pred[type] > best_threshold).astype('int64')
    tn, fp, fn, tp = confusion_matrix(y_true[type],pred_binary).ravel()
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    scores = {'type': type,'F1 score': best_f1, 'Accuracy':  accuracy, 'Precision': best_precision, 'Recall': best_recall, 'Specificity': specificity}
    results = results._append(scores, ignore_index=True)
    legend=y_true.columns.copy().tolist()
    legend[-1]="All FOG"


    ax1.plot(recall, precision)
    if type=='FOG':
        ax1.set_xlabel('Recall', fontsize=16)
        ax1.set_ylabel('Precision', fontsize=16)
        ax1.tick_params(axis='both',labelsize=12)
        # ax1.set_title('#'+str(competition_rank)+ ' ' + submission_name + ' Precision-Recall Curve LB: '+ "%.3f" % round(user_score,3)+' F1: '+ "%.3f" % round(best_f1,3))
        ax1.set_title(mlflow_run_name + ' Precision-Recall Curve MAP: '+ "%.3f" % round(np.mean(average_precision),3)+' F1: '+ "%.3f" % round(best_f1,3), fontsize=16)
        ax1.grid(True)
        fig1.savefig(os.path.join(test_path, 'results', mlflow_run_name, 'p-r_curve1.png'))


    ax2.plot(fpr, tpr)
    if type=='FOG':
        ax2.set_xlabel('False Positive Rate (1-Specificity)', fontsize=16)
        ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=16)
        ax2.tick_params(axis='both',labelsize=12)
        # ax2.set_title('#'+str(competition_rank)+ ' ' + submission_name + ' ROC Curve LB: '+ "%.3f" % round(user_score,3)+' F1: '+ "%.3f" % round(best_f1,3))
        ax2.set_title(mlflow_run_name + ' ROC Curve AUC: '+ "%.3f" % round(roc_auc,3), fontsize=16)
        ax2.grid(True)
        #ax2.legend(legend, loc='upper right')
        fig2.savefig(os.path.join(test_path, 'results', mlflow_run_name, 'roc_curve1.png'))

fig1, ax1 = mpl.pyplot.subplots()
ax1.plot(precision, recall)

if save_results_file:
    results.to_csv(os.path.join(test_path, 'results', mlflow_run_name, 'metrics.png'), index=False)

average_precision_from_curve = np.mean(average_precision) #todo why does it not match the leaderboard-> might be because of interpolation
print(average_precision_from_curve)

a=4