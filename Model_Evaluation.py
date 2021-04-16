"""
Author:         Mahsa Khalili
Date:           2021 April 15th
Purpose:        This Python script classifies terrains using IMU data.
"""

# IMPORT LIBRARIES
import os
import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.metrics import plot_confusion_matrix

# preprocessing
from sklearn.model_selection import train_test_split

# Evaluation metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

# import/export
import joblib

# DEFINITIONS
# IMPORT USER DATA ['All', 'Jamie', 'Keenan', 'Kevin', 'Mahsa']
USER = 'All'

# WINDOW SIZE
SAMP_SLICE = 1024  # good practice for fft analysis: better to have a value that's a power of 2

# target label, choices are ['Label', 'Relabeled', 'Relabeled2', 'Label_IO']
LABELS = 'Label_IO'

# Features for classification ['TimeFeats', 'FreqFeats', 'FFTs', 'PSDs', 'TimeFreqFeats', 'FFTPSDFeats', 'AllFeats']
CLASS_FEATS = 'TimeFreqFeats'

# original labeled data
Label_ls = ['No_motion', 'Concrete', 'Carpet', 'Linoleum', 'Asphalt', 'Sidewalk', 'Grass', 'Gravel']

# relabel data to reflect important classification goals
Relabeled_ls = ['No_motion', 'Indoor', 'Asphalt', 'Sidewalk', 'Grass', 'Gravel']

# relabel data to reflect important classification goals
Relabeled2_ls = ['No_motion', 'Indoor', 'Asphalt-Sidewalk', 'Grass', 'Gravel']

# labeles for indoor/outdooor classification
Label_IO_ls = ['No_motion', 'Indoor', 'Outdoor']

# list of possible classification targets
target_label_list = ['Label', 'Label_IO', 'Relabeled', 'Relabeled2']

# dictionary of target label names & associated list of labels
target_label_dic = \
    {'Label': Label_ls, 'Label_IO': Label_IO_ls, 'Relabeled': Relabeled_ls, 'Relabeled2': Relabeled2_ls}

# feature extracted datasets
features = ['TimeFeats', 'FreqFeats', 'FFTs', 'PSDs']

# original measurements
axes = ['X Accel', 'Y Accel', 'Z Accel', 'X Gyro', 'Y Gyro', 'Z Gyro']

CURR_PATH = os.path.abspath('.')
TRAIN_PATH = os.path.join(CURR_PATH, 'processed_data', 'Train')  # get path to train datasets
TEST_PATH = os.path.join(CURR_PATH, 'processed_data', 'Test')  # get path to test datasets


def get_label_num(list_, label_):
    """Get the integer terrain value of a given label"""
    for i, label in enumerate(list_):
        if label in label_:
            return i


def get_label_name(list_, label_num):
    """Get the name associated with a terrain integer"""
    return list_[label_num]


def combine_datasets(datasets):
    dataset_list = []

    for dataset_ in datasets:
        df_labels = dataset_[['Label', 'Label_IO']]
        dataset_list.append(dataset_.drop(columns=['Label', 'Label_IO']))

    df_concat = pd.concat(dataset_list, axis=1)
    df_combined = pd.concat([df_labels, df_concat], axis=1)

    return df_combined


def dic_feats(path_):
    datasets_dic = {}

    # create a dictionary of all featured datasets
    for feature in features:
        filename = os.path.join(path_, feature + '.csv')

        # Read data and update current user dictionary
        df = pd.read_csv(filename)
        datasets_dic.update({feature: df})

    datasets_dic.update({'AllFeats': combine_datasets(list(datasets_dic.values()))})
    datasets_dic.update({'TimeFreqFeats': combine_datasets([datasets_dic['TimeFeats'], datasets_dic['FreqFeats']])})
    datasets_dic.update({'FFTPSDFeats': combine_datasets([datasets_dic['FFTs'], datasets_dic['PSDs']])})

    return datasets_dic


def relabel_io(dic_df):
    # Add labels according to the relabeled list (group indoor terrains)
    for label, dataset in dic_df.items():

        # add new column
        dataset.insert(0, 'Relabeled', np.nan)

        # relabel according to relabeled list
        for i in range(8):
            if i == 0:
                dataset.loc[dataset.Label == i, 'Relabeled'] = 0
            elif 0 < i < 4:
                dataset.loc[dataset.Label == i, 'Relabeled'] = 1
            elif i >= 4:
                dataset.loc[dataset.Label == i, 'Relabeled'] = i - 2

        dataset['Relabeled'] = dataset['Relabeled'].astype(int)

        dic_df.update({label: dataset})

        # add new column
        dataset.insert(0, 'Relabeled2', np.nan)

        # relabel according to relabeled list
        for i in range(8):
            if i < 3:
                dataset.loc[dataset.Relabeled == i, 'Relabeled2'] = i
            elif i >= 3:
                dataset.loc[dataset.Relabeled == i, 'Relabeled2'] = i - 1

        dataset['Relabeled2'] = dataset['Relabeled2'].astype(int)

        dic_df.update({label: dataset})

    return dic_df


def create_train_test(dataset, target_label, test_size):
    """
    input: get dataset and desired target label to perform classification
    output: train/test splits
    """
    df = dataset.copy()

    target_dic = {}

    # store all label columns in a dictionary
    for label in target_label_list:
        target_dic.update({label: df.pop(label)})

    # split data
    x_tr, x_te, y_tr, y_te = \
        train_test_split(df, target_dic[target_label], test_size=test_size, random_state=0)

    return x_tr, x_te, y_tr, y_te


def model_evaluation(model_):

    # accuracy
    print('Accuracy score = {:0.2f}'.format(model_.score(X_test, y_test) * 100, '.2f'))
    accuracy_ = model_.score(X_test, y_test) * 100

    # predicted labels
    y_pred = model_.predict(X_test)

    # f1_score
    f_score = f1_score(y_test, y_pred, average='macro') * 100
    print('F1-score = {:0.2f}'.format(f_score))

    # recall
    recall_ = recall_score(y_test, y_pred, average='macro') * 100
    print('Recall score = {:0.2f}'.format(recall_))

    return accuracy_, f_score, recall_


def plt_confusion_matrix(clmodel, xtest, ytest, class_labels):
    # confusion matrix
    title = "Normalized confusion matrix"

    disp = plot_confusion_matrix(clmodel, xtest, ytest, display_labels=target_label_dic[class_labels],
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 xticks_rotation=45,
                                 values_format='.2f')
    disp.ax_.set_title(title)
    disp.ax_.grid(False)
    disp.figure_.set_size_inches(12, 12)
    plt.show()


def sfs_eval(model_,xtest):
    # get best features
    best_feats_idx = model_.best_estimator_['selector'].k_feature_idx_
    best_feats = list(xtest.columns[list(best_feats_idx)].values.tolist())
    print('\nBest features: \n{}'.format(best_feats))

    # plotting feature selection characteristics
    plot_sfs(model_.best_estimator_['selector'].get_metric_dict(), kind='std_err', figsize=(12, 5))
    plt.title('Sequential Forward Selection (w. StdErr)')
    plt.grid(b=True, which='major', axis='both')
    plt.show()


# Importing Datasets
train_datasets_dic = dic_feats(TRAIN_PATH)
test_datasets_dic = dic_feats(TEST_PATH)

# add new labels to the dataset
train_datasets_dic = relabel_io(train_datasets_dic)
test_datasets_dic = relabel_io(test_datasets_dic)

# create train set
X_train, X_test_, y_train, y_test_ = create_train_test(train_datasets_dic[CLASS_FEATS], LABELS, test_size=1)

# create test set
X_train_, X_test, y_train_, y_test = create_train_test(test_datasets_dic[CLASS_FEATS], LABELS,
                                                       test_size=len(test_datasets_dic[CLASS_FEATS]) - 1)

# import Model
results_path = os.path.join(CURR_PATH, 'Results')  # create directory to save results
timestr = "20210416-133058"  # get current time to use for saving models/figures
model_path = os.path.join(results_path, timestr)  # create directiry for the current time
model_name = os.path.join(model_path, 'model.joblib')
rf_model = joblib.load(model_name)  # load model

# Part 4 - Evaluation

plt_confusion_matrix(rf_model, X_test, y_test, LABELS)
# get parameters of the best model
print("Best estimator via GridSearch \n", rf_model.best_estimator_)

# Part 5 - GridSearch Summary
# Mean cross-validated score of the best_estimator
print('Best feature combination had a CV accuracy of:', rf_model.best_score_)

# Model best parameters
print("Best parameters via GridSearch \n", rf_model.best_params_)

# Part 6 - Feature selection summary
sfs_eval(rf_model, X_test)

print("\n SUCCESS!!!")
