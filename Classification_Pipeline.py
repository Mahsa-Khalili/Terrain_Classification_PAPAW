"""
Author:         Mahsa Khalili
Date:           2021 April 15th
Purpose:        This Python script classifies terrains using IMU data.
"""

# IMPORT LIBRARIES
import os
import pandas as pd
import numpy as np
import time
import csv

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.metrics import plot_confusion_matrix

# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# feature selection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# pipeline 
from sklearn.pipeline import Pipeline

# grid search
from sklearn.model_selection import GridSearchCV

# ML models
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
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
                dataset.loc[dataset.Label == i, 'Relabeled'] = i-2

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


def clf_pipeline(X_train, y_train):

    # cross validation
    cv = 5

    # classifier to use for feature selection
    feat_selection_clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=0)

    # create pipeline
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('selector', SFS(estimator=feat_selection_clf,
                                      k_features=(10, X_train.shape[1]),
                                      forward=True,
                                      floating=False,
                                      scoring='accuracy',
                                      cv=cv,
                                      n_jobs=-1)),
                     ('classifier', RandomForestClassifier())])

    # set parameter grid
    param_grid = [
        {'selector': [SFS(estimator=feat_selection_clf)],
         'selector__estimator__n_estimators':[50],
         'selector__estimator__max_depth':[10]},


        {'classifier':[RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state = 0)],
         'classifier__n_estimators':[50],
         'classifier__max_depth':[30]}]

    # dictionary of evaluation scores
    scoring = {
        'precision_score': make_scorer(precision_score, average='macro'),
        'recall_score': make_scorer(recall_score, average='macro'),
        'accuracy_score': make_scorer(accuracy_score),
        'f1_score': make_scorer(f1_score, average='macro')}

    grid = GridSearchCV(pipe,
                        param_grid,
                        scoring=scoring,
                        n_jobs=-1,
                        refit='accuracy_score',
                        cv=cv,
                        verbose=0,
                        return_train_score=True)

    grid.fit(X_train, y_train)

    return grid


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
                                                       test_size=len(test_datasets_dic[CLASS_FEATS])-1)

# run the classification pipeline
model_ = clf_pipeline(X_train, y_train)

# get parameters of the best model
print("Best estimator via GridSearch \n", model_.best_estimator_)


# Export Model
results_path = os.path.join(CURR_PATH, 'Results')  # create directory to save results
timestr = time.strftime("%Y%m%d-%H%M%S")  # get current time to use for saving models/figures
model_path = os.path.join(results_path, timestr)  # create directiry for the current time
os.makedirs(model_path)
model_name = os.path.join(model_path, 'model.joblib')
joblib.dump(model_, model_name)  # dump model


print("\n SUCCESS!!!")
# # ## Part 4 - Evaluation
#
# # ### 4.1. Evaluation score
#
# # In[27]:
#
#
# # accuracy
# print('Accuracy score = {:0.2f}'.format(model_.score(X_test, y_test) * 100, '.2f'))
#
# y_pred = model_.predict(X_test)
#
# # f1_score
# f_score = f1_score(y_test, y_pred, average = 'macro')* 100
# print('F1-score = {:0.2f}'.format(f_score))
#
# # recall
# recall_ = recall_score(y_test, y_pred, average = 'macro')* 100
# print('Recall score = {:0.2f}'.format(recall_))
#
#
# # ### 4.2. Confusion matrix
#
# # In[28]:
#
#
# # confusion matrix
# title = "Normalized confusion matrix"
#
# disp = plot_confusion_matrix(model_, X_test, y_test, display_labels=target_label_dic[LABELS],
#                              cmap=plt.cm.Blues,
#                              normalize='true',
#                              xticks_rotation = 45,
#                              values_format = '.2f')
# disp.ax_.set_title(title)
# disp.ax_.grid(False)
# disp.figure_.set_size_inches(12,12)
#
# # save confusion matrix
# fig_name = 'confusion_matrix.jpg'
# fig_name = os.path.join(path_, fig_name)
# plt.savefig(fig_name)
#
#
# # ### 4.3. Analyze computational performance
#
# # In[29]:
#
#
# # computational performance
# X_Test_test = X_test[:1].copy()
#
# # method 1
# get_ipython().run_line_magic('timeit', 'y_pred = model_.predict(X_Test_test)')
#
# # method 2
# time1 = time.time()
# y_pred = model_.predict(X_Test_test)
# time2 = time.time()
# print('prediction time: {} ms'.format((time2-time1)*1000))
#
#
# # ## Part 5 - GridSearch Summary
#
# # ### 5.1. Model best score
#
# # In[30]:
#
#
# # Mean cross-validated score of the best_estimator
# print('Best feature combination had a CV accuracy of:', model_.best_score_)
#
#
# # ### 5.2. Model best parameters
#
# # In[31]:
#
#
# print("Best parameters via GridSearch \n", model_.best_params_)
#
#
# # ### 5.3. Visualize GridSearch results
# # ##### Source: https://www.kaggle.com/grfiv4/displaying-the-results-of-a-grid-search
#
# # In[32]:
#
#
# def GridSearch_table_plot(grid_clf, param_name,
#                           num_results=4,
#                           negative=True,
#                           graph=True,
#                           display_all_params=False):
#
#     '''Display grid search results
#
#     Arguments
#     ---------
#
#     grid_clf           the estimator resulting from a grid search
#                        for example: grid_clf = GridSearchCV( ...
#
#     param_name         a string with the name of the parameter being tested
#
#     num_results        an integer indicating the number of results to display
#                        Default: 15
#
#     negative           boolean: should the sign of the score be reversed?
#                        scoring = 'neg_log_loss', for instance
#                        Default: True
#
#     graph              boolean: should a graph be produced?
#                        non-numeric parameters (True/False, None) don't graph well
#                        Default: True
#
#     display_all_params boolean: should we print out all of the parameters, not just the ones searched for?
#                        Default: True
#
#     Usage
#     -----
#
#     GridSearch_table_plot(grid_clf, "min_samples_leaf")
#
#                           '''
#     from matplotlib      import pyplot as plt
#     from IPython.display import display
#     import pandas as pd
#
#     clf = grid_clf.best_estimator_
#     clf_params = grid_clf.best_params_
#     if negative:
#         clf_score = -grid_clf.best_score_
#     else:
#         clf_score = grid_clf.best_score_
#
#     clf_stdev = grid_clf.cv_results_['std_test_accuracy_score'][grid_clf.best_index_]
#     cv_results = grid_clf.cv_results_
#
#     print("best parameters: {}".format(clf_params))
#     print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
#     if display_all_params:
#         import pprint
#         pprint.pprint(clf.get_params())
#
#     # pick out the best results
#     # =========================
#     scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_accuracy_score')
#
#     best_row = scores_df.iloc[0, :]
#     if negative:
#         best_mean = -best_row['mean_test_accuracy_score']
#     else:
#         best_mean = best_row['mean_test_accuracy_score']
#     best_stdev = best_row['std_test_accuracy_score']
#     best_param = best_row['param_' + param_name]
#
#     # display the top 'num_results' results
#     # =====================================
#     display(pd.DataFrame(cv_results)             .sort_values(by='rank_test_accuracy_score').head(num_results))
#
#     # plot the results
#     # ================
#     scores_df = scores_df.sort_values(by='param_' + param_name)
#
#     if negative:
#         means = -scores_df['mean_test_accuracy_score']
#     else:
#         means = scores_df['mean_test_accuracy_score']
#     stds = scores_df['std_test_accuracy_score']
#     params = scores_df['param_' + param_name]
#
#     # plot
#     if graph:
#         plt.figure(figsize=(8, 8))
#         plt.errorbar(params, means, yerr=stds)
#
#         plt.axhline(y=best_mean + best_stdev, color='red')
#         plt.axhline(y=best_mean - best_stdev, color='red')
#         plt.plot(best_param, best_mean, 'or')
#
#         plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
#         plt.xlabel(param_name)
#         plt.ylabel('Score')
#         plt.show()
#
#
# # In[33]:
#
#
# GridSearch_table_plot(model_, "selector__estimator__n_estimators", negative=False)
#
#
# # ### 5.4. Gridsearch evaluation - multiple scores
# # ##### source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
#
#
#
# def score_evaluation(grid, param):
#
#     # plotting the results
#     results = grid.cv_results_
#
#     scoring = {
#             'precision_score': make_scorer(precision_score, average='macro'),
#             'recall_score': make_scorer(recall_score, average='macro'),
#             'accuracy_score': make_scorer(accuracy_score),
#             'f1_score':make_scorer(f1_score, average='macro')}
#
#     plt.figure(figsize=(10, 8))
#     plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
#               fontsize=16)
#
#     plt.xlabel(param)
#     plt.ylabel("Score")
#
#     ax = plt.gca()
#     ax.set_xlim(0, 100)
#     ax.set_ylim(0.0, 1)
#
#     # Get the regular numpy array from the MaskedArray
#     X_axis = np.array(results['param_selector__estimator__n_estimators'].data, dtype=float)
#
#     for scorer, color in zip(sorted(scoring), ['g', 'k','r','b']):
#         for sample, style in (('train', '--'), ('test', '-')):
#             sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
#             sample_score_std = results['std_%s_%s' % (sample, scorer)]
#             ax.fill_between(X_axis, sample_score_mean - sample_score_std,
#                             sample_score_mean + sample_score_std,
#                             alpha=0.1 if sample == 'test' else 0, color=color)
#             ax.plot(X_axis, sample_score_mean, style, color=color,
#                     alpha=1 if sample == 'test' else 0.7,
#                     label="%s (%s)" % (scorer, sample))
#
#         best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
#         best_score = results['mean_test_%s' % scorer][best_index]
#
#         # Plot a dotted vertical line at the best score for that scorer marked by x
#         ax.plot([X_axis[best_index], ] * 2, [0, best_score],
#                 linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
#
#         # Annotate the best score for that scorer
#         ax.annotate("%0.2f" % best_score,
#                     (X_axis[best_index], best_score + 0.005))
#
#     plt.legend(loc="best")
#     plt.grid(False)
#     plt.show()
#
#
# # In[35]:
#
#
# score_evaluation(model_, "classifier__max_depth")
#
#
# # ### 5.5. Gridsearch report
#
# # In[36]:
#
#
# results = model_.cv_results_
# # results
#
#
# # ## Part 6 - Feature selection summary
#
# # ### 6.1. Feature selection score & best features
#
#
#
# # get prediction score of best selected features
# print('\nFeature selection score: {}'.format(model_.best_estimator_['selector'].k_score_))
#
#
#
#
# # get best features
# best_feats_idx = model_.best_estimator_['selector'].k_feature_idx_;
#
# best_feats = list(X_test.columns[list(best_feats_idx)].values.tolist())
#
# print('\nBest features: \n{}'.format(best_feats))
#
# # save a list of selected features
# filename = 'selected_features.csv'
# filename = os.path.join(path_, filename)
#
# with open(filename, "w") as f:
#     writer = csv.writer(f)
#     writer.writerows([c.strip() for c in r.strip(', ').split(',')] for r in best_feats)
#
#
#
#
#
# # test shape of the feature transformed dataframe
# X_test_transformed = model_.best_estimator_['selector'].transform(X_test)
# print('shape of the transformed dataset with best features: {}'.format(X_test_transformed.shape))
#
#
# # ### 6.2. Visualize feature selection scores
#
#
#
#
# # plotting feature selection characteristics
# plot_sfs(model_.best_estimator_['selector'].get_metric_dict(), kind='std_err', figsize=(12,5))
# plt.title('Sequential Forward Selection (w. StdErr)')
# plt.grid(b=True, which='major', axis='both')
#
# # save confusion matrix
# fig_name = 'feature_selection.jpg'
# fig_name = os.path.join(path_, fig_name)
# plt.savefig(fig_name)
#
