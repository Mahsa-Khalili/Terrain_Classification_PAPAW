"""
Author:         Mahsa Khalili
Date:           2021 April 15th
Purpose:        This Python script prepare IMU data for terrain classification.
"""

# Import relevant modules
import os
import glob
import time
from datetime import datetime

import pandas as pd
import numpy as np

import random
from random import randrange
from scipy.fft import fft, fftfreq
from scipy.signal import sosfiltfilt, butter, welch

import matplotlib.pyplot as plt

# DEFINITIONS

# IMPORT USER DATA ['All', 'Jamie', 'Keenan', 'Kevin', 'Mahsa']
USER = 'All'

# TERRAIN TYPES
terrains = ['Concrete', 'Carpet', 'Linoleum', 'Asphalt', 'Sidewalk', 'Grass', 'Gravel']

# MANEUVER TYPE
movements = ['F8', 'Donut', 'Straight']

# SENSOR PLACEMENT
placements = ['Left', 'Right', 'Middle', 'Synthesis']

# FFT / PSD DATA
transforms = ['FFT', 'PSD']

# LIST OF STANDARD COLUMN NAMES
std_columns = ['X Accel', 'Y Accel', 'Z Accel', 'X Gyro', 'Y Gyro', 'Z Gyro', 'Run Time', 'Epoch Time']

# LIST OF COLUMN NAMES W/O TIME FEATURES
data_columns = ['X Accel', 'Y Accel', 'Z Accel', 'X Gyro', 'Y Gyro', 'Z Gyro']

# UNUSED COLUMNS OF IMPORTED IMU MEASUREMENTS
unused_columns = ['Time Received', 'Timestamp', 'Pitch (Deg)', 'Roll (Deg)', 'Heading (Deg)',
                  'MagX', 'MagY', 'MagZ', 'ACCELEROMETER XY (m/s²)']

# Gravitational acceleration on flat surface
Gz = 9.81

# sampling frequency
f_samp_9250 = 100  # 9250's frame's module
f_samp_6050 = 300  # 6050's frame's module
f_samp_wheels = 333  # wheels' modules (not used for PAPAW data)

# WINDOW SIZE
SAMP_SLICE = 1024  # good practice for fft analysis: better to have a value that's a power of 2

# LOW-PASS FILTER CUT-OFF FREQUENCY
CUT_OFF = 20  # Low-pass cut-off frequency (Hz)

# For small float values
EPSILON = 0.00001

# GET USER INPUT - DETERMINE WHETHER TO EXPORT PROCESSED DATA OR NOT
EXPORT_PROCESSED_DATA = input('Do you want to export processed data? True/False? \n')

# GET USER INPUT - CREATING TEST OR TRAIN DATA SET
TeTr = input('Do you want to create a "Test" or "Train" set? \n')

# Relative path of this file
CURR_PATH = os.path.abspath('.')
    
# Glob all csv files in the folder
glob_paths = glob.glob(os.path.join(CURR_PATH, 'imu_data', '*.csv'))

# Remove 9250 9-axis IMU data (for now)
glob_paths = [path for path in glob_paths if '9250' not in path]


def create_path_list(tetr):
    """
    Purpose: creating a homogeneous train/test datasets
    """

    random.seed(0)

    # select test datasets
    test_dataset_paths = []
    for user in ['Jamie', 'Keenan', 'Kevin', 'Mahsa']:
        for terrain in terrains:
            maneuver = movements[randrange(3)]
            path_name = \
                os.path.join(CURR_PATH, 'imu_data') \
                + '\Middle_' + terrain + 'Power' + maneuver + user + '_Module6050.csv'
            test_dataset_paths.append(path_name)

    # select train datasetes
    train_dataset_paths = [path for path in glob_paths if path not in test_dataset_paths]

    if tetr == 'Test':
        paths = test_dataset_paths
    else:
        paths = train_dataset_paths

    return paths


def imported_datasets(tetr):
    """"
    Import datasets as a dictionary of Pandas DataFrames
    """
    datasets = {}  # creating an empty dictionary to store imported dataframes

    dataset_paths = create_path_list(tetr)

    for dataset_path in dataset_paths:

        # Parse labels from file names
        dataset_label = os.path.split(dataset_path)[1].split('.')[0]

        # Read from CSV to Pandas
        dataset_im = pd.read_csv(dataset_path)

        # Drop unused columns
        unused = [unused_column for unused_column in unused_columns if unused_column in dataset_im.columns]
        dataset_im = dataset_im.drop(unused, axis='columns')

        # Rename columns to easier to work with names
        dataset_im.columns = std_columns.copy()

        # Convert timestamps to epoch time in sec
        dataset_im['Epoch Time'] = dataset_im['Epoch Time'].apply(datetime.strptime, args=("%Y-%m-%d %H:%M:%S:%f", ))
        dataset_im['Epoch Time'] = dataset_im['Epoch Time'].apply(datetime.timestamp)

        # Remove gravitational acceleration from Middle frame data
        if 'Middle' in dataset_label:
            # Remove gravity from z component of acceleration,
            dataset_im['Z Accel'] = dataset_im['Z Accel'].apply(lambda x: x - Gz)

        # Datasets are stored in a dictionary
        datasets.update({dataset_label: dataset_im})

    return datasets


def dataset_compare(dataset1, label1, dataset2, label2, y_axis, filtered=False):
    """
    Purpose: Plotting and compare one-axis measurement (e.g., X-acc) of two Pandas datasets over Run Time
    """

    # Plot parameters
    plt.clf()
    fig_c, ax_c = plt.subplots(figsize=(20, 5))
    ax_c.set_xlabel('Run Time ($s$)')
    ax_c.set_title(y_axis + ' for ' + label1 + ' and ' + label2)

    # Add relevant units to y label
    if 'Gyro' in y_axis:
        ax_c.set_ylabel(y_axis + ' ($rad/s$)')
    elif 'Accel' in y_axis:
        ax_c.set_ylabel(y_axis + ' ($m/s^2$)')
    elif 'Vel' in y_axis:
        ax_c.set_ylabel(y_axis + ' ($m/s$)')
    else:
        ax_c.set_ylabel('Unknown')

    # determine whether plotting filtered or raw data
    if filtered:
        legend1 = label1 + '_raw'
        legend2 = label2 + '_filtered'
    else:
        legend1 = label1
        legend2 = label2

    # remove extra text in the labels
    legend1 = legend1.replace('Middle_', '').replace('_Module6050', '')
    legend2 = legend2.replace('Middle_', '').replace('_Module6050', '')

    ax_c.plot(dataset1[label1][y_axis], label=legend1)
    ax_c.plot(dataset2[label2][y_axis], label=legend2)

    ax_c.legend()
    plt.show()


def pd_to_np(pd_datasets, windowed=False):

    """Convert array of Pandas DataFrames to array of 2D NumPy array"""

    np_datasets = {}

    # Convert each dataset individually
    for label, dataset_pd in pd_datasets.items():
        np_dataset = []

        # Return passed datasets if they are already NumPy ndarrays
        if type(dataset_pd) is np.ndarray:
            print('Note: Already a NumPy array!')
            return pd_datasets

        # If windowed, convert individual windows to Pandas
        if windowed:
            for window in dataset_pd:

                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.as_matrix.html
                np_dataset.append(window.to_numpy())
        else:
            np_dataset = dataset_pd.to_numpy()

        np_datasets.update({label: np_dataset})

    return np_datasets


def np_to_pd(np_datasets, windowed=False):

    """
    :param np_datasets:
    :param windowed:
    Convert array of 2D NumPy arrays to Pandas Data Frames
    """

    pd_datasets = {}

    # Convert each dataset individually
    for label, dataset_np in np_datasets.items():
        pd_dataset = []

        # Return passed datasets if they are already Pandas dataframes
        if type(dataset_np) is pd.DataFrame:
            print('Note: Already a Pandas dataframe!')
            return np_datasets

        # Use correct column names
        new_columns = std_columns.copy()

        # If windowed, convert individual windows to Pandas
        if windowed:
            for window in dataset_np:
                pd_dataset.append(pd.DataFrame(data=window, columns=new_columns))
        else:
            pd_dataset = pd.DataFrame(data=dataset_np, columns=new_columns)

        pd_datasets.update({label: pd_dataset})

    return pd_datasets


def get_frequencies(label):

    """Get relevant frequencies for given label based on whether its a frame or wheel dataset"""

    # Sampling frequency (and thus cutoff frequency) varies between frame and wheel modules
    if 'Left' in label or 'Right' in label:
        f_samp = f_samp_wheels
        f_low = CUT_OFF
        f_high = 1

    elif 'Middle' in label:
        f_samp = f_samp_6050  # Sampling frequency
        f_low = CUT_OFF  # a list of cutoff frequency starting at 20Hz
        f_high = 1  # High pass cut-off frequency

    else:
        raise Exception('Unknown label')

    return f_samp, f_low, f_high


def filtering(datasets_dic):
    """
    Implementing a 4th-order butterworth filter & creating a new dictionary with filtered dataframes
    """
    # Filtered datasets dictionary
    filtered_datasets = {}

    # Filter each dataset individually
    for label, raw_dataset in datasets_dic.items():
        # Sampling rates are not consistent across all datasets
        f_samp, f_low, f_high = get_frequencies(label)

        # Nyquist frequency
        nyq = 0.5 * f_samp

        # Get normalized frequencies
        w_low = f_low / nyq

        # Get Butterworth filter parameters (numerator and denominator)
        # The function sosfiltfilt (and filter design using output='sos') should be preferred over filtfilt for most
        # filtering tasks, as second-order sections have fewer numerical problems.
        sos = butter(N=2, Wn=w_low, btype='low', output='sos')

        # Number of columns containing data
        n_data_col = len(data_columns)

        # Filter all the data columns
        dataset_c = np.copy(raw_dataset)

        for n_col in range(n_data_col):

            # Apply a digital filter forward and backward to a signal.
            # The combined filter (filtfilt) has zero phase and a filter order twice that of the original.
            dataset_c[:, n_col] = sosfiltfilt(sos, dataset_c[:, n_col])

        filtered_datasets.update({label: dataset_c})

    return filtered_datasets


def create_nomotion_dataset(datasets):

    """
    extract no-motion data from all datafames
    :param datasets:
    :return: no motion dataset
    """
    nomotion_datasets = {}

    # axis to use as a reference
    thresh_axes = 'X Accel'  # this is more representative of the state of motion rather than z acceleration

    # start/stop threshold of motion for raw & filtered data
    nomotion_stop_threshold = 1.5

    # no-motion data
    for label, dataset_nm in datasets.items():

        # correct one of measurements
        if 'Grass' in label and 'Straight' in label and 'Kevin' in label:
            stop_index = 0

        else:
            # Calculate first instance below threshold and use as the time domain
            stop_index = dataset_nm[dataset_nm[thresh_axes] > nomotion_stop_threshold].index[0]

        dataset_nm = dataset_nm[:stop_index]

        nomotion_datasets.update({label: dataset_nm})

    # concatenating the DataFrames
    df = pd.concat(nomotion_datasets.values(), ignore_index=True)

    return nomotion_datasets, df


def trim_data(datasets):

    """
    function to remove stationary data from all dataframes
    this function was tuned using visual inspection of trimed output
    """

    trim_datasets = {}

    # axis to use as a reference
    thresh_axes = 'X Accel'  # this is more representative of the state of motion rather than z acceleration

    # start/stop threshold of motion for raw & filtered data
    startup_thresh_filt = 1.5

    # get non-stationary signal
    for label, dataset_ in datasets.items():

        # correct one of measurements
        if 'Grass' in label and 'Straight' in label and 'Kevin' in label:
            dataset_ = dataset_.iloc[300:]

        # Calculate first and last instance above threshold and use as the time domain
        start_index = dataset_[dataset_[thresh_axes] > startup_thresh_filt].index[0]
        stop_index = dataset_[dataset_[thresh_axes] > startup_thresh_filt].index[-1]
        dataset_ = dataset_[start_index:stop_index]

        trim_datasets.update({label: dataset_})

    return trim_datasets


def slice_window(datasets, overlap=True):

    """
    WINDOWING
    By using windowing functions, you can further enhance the ability of an FFT to extract spectral data from signals.
    Windowing functions act on raw data to reduce the effects of the leakage that occurs during an FFT of the data.
    Leakage amounts to spectral information from an FFT showing up at the wrong frequencies.
    """
    seg_datasets = {}  # creating a dictionary of lists of sliced (N=512) dataframes
    win_datasets = {}  # creating a dictionary of sliced & windowed dataframes - used for fft/psd only

    # Trim excess data points, then split into short segments
    for label, dataset_all in datasets.items():

        # window size
        window_size = SAMP_SLICE

        # create an empty list of Sliced/Windowed dataframes
        segmented_dataset = []
        windowed_dataset = []

        # Iterate through dataset by half a window at a time and extract segments
        ii = 0

        # whether to have overalping or non-overlaping segments
        if overlap:
            window_slide = int(window_size / 2)  # to create 50% overalping segments
        else:
            window_slide = window_size

        # create Hanning window
        win = np.hanning(window_size)

        while ii + window_size <= len(dataset_all):

            # update the list of segmented dataframes
            segmented_dataset.append(dataset_all[ii:ii + window_size])

            # multiply han window & data segments
            dataset_copy = dataset_all[ii:ii + window_size] * win[:, None]

            # update the list of windowed segments
            windowed_dataset.append(dataset_copy)

            # slide forward
            ii += window_slide

        seg_datasets.update({label: segmented_dataset})
        win_datasets.update({label: windowed_dataset})

    return seg_datasets, win_datasets


def win_plot(dic_1, dic_2):
    """ plotting two consecutive segmented/windowed dataframe """

    j = 0
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    for i in range(30, 32):
        ax[j].plot(dic_1[dataset_labels[0]][i]['X Accel'], label='filtered')
        ax[j].plot(dic_2[dataset_labels[0]][i]['X Accel'], label='windowed')
        ax[j].legend()
        ax[j].set_xlabel('index')
        ax[j].set_ylabel('segmented window #{}'.format(i))
        j += 1

    plt.show()


def fft_transform(datasets):

    """function to create a dictionary of all FFT'd dataframes"""

    fft_dic = {}

    # sample spacing
    samp_time = 1/f_samp_6050

    # Frequency bin centers
    xf = fftfreq(SAMP_SLICE, samp_time)[:SAMP_SLICE//2]

    # Find the FFT of each column of each data window of each dataset
    for label, dataset_list in datasets.items():

        fft_dataset = []

        for window in dataset_list:

            # create an emtpy dataframe
            fft_df = pd.DataFrame(columns=data_columns)

            # calculate fft of each time window
            for i, column in enumerate(data_columns):

                # calculate fft of each column
                y = fft(window[column].values)
                yf = 2.0/SAMP_SLICE * abs(y[0:SAMP_SLICE//2])  # keeping positive frequencies

                # add fft  signal to a dataframe
                fft_df[column] = yf

            # Append the frequency column
            fft_df['frequency'] = xf

            fft_dataset.append(fft_df)

        fft_dic.update({label: fft_dataset})

    return fft_dic


def psd_transform(datasets):

    """function to create a dictionary of all PSD'd dataframes"""

    psd_dic = {}

    # sampling frequency
    fs = f_samp_6050

    # Find the PSD of each column of each data window of each dataset
    for label, dataset_list in datasets.items():

        psd_dataset = []

        for window in dataset_list:

            # create an emtpy dataframe
            psd_df = pd.DataFrame(columns=data_columns)

            # calculate psd of each time window
            for i, column in enumerate(data_columns):

                # calculate psd of each column
                f, pxx_den = welch(window[column].values, fs)

                # add psd'd signal to a dataframe
                psd_df[column] = pxx_den

            # Append the frequency column
            psd_df['frequency'] = f

            psd_dataset.append(psd_df)

        psd_dic.update({label: psd_dataset})

    return psd_dic


def plot_transforms(fft_, psd_):

    """function to visualise a random window of a random dataset from fft/psd transforms"""

    # choose a random dataframe
    i = randrange(len(fft_)-1)
    j = randrange(len(fft_[dataset_labels[i]])-1)

    df_fft = fft_[dataset_labels[i]][j]
    df_psd = psd_[dataset_labels[i]][j]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for col in data_columns:
        ax[0].plot(df_fft['frequency'], df_fft[col], label=col)
        ax[1].plot(df_psd['frequency'], df_psd[col], label=col)

    ax[0].set_xlabel('frequency (hz)')
    ax[1].set_xlabel('frequency (hz)')
    ax[0].set_ylabel('fft')
    ax[1].set_ylabel('psd')
    ax[0].legend()
    ax[1].legend()
    plt.show()


def trim_transforms(datasets, freq_thresh):

    """function to remove excessive frequency bins of fft & psd dataframes"""

    datasets_trimmed = {}

    for label, dataset_list in datasets.items():
        dataset_trimmed_list = []

        for dataset_l in dataset_list:
            # remove frequencies higher than freq_thresh
            dataset_trimmed_list.append(dataset_l[dataset_l['frequency'] <= freq_thresh].copy())

        datasets_trimmed.update({label: dataset_trimmed_list})
    return datasets_trimmed


def l2norm(array):
    """L2 norm of an array"""
    return np.linalg.norm(array, ord=2)


def rms(array):
    """Root mean squared of an array"""
    return np.sqrt(np.mean(array ** 2))


def zcr(array):
    """Zero crossing rate of an array as a fraction of total size of array"""
    # divide by total datapoints in window
    return len(np.nonzero(np.diff(np.sign(array)))[0]) / len(array)


def feature_extraction(datasets, features_dic, freq_domain=False):

    """Extract given features from column of each dataset
       Converts a dictionary of datasets to a nested dictionary where each dataset has its own dictionary
       of axes/directions"""

    feat_datasets = {}

    # Calculate features for each window of each column of each dataset
    for label, dataset_list in datasets.items():

        # will be updated with keys as data columns (e.g., 'X Accel')
        cols_dic = {}

        # Loop over data columns
        for col in data_columns:

            # will be updated with keys as extracted feature names (e.g., 'Mean')
            feats = {}

            if not freq_domain:

                '''Execute a function over all windows'''
                def function_all_windows(function):
                    featured_column = []

                    for window in dataset_list:

                        # update a list of extracted feature for the ith column
                        featured_column.append(function(window[col]))

                    return featured_column

            else:

                '''Alternate definition for frequency functions'''
                def function_all_windows(function):
                    featured_column = []

                    for window in dataset_list:
                        featured_column.append(function(window.iloc[:, -1], window[col]))

                    return featured_column

            # Execute every function over all windows
            for feat_name, feat_func in features_dic.items():

                # apply feature extraction to the ith column for all windows
                feats.update({feat_name: function_all_windows(feat_func)})

            cols_dic.update({col: pd.DataFrame.from_dict(feats)})

        feat_datasets.update({label: cols_dic})

    return feat_datasets


# dictionary of time-domain features to use in feature extraction step
time_features = {'Mean': np.mean, 'Std': np.std,  'Norm': l2norm,
                 'Max': np.amax, 'Min': np.amin, 'RMS': rms, 'ZCR': zcr}


def msf(freqs, psd_amps):
    """Mean square frequency"""
    num = np.sum(np.multiply(np.resize(np.power(freqs, 2), len(psd_amps)), psd_amps))
    denom = np.sum(psd_amps)

    # In case zero amplitude transform is ecountered
    if denom <= EPSILON:
        return EPSILON

    return np.divide(num, denom)


def rmsf(freqs, psd_amps):
    """Root mean square frequency"""
    return np.sqrt(msf(freqs, psd_amps))


def fc(freqs, psd_amps):
    """Frequency center"""
    num = np.sum(np.multiply(np.resize(freqs, len(psd_amps)), psd_amps))
    denom = np.sum(psd_amps)

    # In case zero amplitude transform is ecountered
    if denom <= EPSILON:
        return EPSILON

    return np.divide(num, denom)


def vf(freqs, psd_amps):
    """Variance frequency"""
    return msf(freqs-fc(freqs, psd_amps), psd_amps)


def rvf(freqs, psd_amps):
    """Root variance frequency"""
    return np.sqrt(msf(freqs, psd_amps))


# dictionary of freq-domain features to use in feature extraction step
freq_features = {'RMSF': rmsf, 'FC': fc, 'RVF': rvf}

def append_all_columns(columns, append_tag):

    """
    Combine IMU data from each direction into single dataframes with columns for each feature in each direction
    Append a tag to the end of every column name of a dataframe
    """

    new_columns = []

    for column in columns:
        if append_tag not in column:
            new_columns.append(column + ' ' + append_tag)
        else:
            new_columns.append(column)

    return new_columns


def combine_extracted_columns(datasets):

    """Combined directions (axes) of a featured dataset"""

    combined_datasets = {}

    for label, dataset_dic in datasets.items():

        # Get labels array of first column
        df_combined = pd.DataFrame()

        # Append direction name to feature name and combine everything in one frame
        for col_label, df in dataset_dic.items():
            df_copy = pd.DataFrame(df)

            # Add direction and placement tags
            df_copy.columns = append_all_columns(df.columns, col_label)

            df_combined = df_combined.join(df, how='outer')

        combined_datasets.update({label: df_combined})

    return combined_datasets


def get_transform(_label):

    """Get the transform used for given label"""

    for transform in transforms:
        if transform in _label:
            return transform

    raise Exception('Unknown transform')


def combine_transform_columns(datasets, trans=''):

    """Combined direction (axes) columns for transformed data"""

    combined_datasets = {}

    for label, dataset_tr in datasets.items():

        # Get frequency bins from frequency column of first window
        freq_bins = dataset_tr[0]['frequency'].tolist()

        # Get more parameter for current label
        trans = trans

        # Combine parameters to form columns for new combined DataFrame
        new_cols = [trans + ' {} Hz '.format(round(f_bin, 1)) + d_col for d_col in data_columns for f_bin in freq_bins]

        # Convert windowed arrays into a single array with each window as a row
        new_data = []

        for window in dataset_tr:
            new_row = []
            for d_col in data_columns:
                new_row.extend(window[d_col].tolist())
            new_data.append(new_row)

        # Create new DataFrame
        combined_df = pd.DataFrame(data=new_data, columns=new_cols)
        combined_datasets.update({label: combined_df})

    return combined_datasets


def get_terrain_num(_label):

    """Get the integer terrain value of a given label"""

    for i, terrain in enumerate(terrains):
        if terrain in _label:
            return i+1
        elif 'no_motion' in _label:
            return 0

    raise Exception('Unknown terrain')


def insert_labels(datasets):

    """Add labels to a dataset"""

    # Returns new datasets
    labeled_datasets = {}

    # Add to each dataframe of a dataset
    for label, dataset_la in datasets.items():

        dataset_copy = dataset_la.copy()
        # get terrain label
        terrain_num = get_terrain_num(label)

        # create a list of labels for each dataset
        labels = [terrain_num for _ in range(len(dataset_la))]

        # insert labels (7 I/O terrain + no-motion)
        dataset_copy.insert(0, 'Label', labels)

        if 'no_motion' in label:
            labels_io = [0 for _ in range(len(dataset_la))]
        elif 'Asphalt' in label or 'Sidewalk' in label or 'Grass' in label or 'Gravel' in label:
            labels_io = [2 for _ in range(len(dataset_la))]  # outdoor terrains
        else:
            labels_io = [1 for _ in range(len(dataset_la))]  # indoor terrains

        dataset_copy.insert(0, 'Label_IO', labels_io)
        labeled_datasets.update({label: dataset_copy})

    return labeled_datasets


def combine_datasets(datasets):

    """Combine data from labelled datasets into a single dataframe"""

    return pd.concat(list(datasets.values()), ignore_index=True)


# Part 1 - Importing IMU measurements
raw_datasets = imported_datasets(TeTr)  # create a dictionary of imported datasets
raw_datasets = {label: raw_datasets[label] for label in sorted(raw_datasets.keys())}  # Sorting dictionary
dataset_labels = list(raw_datasets.keys())  # list of all dataset labels/names

# Part 2 - Signal processing
filt_datasets = filtering(raw_datasets)
filt_datasets = np_to_pd(filt_datasets)
noMotion_datasets, noMotion_dataset = create_nomotion_dataset(filt_datasets) # trim no-motion data

all_datasets = trim_data(filt_datasets)  # create a dictionary of non-stationary datasets
all_datasets.update({'no_motion': noMotion_dataset})  # add no-motion dataset to all the other datasets

# drop time columns
for label, dataset in all_datasets.items():
    dataset = dataset.drop(['Run Time', 'Epoch Time'], axis='columns')
    all_datasets.update({label: dataset})

# Part 3 - Slicing and windowing filtered data
segmented_datasets, windowed_datasets = slice_window(all_datasets)  # create a dictionary of segmented/windowed dfs

# Part 4 - Transforms (FFT, PSD)
fft_datasets = fft_transform(windowed_datasets)  # create a dictionary containing lists of fft'd windowed segments
psd_datasets = psd_transform(segmented_datasets)  # create a dictionary containing lists of psd'd windowed segments

# trim fft & psd dataframes to keep useful frequencies only
fft_datasets_trimmed = trim_transforms(fft_datasets, CUT_OFF + 10)
psd_datasets_trimmed = trim_transforms(psd_datasets, CUT_OFF + 10)

# Part 5 - Feature Engineering
time_featured_datasets = feature_extraction(segmented_datasets, time_features)  # dictionary of feature extracted dfs
freq_featured_datasets = feature_extraction(psd_datasets_trimmed, freq_features, freq_domain=True) #dic freq feature extracted dataframes

# Part 6 - Columning, Combination
columned_time_feat_datasets = combine_extracted_columns(time_featured_datasets)  # Take time feature data and combine axes columns
columned_freq_feat_datasets = combine_extracted_columns(freq_featured_datasets)  # Take frequency feature data and axes columns
columned_fft_datasets = combine_transform_columns(fft_datasets_trimmed, 'FFT')  # create columned fft datasets
columned_psd_datasets = combine_transform_columns(psd_datasets_trimmed, 'PSD')  # create columned psd datasets

# Add labels to each of the feature vector types
labeled_time_feat_datasets = insert_labels(columned_time_feat_datasets)
labeled_freq_feat_datasets = insert_labels(columned_freq_feat_datasets)
labeled_fft_datasets = insert_labels(columned_fft_datasets)
labeled_psd_datasets = insert_labels(columned_psd_datasets)

# For each feature vector, combine datasets in two single dataframes
time_feats = combine_datasets(labeled_time_feat_datasets)
freq_feats = combine_datasets(labeled_freq_feat_datasets)
ffts = combine_datasets(labeled_fft_datasets)
psds = combine_datasets(labeled_psd_datasets)

# Part 7 - Exporting Processed Data
processed_path = os.path.join(CURR_PATH, 'processed_data', TeTr)  # Processed data path with power type folder
os.makedirs(processed_path,exist_ok=True)

# Store feature vectors in a dictionary
vector_dict = {'TimeFeats': time_feats, 'FreqFeats': freq_feats, 'FFTs': ffts, 'PSDs': psds}

# Set in notebook parameters at top of notebook
if EXPORT_PROCESSED_DATA:

    # Save each vector and each placement to .csv file
    for vector_name, vector_data in vector_dict.items():

        # Filename using above dictionary
        filename = os.path.join(processed_path, vector_name + '.csv')
        vector_data.to_csv(filename, index=False)

print("\n Success")
