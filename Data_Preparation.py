#!/usr/bin/env python
# coding: utf-8

# # Data Preparation 
# Created by Keenan McConkey 2019.8.12, edited by Mahsa Khalili 2021.01.08
# 
# - **Import notebook dependencies**
# - **Defining notebook variables**
# - **Defining notebook parameters **
# - **Importing IMU data (i.e., raw IMU data collected from the DAQ modules mounted to the wheelchair frame**
#     - A total of 84 datasets are imported: measurements are from 4 users (three maneuvrs each) and 7 indoor/outdoor terrains
# - **Visualizing imported data & comparing kinematic characteristics across different terrains**
# - **Functions to convert between pandas & numpy**
# - **Signal processing**
#     - Filtering IMU data 
#     - Visualizing & comparing raw and filtered data
#     - Trim stationary data (noMotion_dataset) & update filt_datasets dic (non-stationary data only)
#     - Update filtered dataframe including no-motion dataset
# - **Slicing & Windowing dataframes**
# - **FFT/PSD**
# - **Feature enginering**
# - **Columning, combining features, adding labels**
#     - include no motion dataframes
# - **Visualizing extracted features**
# - **Export dataframes to csv**

# ### Import dependencies

# In[1]:


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
from matplotlib import gridspec

from joblib import dump, load


# ### Notebook Variables

# In[2]:


# Import data
# Importing which user's data ['All', 'Jamie', 'Keenan', 'Kevin', 'Mahsa']
# USER = 'All' 
USER = 'All'

# Which measurements to import ['Manual', 'Power', 'Remote']
POWER_TYPE = 'Power' 

# number of datapoints in each segment 
## good practice for fft analysis: better to have a value that's a power of 2
SAMP_SLICE = 1024 

# filter parameters
CUT_OFF = 20 # lowpass cut-off frequency (Hz) 

# whether to create test or train dataset (if test true: create test dataset otherwise create train dataset)
TEST = True

# Whether to export data (set to false when testing the code)
EXPORT_PROCESSED_DATA = False 


# ### Notebook Parameters (Constant values)

# In[3]:


# renamed column names (easier to read column names)
std_columns = ['X Accel', 'Y Accel', 'Z Accel', 'X Gyro', 'Y Gyro', 'Z Gyro', 'Run Time', 'Epoch Time']
data_columns =  ['X Accel', 'Y Accel', 'Z Accel', 'X Gyro', 'Y Gyro', 'Z Gyro'] # removing time-related columns

# Columns not currently used for classification
unused_columns = ['Time Received', 'Timestamp', 'Pitch (Deg)', 'Roll (Deg)', 'Heading (Deg)',
                  'MagX', 'MagY', 'MagZ', 'ACCELEROMETER XY (m/s²)']

# Types of terrains, placements, and transforms used
terrains = ['Concrete', 'Carpet', 'Linoleum', 'Asphalt', 'Sidewalk', 'Grass', 'Gravel']
placements = ['Left', 'Right', 'Middle', 'Synthesis']
transforms = ['FFT', 'PSD_welch']
movements = ['F8', 'Donut', 'Straight']

# gravitional acceleration on flat surface
Gz = 9.81 

# sampling frequency
f_samp_9250 = 100 # 9250's frame's module
f_samp_6050 = 300 # 6050's frame's module
f_samp_wheels = 333 # wheels' modules (not used for PAPAW data)


# ## Part 1 - Importing IMU measurements

# ### 1.1. Defining datasets' directory: currently importing frame's 6050 IMU measurement for PAPAW

# In[4]:


# Power and manual data stored in separate folders
if POWER_TYPE == 'Manual':
    power_type_folder = 'set_manual'
elif POWER_TYPE == 'Power':
    power_type_folder = 'set_power'
elif POWER_TYPE == 'Remote':
    power_type_fodler = 'set_remote'
else:
    raise Exception('Unknown power type!')

# Relative path of this notebook
CURR_PATH = os.path.abspath('.')
    
# Glob all csv files in the folder
glob_paths = glob.glob(os.path.join(CURR_PATH, 'imu_data', power_type_folder, '*.csv'))

# Remove 9250 9-axis IMU data (for now)
glob_paths = [path for path in glob_paths if '9250' not in path]

# Keenan is "default" user so files without any username are assumed to be his
## Will update this later
if USER == 'All':
    dataset_paths = glob_paths
elif USER == 'Keenan' or USER == 'Mahsa' or USER == 'Kevin' or USER == 'Jamie':
    dataset_paths = [path for path in glob_paths if 'Mahsa' not in path]
else:
    raise Exception('Unknown user!')


# In[6]:


''' new approach to create a homogenous & separate train/test datasets '''

random.seed(0)

# select test datasets
test_dataset_paths = []
for user in ['Jamie', 'Keenan', 'Kevin', 'Mahsa']:
    for terrain in terrains:
        maneuver = movements[randrange(3)]
        path_name = os.path.join(CURR_PATH, 'imu_data', power_type_folder) +'\Middle_' + terrain+ 'Power' + maneuver + user + '_Module6050.csv'
        test_dataset_paths.append(path_name)
        
# select train datasetes
train_dataset_paths = [path for path in glob_paths if path not in test_dataset_paths]


# ### 1.2. Parsing data into Pandas

# In[8]:


# Import datasets as a dictionary of Pandas DataFrames

raw_datasets = {} # creating an empty dictionary to store all dataframes

if TEST:
    DATASET_PATHS = test_dataset_paths
else:
    DATASET_PATHS = train_dataset_paths

## To analyze data from the train_dataset_paths only
for dataset_path in DATASET_PATHS:
    
    # Parse labels from filenames
    dataset_label = os.path.split(dataset_path)[1].split('.')[0]    

    # Read from CSV to Pandas
    dataset = pd.read_csv(dataset_path)

    # Drop unused columns
    unused = [unused_column for unused_column in unused_columns if unused_column in dataset.columns]
    dataset = dataset.drop(unused, axis='columns')
    
    # Rename columns to easier to work with names
    dataset.columns = std_columns.copy()

    # Convert timestamps to epoch time in sec
    dataset['Epoch Time'] = dataset['Epoch Time'].apply(datetime.strptime, args=("%Y-%m-%d %H:%M:%S:%f", ))
    dataset['Epoch Time'] = dataset['Epoch Time'].apply(datetime.timestamp)
    
    # Remove gravitational acceleration from Middle frame data
    ## Can't remove from wheel-mounted Left and Right wheel data because they rotate over time
    if 'Middle' in dataset_label:
        # Remove gravity from z component of acceleration, 
        dataset['Z Accel'] = dataset['Z Accel'].apply(lambda x: x - Gz)
    
    # Datasets are stored in a dictionary
    raw_datasets.update({dataset_label: dataset})


# In[9]:


# Sort dictionary according to keys
raw_datasets = {label: raw_datasets[label] for label in sorted(raw_datasets.keys())}

# Save list of keys to variable
dataset_labels = list(raw_datasets.keys())
print('The total number of imported datasets: {}'.format(len(dataset_labels)))


# In[10]:


# Check dataset formatting
print('sample data from the "{}" measurements '.format(dataset_labels[0]))
raw_datasets[dataset_labels[0]].head()


# ## Part 2 - Visualizing Time Domain Data

# ### 2.1. Plotting functions: plot a single Pandas dataset for a given x and y axes

# In[11]:


def plot_one_dataframe(_datasets, dataset_name, x_axis, y_axes, xlim=None, ylim=None, save_fig=False):
    
    '''Plot a single Pandas dataframe for a given x axis (e.g., time) and one/multiple y axes (e.g., x_acceleration)'''
    
    def create_y_label(y_axes):
        y_axis = ''

        for y_ax in y_axes:
            if y_axis != '':
                y_axis += ', '
            y_axis += y_ax  

        # Add relevant units to y label
        if 'Accel' in y_axis:
            y_label = y_axis + ' ($m/s^2$)'
        elif 'Gyro' in y_axis:
            y_label = y_axis + ' ($rad/s$)'
        elif 'Vel' in y_axis:
            y_label = y_axis + ' ($m/s$)'
        else:
            y_label = y_axis
            
        return(y_label)
    
    # create y label from the input list of y_axes
    y_label = create_y_label(y_axes)
    
    plt.figure(figsize=(10, 5))
    
    # Plot relevant data
    for y_axis in y_axes:
        plt.plot(_datasets[dataset_name][x_axis], _datasets[dataset_name][y_axis], label=y_axis)
    
    # add relevant info to the graphs
    plt.title(dataset_name)
    plt.legend()
    plt.xlabel(x_axis + ' (s)')
    plt.ylabel(y_label)
    
    # Use limits if they've been passed in
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1]) 
        
    plt.show()


# In[12]:


plot_one_dataframe(raw_datasets, dataset_labels[-1], 'Run Time', ['X Accel', 'Z Accel', 'Y Accel'], save_fig = False)


# ### 2.2. Plotting functions: Compare two Pandas datasets by Run Time

# In[13]:


def dataset_compare(dataset1, label1, dataset2, label2, y_axis, t_offset=0, y_offset=0, trim=False, Filtered = False):
    
    '''Plot and compare one-axis measurement (e.g., X-accel) of two Pandas datasets over Run Time'''
    
    # number of datapoints to display for a filtered dataset (currently equal to 1 window size)
    trim_offset = SAMP_SLICE 

    # Plot parameters
    plt.clf()
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_xlabel('Run Time ($s$)')
    ax.set_title(y_axis + ' for ' + label1 + ' and ' + label2)
        
    # Add relevant units to y label
    if 'Gyro' in y_axis:
        ax.set_ylabel(y_axis + ' ($rad/s$)')
    elif 'Accel' in y_axis:
        ax.set_ylabel(y_axis + ' ($m/s^2$)')
    elif 'Vel' in y_axis:
        ax.set_ylabel(y_axis + ' ($m/s$)')
    else:
        ax.set_ylabel('Unknown')
    
    # determine whether plotting filtered or raw data
    if Filtered:
        legend1= label1 + '_raw'
        legend2= label2 +'_filtered'
    else:
        legend1 = label1
        legend2 = label2
   
    # remove extra text in the labels
    legend1 = legend1.replace('Middle_','').replace('_Module6050','')
    legend2 = legend2.replace('Middle_','').replace('_Module6050','')
    
    # Plot data with given y and t offsets applied to first dataset
    if trim:
        ax.plot(dataset1[label1]['Run Time'][:trim_offset].apply(lambda t: t + t_offset), 
            dataset1[label1][y_axis][:trim_offset].apply(lambda y: y + y_offset), label=legend1)
        ax.plot(dataset2[label2]['Run Time'][:trim_offset], 
            dataset2[label2][y_axis][:trim_offset], label=legend2)
        
    else:
        ax.plot(dataset1[label1]['Run Time'].apply(lambda t: t + t_offset), 
            dataset1[label1][y_axis].apply(lambda y: y + y_offset), label=legend1)
        ax.plot(dataset2[label2]['Run Time'], 
            dataset2[label2][y_axis], label=legend2)
    
    # Include offset info text in plot
#     offset_text = 'Offsets\n'
#     offset_text += ': t={}'.format(t_offset) + ', ' + 'y={}'.format(y_offset)
#     ax.text(0.05, 0.05, s=offset_text, 
#             horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    
    ax.legend()
    plt.show()


# In[14]:


# Compare Z Gyro data for two dataframes
dataset_compare(raw_datasets, dataset_labels[0],
               raw_datasets, dataset_labels[1], y_axis='Z Accel', t_offset=0)


# ### 2.3. Plotting functions: Plot given x and y axes for every Pandas DataFrame in given array of datasets

# In[15]:


def plot_selected_datasets(_datasets, x_axis, y_axis, windowed=False, win_num=0, take_row=False):

    '''Plot given x and y axes for every Pandas DataFrame in given array of datasets'''

    # Set paramaeters based on number of datasets to plot
    n_axes = len(_datasets)
    odd_axes = n_axes % 2 == 0
    rows = int((n_axes + 1) / 2)
    
    # Scale approriately
    if (odd_axes):
        fig = plt.figure(figsize=(n_axes*5, n_axes*3))          
    else:
        fig = plt.figure(figsize=(n_axes*5, n_axes*2))
    
    # Grid of subplots
    gs = gridspec.GridSpec(rows, 2)
    axes = []
    row, col = 0, 0
    
    # Plot each of the given datasets
    for i, (label, dataset) in enumerate(_datasets.items()):
        # Take a whole row if odd num of axes
        if (i == n_axes-1 and odd_axes and take_row): 
            axes.append(fig.add_subplot(gs[row, :]))
        else:
            axes.append(fig.add_subplot(gs[row, col]))
        
        # Plot on new subplot
        if (windowed):
            axes[i].plot(dataset[win_num][x_axis], dataset[win_num][y_axis])
        else:
            axes[i].plot(dataset[x_axis], dataset[y_axis])
        axes[i].set_title(label)
        axes[i].set_xlabel(x_axis + ' (s)')
        
        if 'Gyro' in y_axis:
            axes[i].set_ylabel(y_axis + ' ($rad/s$)')
        elif 'Accel' in y_axis:
            axes[i].set_ylabel(y_axis + ' ($m/s^2$)')
        
        # Only go two columns wide
        col += 1
        if (col == 2):
            row += 1
            col = 0
        
    plt.subplots_adjust(hspace=0.35, wspace=0.15)
    plt.show()


# In[16]:


# Plot Z Accel of selected datasets
datasets_to_plot = {label: dataset for label, dataset in raw_datasets.items() 
                    if 'Middle' in label and 'Grass' in label and 'F8' in label}
plot_selected_datasets(datasets_to_plot, x_axis='Run Time', y_axis='Z Accel')


# ### 2.4. Plotting functions: Plot a selected axis of all datasets

# In[17]:


def plot_all_OneAxis(datasets, axis):
    
    '''plotting one axis (e.g., X Accel) measurement of all dataframes '''
    
    if USER == 'All':
        ncol = 4 # N of participants
    else:
        ncol = 1 # N of participants
    
    nrow = int(len(datasets)/ncol)

    fig, axs = plt.subplots(nrow, ncol, sharey=True, figsize=(20,50))
    fig.tight_layout()
    
    axs = axs.ravel()

    for i, (label, dataset) in enumerate(datasets.items()):
        axs[i].plot(dataset[axis])
        axs[i].set_title(label.replace('Middle_','').replace('_Module6050',''))
    
    fig.text(-.01, 0.5, 'Acceleration: X-axis ${(m/s^2)}$', va='center', rotation='vertical', fontsize = 20)


# In[18]:


plot_all_OneAxis(raw_datasets, axis = 'X Accel')


# ## Part 3 - Converting Between Pandas and Numpy

# In[19]:


def pd_to_np(pd_datasets, windowed=False):
    
    '''Convert array of Pandas DataFrames to array of 2D NumPy array'''

    np_datasets = {}
    
    # Convert each dataset individually
    for label, dataset in pd_datasets.items():
        np_dataset = []
        
        # Return passed datasets if they are already NumPy ndarrays
        if type(dataset) is np.ndarray:
            print('Note: Already a NumPy array!')
            return pd_datasets
        
        # If windowed, convert individual windows to Pandas
        if (windowed):
            for window in dataset:
                
                # MK edited, based on this link: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.as_matrix.html
                np_dataset.append(window.to_numpy())  
        else:
            np_dataset = dataset.to_numpy()
        
        np_datasets.update({label: np_dataset})
        
    return np_datasets


# In[20]:


def np_to_pd(np_datasets, windowed=False):
    
    '''Convert array of 2D NumPy arrays to Pandas Data Frames'''

    pd_datasets = {}
    
    # Convert each dataset individually
    for label, dataset in np_datasets.items():
        pd_dataset = []
        
        # Return passed datasets if they are already Pandas dataframes
        if type(dataset) is pd.DataFrame:
            print('Note: Already a Pandas dataframe!')
            return np_datasets
        
        # Use correct column names
        new_columns = std_columns.copy()
            
        # If windowed, convert individual windows to Pandas
        if (windowed):
            for window in dataset:
                pd_dataset.append(pd.DataFrame(data=window, columns=new_columns))     
        else:
            pd_dataset = pd.DataFrame(data=dataset, columns=new_columns)
            
        pd_datasets.update({label: pd_dataset})
    
    return pd_datasets


# In[21]:


# Convert to NumPy
raw_datasets = pd_to_np(raw_datasets)


# In[22]:


# Run to convert back to Pandas
raw_datasets = np_to_pd(raw_datasets)


# In[23]:


# Check if its constructed correctly
print('Number of datasets: {}'.format(len(raw_datasets)))
print('Shape of first dataset: {}'.format(raw_datasets[dataset_labels[4]].shape))


# ## Part 4 - Signal processing

# ### 4.1. Butterworth Filter
# #### 4.1.1. Function to set appropriate cut-off frequency for butterworth filter

# In[24]:


def get_frequencies(label):
    
    '''Get relevant frequencies for given label based on whether its a frame or wheel dataset'''

    # Sampling frequency (and thus cutoff frequency) varies between frame and wheel modules
    ## Currently not using high pass frequency
    if 'Left' in label or 'Right' in label:
        f_samp = f_samp_wheels 
        f_low = CUT_OFF 
        f_high = 1 
        
    elif 'Middle' in label:
        f_samp = f_samp_6050 # Sampling frequency
        f_low = CUT_OFF # a list of cutoff frequency starting at 20Hz
        f_high = 1 # High pass cutoaff frequency
        
    else:
        raise Exception('Unknown label')
        
    return f_samp, f_low, f_high


# #### 4.1.2. Implementing a 4th-order butterworth filter & creating a new dictionary with filtered dataframes

# In[25]:


# Filtered datasets dictionary
filt_datasets = {}

# Filter each dataset individually
for label, raw_dataset in raw_datasets.items():
    # Sampling rates are not consistent across all datasets
    f_samp, f_low, f_high = get_frequencies(label)
    
    # Nyquist frequecy
    nyq = 0.5 * f_samp 
    
    # Get normalized frequencies 
    w_low = f_low / nyq 

    # Get Butterworth filter parameters (numerator and denominator)
    ## The function sosfiltfilt (and filter design using output='sos') should be preferred over filtfilt for most 
    ## filtering tasks, as second-order sections have fewer numerical problems.
    sos = butter(N=2, Wn=w_low, btype='low', output='sos')
    
    # Number of columns containing data
    n_data_col = len(data_columns)
    
    # Filter all the data columns
    dataset = np.copy(raw_dataset)
    
    for i in range(n_data_col):
        
        # Apply a digital filter forward and backward to a signal.
        ## The combined filter (filtfilt) has zero phase and a filter order twice that of the original.
        dataset[:, i] = sosfiltfilt(sos, dataset[:, i])
        
    filt_datasets.update({label: dataset})
#     filt_datasets.update({label: dataset[5:-5,:]})


# In[26]:


# Check construction of filtered dataset
print('Num filtered datasets: {}'.format(len(filt_datasets)))
print('Shape of first filtered dataset: {}'.format(filt_datasets[dataset_labels[1]].shape))


# In[27]:


# Verify we can convert back to Pandas
display(np_to_pd(filt_datasets, windowed=False)[dataset_labels[0]].head())
display(raw_datasets[dataset_labels[0]].head())


# In[28]:


filt_datasets = np_to_pd(filt_datasets)


# In[29]:


# compare a segment of filtered and unfiltered data
dataset_compare(raw_datasets, dataset_labels[-1], 
                filt_datasets, dataset_labels[-1], 'Z Gyro', trim = True, Filtered = True)


# In[30]:


# compare filtered and unfiltered data
dataset_compare(raw_datasets, dataset_labels[-1], filt_datasets, dataset_labels[-1],
                         'Z Accel', Filtered = True)


# ### 4.2. Create a no-motion dataframe

# In[31]:


'''extract no-motion data from all datafames'''
def create_noMotion_dataset(datasets):

    noMotion_datasets = {}
    
    # axis to use as a reference
    thresh_axes = 'X Accel' # this is more representative of the state of motion rather than z acceleration
    
    # start/stop threshold of motion for raw & filtered data
    NOMOTION_STOP_THRESH = 1.5
    
    # no-motion data 
    for label, dataset in datasets.items():
        
        # correct one of measurements
        if  'Grass' in label and 'Straight' in label and 'Kevin' in label:
            stop_index = 0
            
        else:
            # Caluclate first instance below threshold and use as the time domain               
            stop_index = dataset[dataset[thresh_axes] > NOMOTION_STOP_THRESH].index[0]
        
        dataset = dataset[:stop_index]
        
        noMotion_datasets.update({label:dataset})

    # concatenating the DataFrames
    df = pd.concat(noMotion_datasets.values(), ignore_index=True)
    
    return noMotion_datasets, df


# In[32]:


# obtain no-motion data in the form of a dictionary & dataset
noMotion_datasets, noMotion_dataset = create_noMotion_dataset(filt_datasets)

# plot no-motion data
plt.plot(noMotion_dataset['X Accel'])
plt.ylabel('No-motion X Acceleration $(m/s^2)$')

# examine no-motion dataset
noMotion_dataset.head()


# ### 4.3. Update all dataframe and remove no-motion data

# In[33]:


def trim_data(datasets):
    
    '''
    function to remove stationary data from all dataframes
    this function was tuned using visual inspection of trimed output
    '''
    
    trim_datasets = {}
    
    # axis to use as a reference
    thresh_axes = 'X Accel' # this is more representative of the state of motion rather than z acceleration
    
    # start/stop threshold of motion for raw & filtered data
    STARTUP_THRESH_FILT = 1.5 
    STOP_THRESH_FILT = 0.0
    
    # get non-stationary signal 
    for label, dataset in datasets.items():
        
        # correct one of measurements
        if 'Grass' in label and 'Straight' in label and 'Kevin' in label:
            dataset = dataset.iloc[300:]
        
        # Caluclate first and last instance above threshold and use as the time domain
        start_index = dataset[dataset[thresh_axes] > STARTUP_THRESH_FILT].index[0]               
        stop_index = dataset[dataset[thresh_axes] > STARTUP_THRESH_FILT].index[-1]
        dataset = dataset[start_index:stop_index]
        
        trim_datasets.update({label:dataset})
    
    return trim_datasets


# In[34]:


# create a dictionary of dataframes with motion datasets only
all_datasets = trim_data(filt_datasets)


# In[35]:


# compare filtered and unfiltered data
for i in range(len(dataset_labels)):
    dataset_compare(raw_datasets, dataset_labels[i], all_datasets, dataset_labels[i], 
                    'X Accel', Filtered = True)


# In[36]:


# combine motion & no-motion datasets
all_datasets.update({'no_motion':noMotion_dataset})

for label, dataset in all_datasets.items():
    dataset = dataset.drop(['Run Time', 'Epoch Time'], axis='columns')
    all_datasets.update({label:dataset})

all_datasets[dataset_labels[0]].head()


# ## Part 5 - Data segmentation & Windowing (using hanning window)

# In[37]:


def slice_window(datasets, overlap = True):

    '''
    WINDOWING 
    By using windowing functions, you can further enhance the ability of an FFT to extract spectral data from signals. 
    Windowing functions act on raw data to reduce the effects of the leakage that occurs during an FFT of the data. 
    Leakage amounts to spectral information from an FFT showing up at the wrong frequencies.

    '''
    segmented_datasets = {} # creating a dictionary of lists of sliced (N=512) dataframes 
    windowed_datasets = {} # creating a dictionary of sliced & windowed dataframes - this would be used for fft/psd only 

    # Trim excess data points, then split into short segments
    for label, dataset in datasets.items():
        
        # window size
        window_size = SAMP_SLICE 

        # create an empty list of Sliced/Windowed dataframes
        segmented_dataset = []
        windowed_dataset = []

        # Iterate through dataset by half a window at a time and extract segments
        i = 0

        # whether to have overalping or non-overlaping segments
        if overlap:
            window_slide = int(window_size / 2) # to create 50% overalping segments
        else:
            window_slide = window_size

        # create hanning window
        win = np.hanning(window_size)

        while (i + window_size  <= len(dataset)):

            # update the list of segmented dataframes
            segmented_dataset.append(dataset[i:i + window_size])

            # multiply han window & data segments
            dataset_copy = dataset[i:i + window_size] * win[:,None]

            # update the list of windowed segments
            windowed_dataset.append(dataset_copy)

            # slide forward
            i += window_slide

        segmented_datasets.update({label: segmented_dataset})
        windowed_datasets.update({label: windowed_dataset})
        
    return segmented_datasets, windowed_datasets


# In[38]:


# create a dictionary of segmented & windowed dataframes
segmented_datasets, windowed_datasets = slice_window(all_datasets)


# In[39]:


# Check if its constructed correctly
print('Total num segmented/windowed datasets: {}'.format(len(windowed_datasets)))
print('Num of segments (time windows) in first dataset: {}'.format(len(windowed_datasets[dataset_labels[0]])))
print('Shape of individual window in the first dataset: {}'.format(windowed_datasets[dataset_labels[0]][0].shape))
print('Shape of original dataframe in the first dataset: {}'.format(all_datasets[dataset_labels[0]].shape))
print('type of the first dataset:{}'.format(type(windowed_datasets[dataset_labels[0]])))
print('type of the first window in the first dataset:{}'.format(type(windowed_datasets[dataset_labels[0]][0])))


# In[40]:


print('Example of a segmented dataframe')
segmented_datasets[dataset_labels[0]][0]


# In[41]:


examine two consecutive segmented/windowed dataframe
j = 0
fig, ax = plt.subplots(2,1, figsize = (8, 6))

for i in range(60,62):
    ax[j].plot(segmented_datasets[dataset_labels[0]][i]['X Accel'], label='filtered')
    ax[j].plot(windowed_datasets[dataset_labels[0]][i]['X Accel'], label='windowed')
    ax[j].legend()
    ax[j].set_xlabel('index')
    ax[j].set_ylabel('segmented window #{}'.format(i))
    j+=1


# ## Part 6 - Transforms (FFT, PSD)

# ### 6.1. Function to perform Fast Fourier Transform 

# In[42]:


def fft_transform(datasets):
    
    '''function to create a dictionary of all FFT'd dataframes'''
    
    fft_datasets = {}
    
    # number of sample points
    N = SAMP_SLICE

    # sample spacing
    T = 1/f_samp_6050
    
    # Frequency bin centers
    xf = fftfreq(N,T)[:N//2]

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
                yf = 2.0/N * abs(y[0:N//2]) # keeping positive frequencies

                # add fft'd signal to a dataframe
                fft_df[column] = yf

            # Append the frequency column
            fft_df['frequency'] = xf

            fft_dataset.append(fft_df)

        fft_datasets.update({label: fft_dataset})   
            
    return(fft_datasets)


# In[43]:


# create a dictionary containing lists of fft'd windowed segments
fft_datasets = fft_transform(windowed_datasets)


# In[44]:


# Check if fft_datasets are constructed correctly
print('Num of FFT\'d windowed datasets: {}'.format(len(fft_datasets)))
print('Num of FFT\'d windows in first dataset: {}'.format(len(fft_datasets[dataset_labels[0]])))
print('Shape of FFT\'d individual window: {}'.format(fft_datasets[dataset_labels[0]][0].shape))
fft_datasets[dataset_labels[0]][0].head()


# ### 6.2. Creating PSD datasets

# In[45]:


def psd_transform(datasets):
    
    '''function to create a dictionary of all PSD'd dataframes'''

    psd_datasets = {}
    
    # number of sample points
    N = SAMP_SLICE

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
                f, Pxx_den = welch(window[column].values, fs)

                # add psd'd signal to a dataframe
                psd_df[column] = Pxx_den

            # Append the frequency column
            psd_df['frequency'] = f

            psd_dataset.append(psd_df)

        psd_datasets.update({label: psd_dataset})   
            
    return(psd_datasets)


# In[46]:


# create a dictionary containing lists of psd'd windowed segments
psd_datasets = psd_transform(segmented_datasets)


# In[47]:


# Check if fft_datasets are constructed correctly
print('Num of PSD\'d windowed datasets: {}'.format(len(psd_datasets)))
print('Num of PSD\'d windows in first dataset: {}'.format(len(psd_datasets[dataset_labels[0]])))
print('Shape of PSD\'d individual window: {}'.format(psd_datasets[dataset_labels[0]][0].shape))
psd_datasets[dataset_labels[0]][0].head()


# ### 6.3. Plot function for frequency domain dataset

# In[48]:


def plot_transforms(fft, psd):
    
    '''function to visualise a random window of a random dataset from fft/psd transforms'''
    
    # choose a random dataframe
    i = randrange(len(fft)-1)
    j = randrange(len(fft[dataset_labels[i]])-1)
    
    print(dataset_labels[i],'\nwindow #{}'.format(j))
    df_fft = fft[dataset_labels[i]][j]
    df_psd = psd[dataset_labels[i]][j]
    
    fig, ax = plt.subplots(1,2, figsize = (12,4))    
    
    for col in data_columns:
        ax[0].plot(df_fft['frequency'], df_fft[col], label=col)
        ax[1].plot(df_psd['frequency'], df_psd[col], label=col)
        
    ax[0].set_xlabel('frequency (hz)'); ax[1].set_xlabel('frequency (hz)') 
    ax[0].set_ylabel('fft'); ax[1].set_ylabel('psd') 
    ax[0].legend(); ax[1].legend()
    plt.show()


# In[49]:


plot_transforms(fft_datasets, psd_datasets)


# ### 6.4. Trim excessive unused frequencies

# In[50]:


def trim_transforms(datasets, freq_thresh):
    
    ''' function to remove excessive frequency bins of fft & psd dataframes'''

    datasets_trimmed = {}
    
    for label, dataset_list in datasets.items():
        dataset_trimmed_list =[]
        
        for dataset in dataset_list:
            # remove frequencies higher than freq_thresh
            dataset_trimmed_list.append(dataset[dataset['frequency'] <= freq_thresh].copy())
            
        datasets_trimmed.update({label: dataset_trimmed_list})
    return datasets_trimmed


# In[51]:


# trim fft & psd dataframes to keep useful frequencies only
fft_datasets_trimmed= trim_transforms(fft_datasets, CUT_OFF + 10)
psd_datasets_trimmed= trim_transforms(psd_datasets, CUT_OFF + 10)


# In[52]:


# Check if fft_datasets are constructed correctly
print('Num of PSD\'d windowed datasets: {}'.format(len(fft_datasets_trimmed)))
print('Num of PSD\'d windows in first dataset: {}'.format(len(fft_datasets_trimmed[dataset_labels[0]])))
print('Shape of PSD\'d individual window: {}'.format(fft_datasets_trimmed[dataset_labels[0]][0].shape))
fft_datasets_trimmed[dataset_labels[0]][0].head()


# In[53]:


# compare trimmed & original fft datasets
print('full frequency range fft/psd datasets')
plot_transforms(fft_datasets, psd_datasets)
print('fft/psd trimeed datasets')
plot_transforms(fft_datasets_trimmed, psd_datasets_trimmed)


# ## Part 7 - Feature Engineering
# #### Extract relevant features (e.g. Mean, Min, Skew, ...) from each data window

# ### 7.1. Feature Extraction Functions (l2norm, autocorr, rms, zcr, msf, rmsf, fc, vf, rvf)

# In[54]:


# Feature extraction functions
def l2norm(array):
    '''L2 norm of an array'''
    return np.linalg.norm(array, ord=2)

def rms(array):
    '''Root mean squared of an array'''
    return np.sqrt(np.mean(array ** 2))

def zcr(array):
    '''Zero crossing rate of an array as a fraction of total size of array'''
    # divide by total datapoints in window
    return len(np.nonzero(np.diff(np.sign(array)))[0]) / len(array)


# In[55]:



def feature_extraction(datasets, features_dic, freq_domain = False):
    
    '''Extract given features from column of each dataset
       Converts a dictionary of datasets to a nested dictionary where each dataset has its own dictionary
       of axes/directions'''
    
    # will be updated with a nested dictionary of {label:{data column name:[dataframe with feature extracted columns]}}
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
                
                '''Alternate defintion for frequency functions'''
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


# ### 7.2. Time Domain Features

# In[56]:


# dictionary of time-domain features to use in feature extraction step
time_features = {'Mean': np.mean, 'Std': np.std,  'Norm': l2norm, 
                 'Max': np.amax, 'Min' : np.amin, 'RMS': rms, 'ZCR': zcr} 


# In[57]:


# create a dictionary of feature extracted dataframes
time_featured_datasets = feature_extraction(segmented_datasets, time_features)


# In[58]:


# Check if feature data is constructed correctly and print some info
print('Num datasets: {}'.format(len(time_featured_datasets)))
print('Num windows: {}'.format(len(time_featured_datasets[dataset_labels[1]])))
print('Shape of first dataset first column: {}'.format(time_featured_datasets[dataset_labels[1]]['X Accel'].shape))
time_featured_datasets[dataset_labels[1]]['X Accel'].head()


# In[59]:


def plot_set_features(datasets, dirn, feat_name):
    
    '''Plot extracted feature of one direction for all terrains'''
    
    plt.clf()
    plt.figure(figsize=(10,8))
    
    for label, dataset in datasets.items():
        plt.plot(dataset[dirn][feat_name], label=label)
    
    plt.ylabel(feat_name)
    plt.xlabel('Window #')
    plt.title(dirn)
    plt.legend()
    plt.show()


# In[60]:


# Plot some time feature data
feat_datasets_to_plot = {label: dataset for label, dataset in time_featured_datasets.items()
                         if 'Middle' in label and 'Mahsa' in label and 'F8' in label}
plot_set_features(feat_datasets_to_plot, dirn='Z Accel', feat_name='Norm')


# ### 7.3. Frequency Domain Features

# In[61]:


# For small float values
EPSILON = 0.00001

def msf(freqs, psd_amps):
    '''Mean square frequency'''
    num = np.sum(np.multiply(np.resize(np.power(freqs,2), len(psd_amps)), psd_amps))
    denom = np.sum(psd_amps)
    
    # In case zero amplitude transform is ecountered
    if denom <= EPSILON:
        return EPSILON
    
    return np.divide(num, denom)

def rmsf(freqs, psd_amps):
    '''Root mean square frequency'''
    return np.sqrt(msf(freqs, psd_amps))

def fc(freqs, psd_amps):
    '''Frequency center'''
    num = np.sum(np.multiply(np.resize(freqs, len(psd_amps)), psd_amps))
    denom = np.sum(psd_amps)
    
    # In case zero amplitude transform is ecountered
    if denom <= EPSILON:
        return EPSILON
    
    return np.divide(num, denom)

def vf(freqs, psd_amps):
    '''Variance frequency'''
    return msf(freqs-fc(freqs, psd_amps), psd_amps)

def rvf(freqs, psd_amps):
    '''Root variance frequency'''
    return np.sqrt(msf(freqs, psd_amps))


# In[62]:


# dictionary of freq-domain features to use in feature extraction step
freq_features = {'RMSF': rmsf, 'FC': fc, 'RVF': rvf}


# In[63]:


# create a dictionary of freq-domain feature extracted dataframes
freq_featured_datasets = feature_extraction(psd_datasets_trimmed, freq_features, freq_domain = True)


# In[64]:


# Check if feature data is constructed correctly and print some info
print('Num datasets: {}'.format(len(freq_featured_datasets)))
print('Num directions: {}'.format(len(freq_featured_datasets[dataset_labels[0]])))
print('Shape of one direction: {}'.format(freq_featured_datasets[dataset_labels[0]]['X Accel'].shape))
freq_featured_datasets[dataset_labels[0]]['X Accel'].head()


# In[65]:


# Plot some frequency feature data
feat_datasets_to_plot = {label: feature for label, feature in freq_featured_datasets.items()
                         if 'Mahsa' in label and 'F8' in label}
plot_set_features(feat_datasets_to_plot, dirn='Z Accel', feat_name='RMSF')


# ## Part 8 - Columning, Combination, and Standardization of Datasets

# ### 8.1. Columning Data
# ##### Combine IMU data from each direction into single dataframes with columns for each feature in each direction

# In[66]:


def append_all_columns(columns, append_tag):

    '''Append a tag to the end of every column name of a dataframe'''

    new_columns = []
    
    for column in columns:
        if append_tag not in column:
            new_columns.append(column + ' ' + append_tag)
        else:
            new_columns.append(column)
    
    return new_columns


# #### 8.1.1. Time & Freq Extracted Featured Data

# In[67]:


def combine_extracted_columns(datasets):
    
    '''Combined directions (axes) of a featured dataset'''

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


# In[68]:


# Take time feature data and combine axes columns
columned_time_feat_datasets = combine_extracted_columns(time_featured_datasets)

# Confirm formatting
columned_time_feat_datasets[dataset_labels[0]].head()


# In[69]:


# Take frequency feature data and axes columns
columned_freq_feat_datasets = combine_extracted_columns(freq_featured_datasets)

# Confirm formatting
columned_freq_feat_datasets[dataset_labels[0]].head()


# #### 8.1.2. Transformed Data

# In[70]:


def get_transform(_label):

    '''Get the transform used for given label'''

    for transform in transforms:
        if transform in _label:
            return transform
    
    raise Exception('Unkown transform')


# In[71]:


def combine_transform_columns(datasets, trans = ''):
    
    '''Combined direction (axes) columns for transformed data'''

    combined_datasets = {}
    
    for label, dataset in datasets.items():
        
        # Get frequency bins from frequency column of first window
        freq_bins = dataset[0]['frequency'].tolist()
        
        # Get more parameter for current label
        trans = trans
        
        # Combine parameters to form columns for new combined DataFrame
        new_cols = [trans + ' {} Hz '.format(round(f_bin, 1)) + d_col for d_col in data_columns for f_bin in freq_bins]
        
        # Convert windowed arrays into a single array with each window as a row
        new_data = []
        
        for window in dataset:
            new_row = []
            for d_col in data_columns:
                new_row.extend(window[d_col].tolist())
            new_data.append(new_row)
            
        # Create new DataFrame
        combined_df = pd.DataFrame(data=new_data, columns=new_cols)
        combined_datasets.update({label: combined_df})

    return combined_datasets


# In[72]:


columned_fft_datasets = combine_transform_columns(fft_datasets_trimmed, 'FFT')

# Confirm FFT formatting
columned_fft_datasets[dataset_labels[0]].head()


# In[73]:


columned_psd_datasets = combine_transform_columns(psd_datasets_trimmed, 'PSD')

# Check PSD formatting
columned_psd_datasets[dataset_labels[0]].head()


# ### 8.2. Adding Labels
# 
# ##### Create a new column containg the an integer label for each terrain.

# In[74]:


def get_terrain_num(_label):

    '''Get the integer terrain value of a given label'''

    for i, terrain in enumerate(terrains):
        if terrain in _label:
            return (i+1)
        elif 'no_motion' in _label:
            return 0
        
    raise Exception('Unknown terrain')


# In[75]:


def insert_labels(datasets):
    
    'Add labels to a dataset'

    # Returns new datasets
    labeled_datasets = {}
    
    # Add to each dataframe of a dataset
    for label, dataset in datasets.items():
        
        dataset_copy = dataset.copy()
        # get terrain label
        terrain_num = get_terrain_num(label)
        
        # create a list of labels for each dataset
        labels = [terrain_num for _ in range(len(dataset))]
        
        # insert labels (7 I/O terrain + no-motion)
        dataset_copy.insert(0, 'Label', labels)
        
        if 'no_motion' in label:
            labels_IO = [0 for _ in range(len(dataset))]
        elif 'Asphalt' in label or 'Sidewalk' in label or 'Grass' in label or 'Gravel' in label:
            labels_IO = [2 for _ in range(len(dataset))] # outdoor terrains
        else:
            labels_IO = [1 for _ in range(len(dataset))] # indoor terrains
        
        dataset_copy.insert(0, 'Label_IO', labels_IO)
        labeled_datasets.update({label: dataset_copy})
            
    return labeled_datasets


# In[76]:


# Add labels to each of the feature vector types
labeled_time_feat_datasets = insert_labels(columned_time_feat_datasets)
labeled_freq_feat_datasets = insert_labels(columned_freq_feat_datasets)
labeled_fft_datasets = insert_labels(columned_fft_datasets)
labeled_psd_datasets = insert_labels(columned_psd_datasets)


# In[77]:


# Check labelled data
labeled_time_feat_datasets[dataset_labels[0]].head()


# ### 8.3. Combining Datasets
# ##### Convert data from each dataset into rows of a single dataframe

# In[78]:


def combine_datasets(datasets):

    '''Combine data from labelled datasets into a single dataframe'''

    return pd.concat(list(datasets.values()), ignore_index=True)


# In[79]:


# For each feature vector, combine datasets in two single dataframes
time_feats = combine_datasets(labeled_time_feat_datasets)
freq_feats = combine_datasets(labeled_freq_feat_datasets)                                      
ffts = combine_datasets(labeled_fft_datasets)
psds = combine_datasets(labeled_psd_datasets)


# In[80]:


# Check unnormalized data
display(freq_feats.tail())
display(time_feats.tail())
display(ffts.tail())
display(psds.tail())


# In[81]:


# plot selected features of combined datasets 
feats_to_plot = ['Norm X Accel', 'Norm Z Accel']
for feat in feats_to_plot:
    plt.figure(figsize = (16,4))
    plt.plot(time_feats[feat])
    plt.ylabel(feat)
    plt.show()


# Note: More rows of exracted feature data are lost than transform features

# ## Part 9 - Visualized mean and std of all columns of all datasets and  for all terrains

# ### Time feature dataframes

# In[82]:


def plt_mean_std(dataset, freq = False):
    
    dataset_mean = pd.DataFrame(columns=dataset.columns)
    dataset_std = pd.DataFrame(columns=dataset.columns)

    x = np.linspace(0,7,8)

    for i in range(len(terrains)+1): 
        dataset_mean = dataset_mean.append(dataset.loc[dataset['Label'] == i].mean(), ignore_index = True)
        dataset_std = dataset_std.append(dataset.loc[dataset['Label'] == i].std(), ignore_index = True)

    if freq:
        ncol = 4
    else:
        ncol = 7
        
    nrow = int((len(dataset_mean.columns)-2)/ncol)

    fig, ax = plt.subplots(nrow, ncol, figsize=(50,40), constrained_layout = True)
    
    # to remove unused columns (label & label_IO)
    k = 2

    for i in range(nrow):
        for j in range(ncol):
            ax[i, j].scatter(x,dataset_mean[dataset_mean.columns[k]])
            ax[i, j].errorbar(x,dataset_mean[dataset_mean.columns[k]], yerr = dataset_std[dataset_mean.columns[k]].to_numpy(), fmt='none')
            ax[i, j].set_title(dataset_mean.columns[k], fontsize = 30)
            k +=1


# In[83]:


plt_mean_std(time_feats)


# In[84]:


plt_mean_std(freq_feats, freq = True)


# In[85]:


# number of datapoints for each terrain/ motion-state
x = np.linspace(0,7,8)
fig, ax = plt.subplots(figsize = (8,4))
y = time_feats['Label'].value_counts().sort_index()
ax.bar(x, y)
for i, value in enumerate(y):
    plt.text(i-0.2,value+1, str(value))
plt.show()


# ## Part 10 - Computational performance analysis

# In[86]:


def latency_analysis(time_window, TIME, FREQ, FFT, PSD):
    
    ''' function to estimate computational time of pre-processing each time window '''
    
    # number of sample points in each window
    N = SAMP_SLICE
    
    def filt_(window):
        # filtering signal
        f_samp = f_samp_6050 # Sampling frequency
        f_low = CUT_OFF # a list of cutoff frequency starting at 20Hz
        f_high = 1 # High pass cutoaff frequency

        # Nyquist frequecy
        nyq = 0.5 * f_samp 

        # Get normalized frequencies 
        w_low = f_low / nyq 

        # Get Butterworth filter parameters (numerator and denominator)
        ## The function sosfiltfilt (and filter design using output='sos') should be preferred over filtfilt for most 
        ## filtering tasks, as second-order sections have fewer numerical problems.
        sos = butter(N=2, Wn=w_low, btype='low', output='sos')

        # Number of columns containing data
        n_data_col = 6

        # Filter all the data columns
        filt_window = np.copy(window)

        for i in range(n_data_col):

            # Apply a digital filter forward and backward to a signal.
            ## The combined filter (filtfilt) has zero phase and a filter order twice that of the original.
            filt_window[:, i] = sosfiltfilt(sos, filt_window[:, i])   

        window_ = pd.DataFrame(data = filt_window, columns = data_columns)

        return window_
    
    # filter
    filt_window = filt_(time_window)
    
    if FREQ or FFT or PSD:

        def windowed_(window):

            # create hanning window
            win = np.hanning(512)

            # multiply han window & data segments
            window_ = window * win[:,None]

            return window_
        
        # windowed
        windowed_window = windowed_(filt_window)
    
    if FFT:
        def fft_(window):

            # sample spacing
            T = 1/f_samp_6050

            # Frequency bin centers
            xf = fftfreq(N,T)[:N//2]

            # create an emtpy dataframe
            fft_df = pd.DataFrame(columns=data_columns)

            # calculate fft of each time window 
            for i, column in enumerate(data_columns):

                # calculate fft of each column
                y = fft(window[column].values)
                yf = 2.0/N * abs(y[0:N//2]) # keeping positive frequencies

                # add fft'd signal to a dataframe
                fft_df[column] = yf

            # Append the frequency column
            fft_df['frequency'] = xf

            fft_df = fft_df[fft_df['frequency'] <= CUT_OFF + 10]

            return(fft_df)
        
        # fft
        fft_window = fft_(windowed_window)
    
    if PSD or FREQ:
        def psd_(window):

            psd_datasets = {}

            # sampling frequency
            fs = f_samp_6050    

            # create an emtpy dataframe
            psd_df = pd.DataFrame(columns=data_columns)

            # calculate psd of each time window 
            for i, column in enumerate(data_columns):

                # calculate psd of each column
                f, Pxx_den = welch(window[column].values, fs)

                # add psd'd signal to a dataframe
                psd_df[column] = Pxx_den

            # Append the frequency column
            psd_df['frequency'] = f  

            # trim unwanted frequencies
            psd_df = psd_df[psd_df['frequency'] <= CUT_OFF + 10]

            return(psd_df)
        
        # psd
        psd_window = psd_(windowed_window)
    
    if TIME:
        def time_feature_extraction(window, features_dic):
            # will be updated with keys as data columns (e.g., 'X Accel') 
            cols_dic = {}

            window_copy = window.copy()

            # Loop over data columns
            for col in data_columns:

                # will be updated with keys as extracted feature names (e.g., 'Mean')
                feats = {}

                def function_all_windows(function):

                    featured_column = function(window_copy[col])

                    return featured_column 

                # Execute every function over all windows    
                for feat_name, feat_func in features_dic.items():

                    # apply feature extraction to the ith column for all windows
                    feats.update({feat_name: function_all_windows(feat_func)})


                cols_dic.update({col: pd.DataFrame(feats, index=[0])})

            return cols_dic
        
        # time feature extraction
        time_featured_window = time_feature_extraction(filt_window, time_features)
    
    if FREQ:
        def freq_feature_extraction(window, features_dic):
            # will be updated with keys as data columns (e.g., 'X Accel') 
            cols_dic = {}

            window_copy = window.copy()

            # Loop over data columns
            for col in data_columns:

                # will be updated with keys as extracted feature names (e.g., 'Mean')
                feats = {}

                def function_all_windows(function):

                    featured_column = function(window_copy.iloc[:, -1], window_copy[col])

                    return featured_column 

                # Execute every function over all windows    
                for feat_name, feat_func in features_dic.items():

                    # apply feature extraction to the ith column for all windows
                    feats.update({feat_name: function_all_windows(feat_func)})


                cols_dic.update({col: pd.DataFrame(feats, index=[0])})

            return cols_dic
        
        # frequency feature extraction
        freq_featured_window = freq_feature_extraction(psd_window, freq_features)
    
    if TIME or FREQ:
        def combine_extracted_columns_window(window_dic):

            def append_all_columns(columns, append_tag):
                new_columns = []

                for column in columns:
                    if append_tag not in column:
                        new_columns.append(column + ' ' + append_tag)
                    else:
                        new_columns.append(column)

                return new_columns

            # Get labels array of first column
            df_combined = pd.DataFrame()

            # Append direction name to feature name and combine everything in one frame
            for col_label, df in window_dic.items():
                df_copy = pd.DataFrame(df)

                # Add direction and placement tags
                df_copy.columns = append_all_columns(df.columns, col_label)

                df_combined = df_combined.join(df, how='outer')

            return df_combined
        
        if TIME:
            # columbing
            columned_time_feats = combine_extracted_columns_window(time_featured_window)
        else:
            columned_freq_feats = combine_extracted_columns_window(freq_featured_window)
    
    if FFT or PSD:
        def combine_transform_columns(window, trans = ''):

            window_copy = window.copy()

            # Get frequency bins from frequency column of first window
            freq_bins = window['frequency'].tolist()

            # Get more parameter for current label
            trans = trans

            # Combine parameters to form columns for new combined DataFrame
            new_cols = [trans + ' {} Hz '.format(round(f_bin, 1)) + d_col for d_col in data_columns for f_bin in freq_bins]

            # Convert windowed arrays into a single array with each window as a row
            new_data = []

            new_row = []
            for d_col in data_columns:
                new_row.extend(window_copy[d_col].tolist())
            new_data.append(new_row)

            # Create new DataFrame
            df_combined = pd.DataFrame(data=new_data, columns=new_cols)

            return df_combined
        
        if FFT:
            # combine fft
            columned_fft_window = combine_transform_columns(fft_window, 'FFT')
        else:
            # combine psd
            columned_psd_window = combine_transform_columns(psd_window, 'PSD')
    
#   columned_fft_window
#   columned_psd_window
#   columned_freq_feats
#   columned_freq_feats
    return()


# In[87]:


# create a test window
test_window = raw_datasets[dataset_labels[0]][:512][data_columns]

# determine which feature sets should be created
TIME = True
FREQ = True
FFT = True
PSD = True

# method 1
get_ipython().run_line_magic('timeit', 'transformed_window = latency_analysis(test_window, TIME, FREQ, FFT, PSD)')

# method 2
time1 = time.time()
transformed_window = latency_analysis(test_window, TIME, FREQ, FFT, PSD)
time2 = time.time()
print('data preparation time: {} ms'.format((time2-time1)*1000))

# check output
transformed_window


# ## Part 10 - Exporting Processed Data 

# In[88]:


# Processed data path with power type folder
processed_path = os.path.join(CURR_PATH, 'processed_data')

# Store feature vectors in a dictionary
vector_dict = {'TimeFeats': time_feats, 'FreqFeats': freq_feats, 'FFTs': ffts, 'PSDs': psds}

# Set in notebook parameters at top of notebook
if EXPORT_PROCESSED_DATA:
    
    # Save each vector and each placement to .csv file
    for vector_name, vector_data in vector_dict.items():
        
        # Filename using above dictionary         
            filename = os.path.join(processed_path, vector_name + '.csv')
            vector_data.to_csv(filename, index=False)

