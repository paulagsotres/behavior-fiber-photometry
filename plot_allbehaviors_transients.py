# -*- coding: utf-8 -*-
"""
Created on Tue May 23 16:25:55 2023

@author: pgomez
"""
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks
from scipy.signal import butter
from scipy.signal import filtfilt
from process_allfiles_reportFP import data_dict, summary, all_info
from configuration import dir_path, filename_analysis, filename_behavior, framerate, window_size, bandpass, start_behavior, name_behav,distance, threshold, baseline

filename_extracted = dir_path.stem + ("_extracted.csv")
save_path_analysis= Path (dir_path.joinpath(filename_analysis))
save_path_behaviorpdf = Path (dir_path.joinpath(filename_behavior))


def open_behavior (file_path):
    """
    Opens the behavior from a boris csv file 
    """
    behavior = pd.read_csv(file_path)
    return behavior #specific to open keys from a h5py

def take_all_files (dir_path, name_behav):
    """
    Function to take all the files that contain files with item specified in config as "name_behav" in the parent folder, and takes BORIS binary data from all behaviors
    Returns a dictionary with the mouse number and the behaviors extracted for each file

    """
    # Create an empty dictionary to store the results
    behavior_dict = {}
    # Iterate over all files in the directory
    for file_path in Path(dir_path).glob(name_behav):
        # Extract the mouse number from the file name
        mouse_num = file_path.name.split("_")[0]
        # Call the extract_behaviors_file function to get a dictionary of variables for this file
        variables_dict = open_behavior(str(file_path))
        # Check if the mouse number is already a key in the results dictionary
        if mouse_num in behavior_dict:
            # If it is, append the variables dictionary to the list of dictionaries for that mouse
            behavior_dict[mouse_num].append(variables_dict)
        else:
            # If it's not, create a new list with the variables dictionary and store it in the results dictionary
            behavior_dict[mouse_num] = [variables_dict]
       
            
    return behavior_dict

behavior_dict = take_all_files (dir_path, name_behav)

#-------------------------------
def baseline_z_score(signal, baseline_start, baseline_end):
    """
    Calculates the baseline z-score of a signal by subtracting the mean and dividing by the standard deviation of
    a designated baseline period.

    Args:
        signal (numpy array): The signal to calculate the baseline z-score of.
        baseline_start (int): The starting index of the baseline period in the signal.
        baseline_end (int): The ending index of the baseline period in the signal.

    Returns:
        numpy array: The baseline z-scored signal.
    """
    baseline_signal = signal[baseline_start:baseline_end]
    mean_baseline = np.mean(baseline_signal)
    std_baseline = np.std(baseline_signal)
    baseline_z_score = (signal - mean_baseline) / std_baseline

    return baseline_z_score
#-------------------------------


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut/nyq  # Normalized lower cutoff frequency
    high = highcut/nyq  # Normalized upper cutoff frequency
    b, a = butter(order, [low, high], btype='band')  # Compute filter coefficients
    return b, a

def apply_butterpass (signal, lowcut, highcut, framerate, order):
    b, a = butter_bandpass(lowcut, highcut, framerate, order)
    filtered = filtfilt(b, a, signal)
    return filtered 



#------------------------------
def detect_transients_prominence(signal, window_size, threshold):
    """
    Function that detects calcium peak transients using a dynamic threshold calculated as Threshold * MAD during the specified window size
    Uses the peak prominence, meaning how much a peak stands out from the surrounding baseline of the signal and is defined as the vertical 
    distance between the peak and its lowest contour line, to determine whats a peak. 

    """
    # Calculate the median absolute deviation (MAD) in the moving window
    kernel = np.ones(window_size) / window_size
    
    # Calculate the median absolute deviation (MAD) in the moving window
    mad = np.zeros(signal.shape)
    for i in range(signal.shape[0]):
        start = max(0, i - window_size // 2)
        end = min(signal.shape[0], i + window_size // 2)
        mad[i] = np.median(np.abs(signal[start:end] - np.median(signal[start:end])))
    
    # Convolve the signal with the kernel to obtain the moving average
    moving_avg = np.convolve(signal, kernel, mode='same')
    # Use the MAD as a threshold to detect calcium events in the signal
    thres = threshold * mad
    peaks, properties = find_peaks(signal, prominence=thres)
    
    # Return the indices and values of the detected calcium events
    event_values = signal[peaks]
    
    return peaks, event_values

#-----------------------------
def detect_transients_by_height(data, threshold, distance):
    """
    Function that detects peaks by height from zero, and within a specific distance 

    """
    # Find peaks in the signal
    mad = np.median(np.abs(data - np.median(data)))
    thres = threshold * mad
    peaks, _ = find_peaks(data, height=thres, distance=distance)
    event_values = data [peaks]
    return peaks, event_values

#--------------------

def match_behavior_data (behavior_dict, data_dict):
    extracteddata_dict = {}
    
    for key, value in data_dict.items():
            extracteddata_dict[key] = value.iloc[:, -2:]
    
    matched_dict = {}
    
    for key in behavior_dict:
        if key in extracteddata_dict:
            value1 = behavior_dict[key]
            value2 = extracteddata_dict[key]
            matched_dict[key] = [value1, value2]
    
    matched_dict_corrected = {}
    for key in matched_dict:
        length_signal = len (matched_dict[key][1])
        difference = len (matched_dict[key][0][0])- length_signal -1
        behavior_corrected = matched_dict[key][0][0].iloc [difference:-1, :]
        matched_dict_corrected[key]= [behavior_corrected, matched_dict[key][1]]
        
    return matched_dict_corrected

dict3 = match_behavior_data (behavior_dict, data_dict)

#------------------------

    
def plot_signal_behavior_peaks(timestamp, signal, behavior, peaks, window_size, threshold, distance):
    fig, ax = plt.subplots(figsize=(10, 3), sharex = True)
    ax.plot (peaks, signal[peaks], "r+")
    ax.plot(signal, c = 'black', label='Zscore')
    legends = []
    # Get the column names starting from the second column
    for column in behavior:
        if column == "time":
            columns = behavior.columns[1:]
            break
        else:
            columns = behavior.columns

    for i, column in enumerate(columns):
        events =np.where(behavior[column]>0)[0]
        for j in range(len(events)//2):
            start_idx = events[2*j]
            color = plt.cm.get_cmap('tab10')(i)  # Get a color based on column index
            rect = plt.Rectangle((start_idx, np.min(signal)), 1, np.max(signal)-np.min(signal), color=color, alpha=0.2)
            ax.add_patch(rect)
    
        legends.append((rect, column))
        
    ax.xaxis.set_major_locator(MultipleLocator(1200))
    ax.set_ylabel("Zscore")
    ax.set_xlabel("Time(s)")
    ax.legend(*zip(*legends), loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    return fig, ax

#----------------------------


def all_plots_behavior(merged_dict, save_path_behaviorpdf, window_size, threshold, distance, framerate, bandpass, baseline):
    pdf_pages = PdfPages(save_path_behaviorpdf)
    for key in merged_dict:
        behavior = merged_dict [key][0]
        timestamp = behavior.iloc [:, 0]
        signal = merged_dict [key][1]
        deltafof = signal.iloc [:, 0]
        
        if baseline is None:
            zscore =  signal.iloc [:, 1]
        else:
            zscore = baseline_z_score(deltafof, 0, baseline)

        filtered_zscore = apply_butterpass (np.array(zscore).T, bandpass['lowcut'], bandpass['highcut'], framerate, bandpass['order'])
        peaks_idx, event_values = detect_transients_prominence(filtered_zscore, window_size, threshold)
        fig, ax = plot_signal_behavior_peaks(timestamp, filtered_zscore, behavior, peaks_idx, window_size, threshold, distance)
        ax.set_xticklabels(ax.get_xticks() // framerate)
        fig.suptitle(str(key))
        pdf_pages.savefig(fig, bbox_inches='tight', dpi=150, metadata={'Title': str(key)})
    
    pdf_pages.close()
    plt.show()
        
        
all_plots_behavior(dict3, save_path_behaviorpdf, window_size, threshold, distance, framerate, bandpass, baseline)



#-----------------
def AUC_extracted_signal (extracted_signal, time_exposure, time_before_onset):
    start_baseline = 0
    stop_baseline = time_before_onset*framerate

    
    start_exposure = time_before_onset*framerate
    stop_exposure = start_exposure + (time_exposure*framerate)

    
    # Calculate area under the curve for baseline and exposure periods
    baseline_area = np.trapz(extracted_signal[start_baseline:stop_baseline])
    exposure_area = np.trapz(extracted_signal[start_exposure:stop_exposure])
    
    return baseline_area, exposure_area