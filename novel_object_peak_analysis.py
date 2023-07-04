# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:20:53 2023

@author: pgomez
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from scipy.signal import find_peaks
from scipy.signal import butter
from scipy.signal import filtfilt
from process_allfiles_reportFP import data_dict, summary, all_info
from configuration import dir_path,path_summary, filename_analysis, filename_behavior, framerate, window_size, bandpass, start_behavior, name_behav,distance, threshold, baseline, nor_baseline, type_detection, num_frames, drug, offset



filename_extracted = dir_path.stem + ("_extracted.csv")
save_path_analysis_behavior= Path (dir_path.joinpath(dir_path.stem +("_ALL_BEHAVIORS_analyzed.csv")))
save_path_peaks = Path (dir_path.joinpath(dir_path.stem +("_PEAKS_analyzed.csv")))
save_path_behavior_pdf = Path (dir_path.joinpath(filename_behavior))

def open_behavior (file_path):
    """
    Opens the behavior from a boris csv file 
    """
    behavior_excel= pd.read_csv(file_path)
    character_change = behavior_excel.columns.str.contains(';').any()
    if character_change == True:
        behavior = pd.read_csv(file_path, sep = ";")
    else:
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

#---------------------------------
def count_behavior(behavior, framerate):
    """
    Function that counts total time, number of events and mean latency between those events of a binary
    BORIS behavior exported file
    """
    #calculate total time
    time_behavior = float(sum(behavior)/framerate)
    
    #calculate number of events 
    in_behavioral_event = False
    behavioral_events = []
    for index, value in enumerate(behavior):
        if value == 1:
            if not in_behavioral_event:
                in_behavioral_event = True
                start_index = index
        else:
            if in_behavioral_event:
                in_behavioral_event = False
                end_index = index - 1
                behavioral_events.append((start_index, end_index))
                
    number_events =  float(len(behavioral_events))
    
    #calculate latency between events
    durations = []
    for i in range(len(behavioral_events) - 1):
        # Calculate the duration between the end of the current tuple and the start of the next tuple
        current_end = behavioral_events[i][1]
        next_start = behavioral_events[i + 1][0]
        duration = next_start - current_end
        durations.append(duration)
        
    durations_seconds = np.array(durations)/framerate
    
    average_latency =  float(np.mean(durations_seconds))

    return  time_behavior, number_events, average_latency
#---------------------------------- 

def behav_allmice (input_dict, framerate):
    """
    Function that runs the count_behavior function in all elements of the dictionary that stores the 
    values for the behavior of all mice and gives a dictionary with the resulting calculation

    """
    output_dict = {}
    # Loop through the input dictionary
    for key, value in input_dict.items():
        # Create an empty list to store the output for this key
        output_list = {}
        # Loop through the arrays for this key
        for array_dict in value:
            # Create a new dictionary to store the processed arrays
            processed_dict = {}
            # Loop through the arrays in the array dictionary
            for array_key, array_value in array_dict.items():
                # Process the array using the sample function
                time_behavior, number_events, average_latency = count_behavior(array_value, framerate)
                # Store the processed array in the new dictionary
                processed_dict[array_key] = {"total_time": time_behavior,
                                          "number_events": number_events,
                                          "average_latency": average_latency}
            
            # Append the new dictionary to the output list
            output_list.update(processed_dict)
        # Store the output list for this key in the output dictionary
        output_dict[key] = output_list
    return output_dict

#------------------

def identify_novelfamiliar (path_summary, behavior_dict):
    """
    Function identifies which objects are novel or familiar based on a previously defined excel
    """
    if path_summary is not None:
        nor_excel= pd.read_csv(path_summary)
        character_change = nor_excel.columns.str.contains(';').any()
        if character_change == True:
            dataframe = pd.read_csv(path_summary, sep = ";")
        else:
            dataframe = pd.read_csv(path_summary)
        dictionary = behavior_dict
        if path_summary is not None:    
            #FOR TEST 
            for key in dictionary.keys():
                for i in range(len(dataframe)):
                    if dataframe.loc[i]["Novel"] == "Same":
                        pass
                    else:
                        if dataframe.loc[i]['Mouse #'] == int(key):
                            if dataframe.loc[i]['Novel'] == "Right":
                                dictionary[key][0].rename(columns={'turning_left': 'turning_novel'}, inplace=True)
                                dictionary[key][0].rename(columns={'turning_right': 'turning_familiar'}, inplace=True)
                                dictionary[key][0].rename(columns={'left_object': 'novel_object'}, inplace=True)
                                dictionary[key][0].rename(columns={'right_object': 'familiar_object'}, inplace=True)
                            else:
                                dictionary[key][0].rename(columns={'turning_right': 'turning_novel'}, inplace=True)
                                dictionary[key][0].rename(columns={'turning_left': 'turning_familiar'}, inplace=True)
                                dictionary[key][0].rename(columns={'right_object': 'novel_object'}, inplace=True)
                                dictionary[key][0].rename(columns={'left_object': 'familiar_object'}, inplace=True)
    else: 
        dictionary = behavior_dict
    return dictionary

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
def detect_transients_prominence(signal, window_size, threshold, distance):
    """
    Function that detects calcium peak transients using a dynamic threshold calculated as Threshold * MAD during the specified window size
    Uses the peak prominence, meaning how much a peak stands out from the surrounding baseline of the signal and is defined as the vertical 
    distance between the peak and its lowest contour line, to determine whats a peak. 

    """
    # Calculate the median absolute deviation (MAD) in the moving window
    kernel = np.ones(window_size) / window_size
    signal = np.ravel(signal.T)
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
    peaks, properties = find_peaks(signal, prominence=thres, distance= distance)
    
    amplitude = next(iter(properties.values()))
    
    # Return the indices and values of the detected calcium events
    event_values = signal[peaks]
    
    return peaks, event_values, amplitude

#-----------------------------
def detect_transients_by_height(signal, threshold, distance):
    """
    Function that detects peaks by height from zero, and within a specific distance 

    """
    # Find peaks in the signal
    signal = np.ravel(signal.T)
    mad = np.median(np.abs(signal - np.median(signal)))
    thres = threshold * mad
    peaks, properties = find_peaks(signal, height=thres, distance=distance)
    event_values = signal [peaks]
    amplitude = next(iter(properties.values()))
    return peaks, event_values, amplitude

#------------------
def calculate_peak_properties(signal, peaks, framerate):
    # Find peaks and their properties
    peak_height = peaks [:,1]
    peak_amplitudes = peaks [:,2]
    peak_indices = peaks [:,0]
    number_peaks = len(peak_indices)
    peak_frequencies = (number_peaks / len(signal)) * framerate

    return peak_height, peak_amplitude, peak_frequencies
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


#-----------------

def extract_peaks(signal, behavior, peaks, num_frames):
    peak_amplitudes = peaks [:,2]
    
    behav_indices = np.nonzero(behavior)[0]
    
    # Extend the exploration indices by the specified number of frames before and after
    behav_idx = np.concatenate([np.arange(index - num_frames, index + num_frames + 1) for index in behav_indices])
    
    # Remove any out-of-bounds indices
    behav_idx = behav_idx[np.logical_and(behav_idx >= 0, behav_idx < len(signal))]
  
    # Initialize lists to store peak values 
    beh_peaks = []
    behav_peak_indices = []
    behav_amplitude = []
    

    # Iterate over the extended behavior indices and check if the index is in the peaks
    for index in behav_idx:
        if index in peaks and index not in behav_peak_indices:
            beh_peaks.append(signal[index])
            behav_peak_indices.append(index)
            idx_peak = np.where (peaks [:,0] == index)
            behav_amplitude.append (float(peak_amplitudes[idx_peak]))
            
    result_beh_idx = np.array(behav_peak_indices)  
    result_beh_events = np.array(beh_peaks)
    result_behav_amplitude = np.array (behav_amplitude)
    result = np.column_stack ((result_beh_idx, result_beh_events, result_behav_amplitude))
    
    return result


#-----------
def generate_summary (d, path_summary):
    """
    Creates summary file with all the parameters of the NOR 
    """
    nor_excel= pd.read_csv(path_summary)
    character_change = nor_excel.columns.str.contains(';').any()
    if character_change == True:
        summary= pd.read_csv(path_summary, sep = ";")
    else:
        summary = pd.read_csv(path_summary)
    dfs = []
    for key in d:
        novel_object = d[key]['novel_object']
        familiar_object = d[key]['familiar_object']
        column_names_novel = [f"{col}_Novel Object" for col in novel_object.keys()]
        df_novel = pd.DataFrame([novel_object.values()], columns=column_names_novel)
        column_names_familiar = [f"{col}_Familiar Object" for col in familiar_object.keys()]
        df_familiar = pd.DataFrame([familiar_object.values()], columns=column_names_familiar)
        df_summary = pd.concat([df_novel, df_familiar], axis=1).T.sort_index(ascending=False).T
        dfs.append(df_summary)
    final_summary = pd.concat(dfs, axis=0, ignore_index=True)
    all_mice = pd.concat ([summary, final_summary], axis = 1)
    all_mice.to_csv(path_summary)
    return all_mice

#--------------------------
def generate_excel (d, save_path):
    """
    Takes the dictionary, transforms it into a dataframe with the name of the behaviors

    """
    data = {}
    for key in d:
        subdict = d[key]
        for event in subdict:
            if event == 'time':
                continue
            new_key = event + '_' + key
            event_dict = subdict[event]
            data.setdefault(new_key, {})
            for subkey in event_dict:
                data[new_key][subkey] = event_dict[subkey]
    df = pd.DataFrame.from_dict(data, orient='index').sort_index()
    df.to_csv(save_path, index=True)
    return df
#--------------------------

def analyze_all_behavior (folder_path, summary_path, name_behav, framerate, save_path_analysis):
    behavior_dict = take_all_files (folder_path, name_behav)
    new_behav_dict = identify_novelfamiliar (summary_path, behavior_dict)
    all_behav_summary = behav_allmice (new_behav_dict, framerate)
    if path_summary is not None:
        nor_summary = generate_summary (all_behav_summary, path_summary) 
    else:
        full_dic = generate_excel (all_behav_summary, save_path_analysis)
    return new_behav_dict


#------------------------
def postprocessing_signal (merged_dict, baseline, nor_baseline, start_behavior, bandpass, window_size, threshold, distance, num_frames):
    processed_dictionary = {}
    final_zscore_dict = {}
    final_peak_dict = {}
    for key, value in merged_dict.items():
        behavior = value[0]
        extracted_deltafof= value[1].iloc[:, -2:-1]     #from dictionary get AF/F
        start_behav_data= np.array(behavior[start_behavior])
        start_nor = np.nonzero (start_behav_data)[0][0]  #see when NOR starts
            
        #depending on baseline you analyze it in a way or other
        if baseline is None:
            zscore = baseline_z_score(extracted_deltafof, start_nor-nor_baseline, start_nor)
        else:
            zscore = baseline_z_score(extracted_deltafof, 0, baseline)
        #apply bandpass filter or nt
        if bandpass is None:
            filtered_zscore = zscore
        else:
            filtered_zscore = apply_butterpass (np.array(zscore).T, bandpass['lowcut'], bandpass['highcut'], framerate, bandpass['order'])
        
        filtered_zscore = np.ravel(filtered_zscore.T)

        peak_dict = {}    
        if type_detection == "prominence":
            all_peaks, event_values, amplitude = detect_transients_prominence(filtered_zscore, window_size, threshold, distance)
        else:
            all_peaks, event_values, amplitude = detect_transients_by_height(filtered_zscore, threshold, distance)
        
        peaks = np.column_stack ((all_peaks, event_values, amplitude))
    
        for column in behavior.columns[1:]:
            all_peak_events = extract_peaks(filtered_zscore, np.array(behavior[column]), peaks, num_frames)
            peak_dict [column] = all_peak_events

        processed_dictionary [key] = {"zscore": filtered_zscore, "behavior": behavior, "all_peaks": peaks, "peaks_per_behavior": peak_dict}
    
    return processed_dictionary


#-------------------------

def peaks_to_excel(peaks_per_behavior, save_path):
    result_df = pd.DataFrame()
    for key in new_dict:
        peaks_per_behavior = new_dict [key]["peaks_per_behavior"]
        df = pd.DataFrame()
        for behavior, data_array in peaks_per_behavior.items():
            behavior_df = pd.DataFrame(data_array, columns=[f"{key}_{behavior}_peak_index", f"{key}_{behavior}_peak_height", f"{key}_{behavior}_peak_amplitude"])
            df = pd.concat([df, behavior_df], axis=1)
        result_df = pd.concat ([result_df, df], axis =1)
    pd.DataFrame(result_df).to_csv (save_path)
    return result_df


#------------------------------------
def calculate_peaks_parameters_injection (new_dict, drug, offset, framerate):
    for key, value in new_dict.items():
        zscore = new_dict[key]["zscore"]
        behavior = new_dict[key]["behavior"]
        peaks= new_dict[key]["all_peaks"][:, 0]
        properties = new_dict[key]["peak_properties"]
        injection= np.nonzero(behavior["injection"])[0]
        start_injection = injection[0]
        finish_injection = injection[-1]
        
        start_baseline_injection = start_injection - (drug["baseline_start"]*framerate)
        stop_baseline_injection = start_injection - (drug["baseline_end"]*framerate)
        stop_injection_period = finish_injection + (offset*framerate)
        
        baseline_injection_zscore = zscore [start_baseline_injection:stop_baseline_injection]
        baseline_injection_peaks = (peaks [[(peaks >= start_baseline_injection) & (peaks <= stop_baseline_injection)]]- start_baseline_injection).astype(int)
        baseline_amplitude, baseline_frequency = calculate_peak_properties(baseline_injection_zscore, baseline_injection_peaks, properties, framerate)
        
        post_injection_zscore = zscore [finish_injection:stop_injection_period]
        post_injection_peaks = (peaks [[(peaks >= finish_injection) & (peaks <= stop_injection_period)]]- finish_injection).astype(int)
        post_amplitude, post_frequency = calculate_peak_properties(post_injection_zscore, post_injection_peaks, properties, framerate)
        
#-------------------------
def group_by_condition (animal_data, all_info):
    grouped_data = {}
    for mouse, condition in zip(all_info['mouse #'], all_info['condition']):
        if str(mouse) in animal_data:
            if condition in grouped_data:
                if str(mouse) in grouped_data[condition]:
                    grouped_data[condition][str(mouse)].extend(animal_data[str(mouse)])
                else:
                    grouped_data[condition][str(mouse)] = animal_data[str(mouse)]
            else:
                grouped_data[condition] = {str(mouse): animal_data[str(mouse)]}
        else:
            grouped_data[condition] = {}
    return grouped_data

#------------------------

merged_behavior = analyze_all_behavior (dir_path, path_summary, name_behav, framerate, save_path_analysis_behavior)
merged_dict = match_behavior_data (merged_behavior, data_dict)
new_dict = postprocessing_signal (merged_dict, baseline, nor_baseline, start_behavior, bandpass, window_size, threshold, distance, num_frames)         
peaks_to_excel(new_dict, save_path_peaks)
condition_dict = group_by_condition (new_dict, all_info)


