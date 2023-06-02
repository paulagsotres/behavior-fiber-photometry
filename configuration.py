# -*- coding: utf-8 -*-
"""
Created on Tue May 23 17:32:34 2023

@author: pgomez
"""

from pathlib import Path


# parameters to change
dir_path = Path(r"C:\Users\pgomez\Desktop\dlx cre") #change to point to folder
path_summary = r"C:\Users\pgomez\Desktop\dlx cre\Summary.csv" #change to point to summary file for NOR
filename_analysis = dir_path.stem + ("_ALL_MICE_analyzed.csv") #if you want to change the output name of the files 
filename_pdf = dir_path.stem + ("_ALL_MICE_analyzed.pdf") #change to your liking
filename_behavior = dir_path.stem + ("_ALL_MICE_behavior.pdf")
name_behav = "*behavior*.csv" #what item do your behavioral files have to identify them


#adjust parameters to your experiment (if you dont need them dont change them)
filter_window = 10 #average moving window to calculate Fmean in frames
framerate = 20 #of data acquired
photobleach = 1200 #time to remove because of the photobleaching in frames
baseline = None #simple baseline in frames in case your experiment doesn't have specific behaviors, if specific baseline put None
type_smoothing = None #or "savitzky" or None if you dont want to smooth

#peak detection
window_size = 2400 #to calculate dynamic MAD instead of in the whole recording (put original number of frames if you want the whole recording)
threshold = 3 #times MAD
distance = 100 #Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
type_detection = "threshold" #vs "threshold"
num_frames = 25 #to take before and after behavioral event


# Define the bandpass filter parameters
bandpass = {"lowcut": 0.01, "highcut":7, "order": 4} #add None if you dont use 
# Lower cutoff frequency in Hz
# Upper cutoff frequency in Hz
# Filter order


#extract signal for long behaviors
time_before_onset = 15
time_exposure = 30
time_after_onset = 60

#baseline for injections in seconds 
drug = {"baseline_start": -120, "baseline_end": -60} #baseline is how much time before injection you want to measure before you mark the mouse back in the cage
vehicle = {"baseline_start": -120, "baseline_end": -60} 

#starting behavior marking your interest period
start_behavior = "turning_novel"


#vs nor_position = "Up"
nor_baseline = 2400 #time before nor starts to take as baseline