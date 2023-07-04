# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 19:56:34 2023

@author: pgomez
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from scipy import stats
from configuration import time_before_onset, time_exposure, time_after_onset, drug, offset, start_behavior, framerate
from process_allfiles_reportFP import all_info
from novel_object_peak_analysis import new_dict, condition_dict, calculate_peak_properties


def plot_signal_behavior_peaks(signal, behavior, peaks, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
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
    
    
#-------------------------
def plot_signal_and_behavior(signal, behavior, ax= None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    # Plot signal
    ax.plot(signal, c = 'black', label='Zscore')
    legends = []
    for column in behavior:
        if column == "time":
            columns = behavior.columns[1:]
            break
        else:
            columns = behavior.columns
    
    for col in behavior.columns: 
        selected_columns = []
        for column in behavior.columns:
            first_word = column.split()[0]  # Extract the first word of the column name
            matching_columns = [col for col in df.columns if col.startswith(first_word)]
            if len(matching_columns) >= 2:  # Only consider if at least 2 columns have the same first word
                selected_columns.extend(matching_columns)
    
            new_df = df[selected_columns]

    
            for row in new_df:
                # get the start and stop indexes from the row
                start_idx, stop_idx = row[0], row[1]
                # calculate the width and height of the rectangle
                width = stop_idx - start_idx
                height = np.min(signal) -  np.max(signal)
                color = plt.cm.get_cmap('tab10')(i) 
                # create a rectangle and add it to the axis
                rect = plt.Rectangle((start_idx, 0), width, height, color=color, alpha=0.5)
                ax.add_patch(rect)


        legends.append((rect, column))
        
    ax.xaxis.set_major_locator(MultipleLocator(1200))
    ax.set_ylabel("Zscore")
    ax.set_xlabel("Time(s)")
    ax.legend(*zip(*legends), loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
#--------------------------
    
def plot_familiar_novel (new_dict, start_behavior, framerate):
    for key, value in new_dict.items():
        fig_novel, axs_novel= plt.subplots(2, 1, figsize=(10, 6), sharex = True)
        zscore = new_dict[key]["zscore"]
        behavior = new_dict[key]["behavior"]
        peaks= new_dict[key]["all_peaks"][:,0]    #from dictionary get AF/F
        start_behav_data= behavior[start_behavior]
        start_nor = np.nonzero (start_behav_data)[0][0]
        
        nor_zscore = zscore [start_nor:]
        nor_behavior = behavior [start_nor:]
        turning = nor_behavior[["turning_novel", "turning_familiar"]].copy()
        object_exploration = nor_behavior[["novel_object", "familiar_object"]].copy()
        
        nor_peaks = (peaks [peaks >= start_nor] - start_nor).astype(int)
        plot_signal_behavior_peaks(nor_zscore, turning, nor_peaks, axs_novel[0])
        plot_signal_behavior_peaks(nor_zscore, object_exploration, nor_peaks, axs_novel[1])
        
        axs_novel[0].set_title("Turning")  # Set title for the upper subplot
        axs_novel[1].set_title("Object exploration")
        axs_novel[0].set_xticklabels(axs_novel[0].get_xticks() // framerate)
        axs_novel[1].set_xticklabels(axs_novel[1].get_xticks() // framerate)
        fig_novel.suptitle(f"Mouse #{str(key)}", fontsize=14)
        
      
        fig_novel.subplots_adjust(hspace=0.4)
        fig_novel.subplots_adjust(top=0.9)
        for ax in axs_novel:
            ax.spines['bottom'].set_color('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axhline(0, color="black", linestyle='-', linewidth = 0.5)
            
#--------------------------------
            

def plot_injection (new_dict, framerate, drug, offset):
    for key, value in new_dict.items():
        fig, ax = plt.subplots(figsize=(10, 3), sharex = True)
        zscore = new_dict[key]["zscore"]
        behavior = new_dict[key]["behavior"]
        peaks= new_dict[key]["all_peaks"][:, 0]
        injection= np.nonzero(behavior["injection"])[0]
        start_injection = injection[0]
        finish_injection = injection[-1]
        
        start_baseline_injection = start_injection - (drug["baseline_start"]*framerate)
        stop_injection_period = finish_injection + (offset*framerate)
        injection_zscore = zscore [start_baseline_injection:stop_injection_period]
        injection_allbehavior = behavior [start_baseline_injection:stop_injection_period]
        injection_only = injection_allbehavior [["injection"]].copy
        
        injection_peaks = (peaks [[(peaks >= start_baseline_injection) & (peaks <= stop_injection_period)]]- start_baseline_injection).astype(int)
        ax.plot (injection_peaks, injection_zscore[injection_peaks], "r+")
        ax.plot(injection_zscore, c = 'black')
 

        rect = plt.Rectangle((start_injection- start_baseline_injection, np.min(injection_zscore)), (start_injection-finish_injection), np.max(injection_zscore)-np.min(injection_zscore), color="grey", alpha=0.7)
        ax.add_patch(rect)
        
        legend_label = 'Injection'
        ax.legend([rect], [legend_label])
        ax.set_ylabel("Zscore")
        ax.set_xlabel("Time(s)")

        
        ax.set_xticklabels(ax.get_xticks() // framerate)
        fig.suptitle(f"Mouse #{str(key)}", fontsize=14)
        fig.subplots_adjust(hspace=0.9)
        fig.subplots_adjust(top=0.9)

        ax.spines['bottom'].set_color('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0, color="black", linestyle='-', linewidth = 0.5)
        
#-----------------

def analyze_peaks_per_behavior(condition_dict, behavior):
    all_data_condition = {}
    for key, value in condition_dict.items():
        all_peak_mean_amp_height = []
        all_peak_sem_amp_height = []
        all_peak_mean_amp_prominence = []
        all_peak_sem_amp_prominence = []
        all_peak_mean_amp_prominence = []
        for key2, value2 in value.items():
            peaks_behavior = value2 ["peaks_per_behavior"][f"{behavior}"]
            peak_mean_amp_height = np.mean (peaks_behavior [:,1])
            peak_sem_amp_height = stats.sem (peaks_behavior [:,1])
            peak_mean_amp_prominence = np.mean (peaks_behavior [:,2])
            peak_sem_amp_prominence = stats.sem (peaks_behavior [:,2])
            all_peak_mean_amp_height.append (peak_mean_amp_height)
            all_peak_sem_amp_height.append (peak_sem_amp_height)
            all_peak_mean_amp_prominence.append(peak_mean_amp_prominence)
            all_peak_sem_amp_prominence.append (peak_sem_amp_prominence)
        
        all_data_condition [key] = {'mean_peak_height':all_peak_mean_amp_height, 'sem_peak_height':all_peak_sem_amp_height, 'mean_peak_prominence': all_peak_mean_amp_prominence, 'sem_peak_prominence': all_peak_sem_amp_prominence}
    return all_data_condition

#------------------



    
def plots_peaks_analyzed_behavior (behav_peaks_dict):

    turning = {}
    turning_novel = analyze_peaks_per_behavior(condition_dict, behavior="turning_novel")
    turning_familiar = analyze_peaks_per_behavior(condition_dict, behavior="turning_familiar")
    turning = {"turning_novel": turning_novel, "turning_familiar": turning_familiar}

    exploration = {}
    novel_object = analyze_peaks_per_behavior(condition_dict, behavior="novel_object")
    familiar_object = analyze_peaks_per_behavior(condition_dict, behavior="familiar_object")
    exploration = {"novel_object": novel_object, "familiar_object": familiar_object}

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=False, sharey=True, constrained_layout=True)

    def generate_bar_plots(data, ax_height, ax_prominence):
        groups = list(data.keys())  # Get the top-level keys in the dictionary
        metrics = list(data[groups[0]].keys())  # Get the second-level keys in the dictionary

        # Extracting data from the dictionary
        mean_height = {}
        sem_height = {}
        mean_prominence = {}
        sem_prominence = {}

        for metric in metrics:
            mean_height[metric] = [np.mean(data[group][metric]['mean_peak_height']) for group in groups]
            sem_height[metric] = [np.std(data[group][metric]['mean_peak_height']) / np.sqrt(len(data[group][metric]['mean_peak_height'])) for group in groups]
            mean_prominence[metric] = [np.mean(data[group][metric]['mean_peak_prominence']) for group in groups]
            sem_prominence[metric] = [np.std(data[group][metric]['mean_peak_prominence']) / np.sqrt(len(data[group][metric]['mean_peak_prominence'])) for group in groups]

        # Configuring the plot
        bar_width = 0.35
        index = np.arange(len(groups))

        for i, metric in enumerate(metrics):
            ax_height.bar(index + i * bar_width, mean_height[metric], bar_width, label=metric, yerr=sem_height[metric], capsize=3)
            ax_prominence.bar(index + i * bar_width, mean_prominence[metric], bar_width, label=metric, yerr=sem_prominence[metric], capsize=3)

        ax_height.set_ylabel('mean peak height (z-score)') 
        ax_height.set_xticks(index + (len(metrics) - 1) * bar_width / 2)
        ax_height.set_xticklabels(groups)
        ax_height.legend()

        ax_prominence.set_ylabel('mean peak prominence (z-score)')
        ax_prominence.set_xticks(index + (len(metrics) - 1) * bar_width / 2)
        ax_prominence.set_xticklabels(groups)
        ax_prominence.legend()
        

        ax_height.spines['top'].set_visible(False)
        ax_height.spines['right'].set_visible(False)
    

        ax_prominence.spines['top'].set_visible(False)
        ax_prominence.spines['right'].set_visible(False)

        
        
        plt.show()
   

    generate_bar_plots(turning, axs[0, 0], axs[0, 1])
    generate_bar_plots(exploration, axs[1, 0], axs[1, 1])


plot_familiar_novel (new_dict, start_behavior, framerate)
plot_injection (new_dict, framerate, drug, offset)  
plots_peaks_analyzed_behavior (condition_dict)
