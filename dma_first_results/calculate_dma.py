import os
from scipy.io import loadmat
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dfa_wrapper import get_log_log_plot_bending_point, calculate_alpha_exp 

import sys
sys.path.append('C:\\Users\\bohdank\\Documents\\MyProjects\\stabilometry_analysis')
from record_model import Record
from feature_extractor_model import FeatureExtractor
from record_features_model import RecordFeatures

import dma as DMA


def get_file_pathes_in_dir(path: str, extension: str):
    file_names = []
    for file_name in os.listdir(path):
        if file_name.endswith(f".{extension}"):
            file_names.append(f"{path}/{file_name}")
    return file_names


def extract_records_from_mat_files(mat_file_pathes):
    records_dict = {}
    for file_path in mat_file_pathes:
        mat_data = loadmat(file_path).get('s')[0]
        file_name = extract_name_from_path(file_path)

        patient_records = []
        for mat_data_element in mat_data:
            record = Record(mat_data_element, file_name)
            if "ОС ЗГ" in record.record_name or "ОС ОГ" in record.record_name:
                patient_records.append(record)
        records_dict[file_name] = patient_records

    return records_dict


def extract_name_from_path(path):
    file_name_with_extension = path.split('/')[-1]
    return file_name_with_extension.split('.')[0]



# def func(x, a0, a1, a2, a3):
#     # Function that includes x0 (bending point) as a parameter
#     x00 = 0
#     I = np.ones_like(x)
#     I[np.argwhere(x <= x00)] = 0
#     global x0
#     return a0 * I + a1 * I * x + a2 * (1 - I) + a3 * x * (1 - I)

# def aic(y_observed, y_predicted, k):
#     return 2 * k + y_predicted.size * np.log(np.sum(np.sqrt(((y_predicted - y_observed) ** 2)))/y_predicted.size)


# def get_log_log_plot_bending_point(xvals=None, yvals=None):

#     aic_vals = []
#     x_points = []

#     from scipy.optimize import curve_fit

#     for i, point in enumerate(xvals):

#         if i > 2:
#             global x0
#             x0 = point
#             popt, pcov = curve_fit(func, xvals, yvals)  # fitting our model to the observed data
#             y_predicted = func(xvals, *popt)  # calculating predictions, given by our fitted model
#             aic_vals.append(aic(y_observed=yvals, y_predicted=y_predicted, k=5))
#             x_points.append(point)

#     bending_point = np.min(aic_vals)
#     x_bend_p = aic_vals.index(bending_point)

#     return int(np.exp(x_points[x_bend_p])), x_bend_p


# log_F = "-1.443273 -1.006635 -0.745428 -0.564785 -0.430679 -0.326343 -0.242276 -0.172616 -0.11348  -0.061906 -0.015629  0.026708  0.065418  0.100505 0.132186  0.18726   0.211581  0.254717  0.292018  0.324776  0.353348 0.377794  0.407224  0.431304"
# log_n = "0.477121 0.69897  0.845098 0.954243 1.041393 1.113943 1.176091 1.230449 1.278754 1.322219 1.361728 1.39794  1.431364 1.462398 1.491362 1.544068 1.568202 1.612784 1.653213 1.690196 1.724276 1.755875 1.799341 1.838849"
# log_F = [float(el) for el in log_F.split()]
# log_n = [float(el) for el in log_n.split()]

# last_index = len(log_F) - 1
# tg_a_vector = []
# for curr_index, value in enumerate(log_F):

#     next_index = curr_index + 1
#     if next_index == last_index:
#         break
#     tg_a = (log_F[next_index] - log_F[curr_index])/(log_n[next_index] - log_n[curr_index])
#     tg_a_vector.append(tg_a)
#     print(value, curr_index, tg_a)
# plt.plot(tg_a_vector)
# plt.show()

def plot_dma(log_n, log_F, title):
        (_, index) = get_log_log_plot_bending_point(log_n, log_F)
        alpha = calculate_alpha_exp(log_n, log_F, index)[0]
        # b = log_F[index] - alpha*log_n[index]
        # y = alpha * log_n + b
        # plt.plot(log_n, y)
        p = plt.plot(log_n, log_F, label=f"{title}: alpha={alpha}, index={index}")
        plt.plot(log_n[index], log_F[index], color=p[0].get_color(), marker='o')


folder_names = ["rowing", "water_jumps", "healthy"]

for folder_name in folder_names:
    mat_file_pathes = get_file_pathes_in_dir(f"C:/Users/BohdanK/Dropbox/StabiloData/{folder_name}", extension="mat")
    records_dict = extract_records_from_mat_files(mat_file_pathes)

    log_F_open_x_list = []; log_F_open_y_list = []; log_F_open_2d_list = []
    log_F_closed_x_list = []; log_F_closed_y_list = []; log_F_closed_2d_list = []
    log_n_for_mean_1d = []; log_n_for_mean_2d = []
    for filename, records_array in records_dict.items(): 
        fig = plt.figure()
        fig.suptitle(f"{folder_name}: {filename}")
        fig.set_size_inches((8, 11)) 

        if folder_name == "healthy":
                _break = 'here' 
        plt.subplot(2,1,1)
        open_eyes_record = records_array[0]
        (log_n, log_F) = DMA.dma_d1(open_eyes_record.cop.x); log_n_for_mean_1d = log_n
        if log_F.size in (24,25): log_F_open_x_list.append(log_F)
        plot_dma(log_n, log_F, 'DMA_X')
        (log_n, log_F) = DMA.dma_d1(open_eyes_record.cop.y)
        if log_F.size in (24,25): log_F_open_y_list.append(log_F)
        plot_dma(log_n, log_F, 'DMA_Y')
        (log_n, log_F) = DMA.dma_d2(open_eyes_record.cop.x, open_eyes_record.cop.y); log_n_for_mean_2d = log_n
        if log_F.size == 27: log_F_open_2d_list.append(log_F)
        plot_dma(log_n, log_F, 'DMA_2D')
        plt.legend(); plt.grid(); plt.title("Open eyes")
        
        plt.subplot(2,1,2)
        closed_eyes_record = records_array[1]
        (log_n, log_F) = DMA.dma_d1(closed_eyes_record.cop.x) 
        if log_F.size in (24,25): log_F_closed_x_list.append(log_F)
        plot_dma(log_n, log_F, 'DMA_X') 
        (log_n, log_F) = DMA.dma_d1(closed_eyes_record.cop.y)
        if log_F.size in (24,25): log_F_closed_y_list.append(log_F)
        plot_dma(log_n, log_F, 'DMA_Y')
        (log_n, log_F) = DMA.dma_d2(closed_eyes_record.cop.x, closed_eyes_record.cop.y)
        if log_F.size == 27: log_F_closed_2d_list.append(log_F)
        plot_dma(log_n, log_F, 'DMA_2D')
        plt.legend(); plt.grid(); plt.title("Closed eyes")

        plots_path = f"C:\\Users\\bohdank\\Documents\\MyProjects\\stabilometry_analysis\\dma_first_results\\dma_plots\\{folder_name}"
        plt.savefig(f"{plots_path}\\{filename}.png")
        plt.close()
        
    fig = plt.figure()
    fig.suptitle(folder_name)
    fig.set_size_inches((8, 11)) 

    plt.subplot(2,1,1)
    log_F_open_x_mean = np.mean(log_F_open_x_list, axis=0)
    plot_dma(log_n_for_mean_1d, log_F_open_x_mean, f'mean DMA_X (N = {len(log_F_open_x_list)})')
    log_F_open_y_mean = np.mean(log_F_open_y_list, axis=0)
    plot_dma(log_n_for_mean_1d, log_F_open_y_mean, f'mean DMA_Y (N = {len(log_F_open_y_list)})')
    log_F_open_2d_mean = np.mean(log_F_open_2d_list, axis=0)
    plot_dma(log_n_for_mean_2d, log_F_open_2d_mean, f'mean DMA_2D (N = {len(log_F_open_2d_list)})')
    plt.legend(); plt.grid(); plt.title("Open eyes")
    
    plt.subplot(2,1,2)
    log_F_closed_x_mean = np.mean(log_F_closed_x_list, axis=0)
    plot_dma(log_n_for_mean_1d, log_F_closed_x_mean, f'mean DMA_X (N = {len(log_F_closed_x_list)})')
    log_F_closed_y_mean = np.mean(log_F_closed_y_list, axis=0)
    plot_dma(log_n_for_mean_1d, log_F_closed_y_mean, f'mean DMA_Y (N = {len(log_F_closed_y_list)})')
    log_F_closed_2d_mean = np.mean(log_F_closed_2d_list, axis=0)
    plot_dma(log_n_for_mean_2d, log_F_closed_2d_mean, f'mean DMA_2D (N = {len(log_F_closed_2d_list)})')
    plt.legend(); plt.grid(); plt.title("Closed eyes")
    
    plots_path = "C:\\Users\\bohdank\\Documents\\MyProjects\\stabilometry_analysis\\dma_first_results\\dma_plots"
    plt.savefig(f"{plots_path}\\{folder_name}.png")
    plt.close()

        