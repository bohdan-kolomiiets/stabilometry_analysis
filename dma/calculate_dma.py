import os
from scipy.io import loadmat
import json
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import numpy as np

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


def plot_dma(log_n, log_F, title):
        (alpha, b, index) = DMA.calc_scaling_exponent(log_n, log_F)

        p = plt.plot(log_n, log_F, label=f"{title}, alpha={alpha:.3f}")

        # b = log_F[index] - alpha*log_n[index]
        y = alpha * log_n + b
        plt.plot(log_n, y, '--', color=p[0].get_color())
        
        plt.plot(log_n[index], log_F[index], color=p[0].get_color(), marker='o')

def save_directed_dma_plots(x_open, y_open, x_closed, y_closed, title, folder_name, filename):
        fig = plt.figure()
        fig.suptitle(title)
        fig.set_size_inches((8, 11)) 
        plt.subplot(2,1,1)
        (angle, alpha) = DMA.exponent_for_angle(x_open, y_open)
        plt.plot(angle, alpha); plt.grid(); plt.title('Open eyes')
        plt.subplot(2,1,2)
        (angle, alpha) = DMA.exponent_for_angle(x_closed, y_closed)
        plt.plot(angle, alpha); plt.grid(); plt.title('Closed eyes')
        plots_path = f"C:\\Users\\bohdank\\Documents\\MyProjects\\stabilometry_analysis\\dma_first_results\\dma_plots\\{folder_name}"
        plt.savefig(f"{plots_path}\\{filename}-directed-dma.png")
        plt.close()


# data = pd.read_csv("C:\\Users\\bohdank\\Documents\\MyProjects\\stabilometry_analysis\\dma_first_results\\samp0.csv")
# x = data.iloc[:, 0]
# y = data.iloc[:, 1]
# (angle, alpha) = DMA.exponent_for_angle_debug(x, y)
# plt.plot(angle, alpha); plt.grid(); plt.title('samp0.csv')
# plt.show()


folder_names = {"rowing": "Rowing athletes", "diving": "Diving athletes", "healthy" : "Non sportsmen" }

for folder_name, title_text in folder_names.items():
    mat_file_pathes = get_file_pathes_in_dir(f"C:/Users/BohdanK/Dropbox/StabiloData/{folder_name}", extension="mat")
    records_dict = extract_records_from_mat_files(mat_file_pathes)

    log_F_open_x_list = []; log_F_open_y_list = []; log_F_open_2d_list = []
    log_F_closed_x_list = []; log_F_closed_y_list = []; log_F_closed_2d_list = []
    log_n_for_mean_1d = []; log_n_for_mean_2d = []
    for filename, records_array in records_dict.items(): 
        fig = plt.figure()
        fig.suptitle(f"{folder_name}: {filename}")
        fig.set_size_inches((24, 10)) 

        plt.subplot(1,2,1)
        open_eyes_record = records_array[0]
        (log_n, log_F) = DMA.dma_d1(open_eyes_record.cop.x); log_n_for_mean_1d = log_n
        if log_F.size in (24,25): log_F_open_x_list.append(log_F)
        plot_dma(log_n, log_F, 'frontal plane')
        (log_n, log_F) = DMA.dma_d1(open_eyes_record.cop.y)
        if log_F.size in (24,25): log_F_open_y_list.append(log_F)
        plot_dma(log_n, log_F, 'sagittal plane')
        (log_n, log_F) = DMA.dma_d2(open_eyes_record.cop.x, open_eyes_record.cop.y); log_n_for_mean_2d = log_n
        if log_F.size == 27: log_F_open_2d_list.append(log_F)
        plot_dma(log_n, log_F, '2D')
        plt.legend(); plt.grid(); plt.title("Open eyes"); plt.xlabel('log(n)'); plt.ylabel('log(F)')
        
        plt.subplot(1,2,2)
        closed_eyes_record = records_array[1]
        # (angle, alpha) = DMA.exponent_for_angle_debug(closed_eyes_record.cop.x, closed_eyes_record.cop.y)
        (log_n, log_F) = DMA.dma_d1(closed_eyes_record.cop.x) 
        if log_F.size in (24,25): log_F_closed_x_list.append(log_F)
        plot_dma(log_n, log_F, 'frontal plane') 
        (log_n, log_F) = DMA.dma_d1(closed_eyes_record.cop.y)
        if log_F.size in (24,25): log_F_closed_y_list.append(log_F)
        plot_dma(log_n, log_F, 'sagittal plane')
        (log_n, log_F) = DMA.dma_d2(closed_eyes_record.cop.x, closed_eyes_record.cop.y)
        if log_F.size == 27: log_F_closed_2d_list.append(log_F)
        plot_dma(log_n, log_F, '2D')
        plt.legend(); plt.grid(); plt.title("Closed eyes"); plt.xlabel('log(n)'); plt.ylabel('log(F)')

        plots_path = f"C:\\Users\\bohdank\\Documents\\MyProjects\\stabilometry_analysis\\dma_first_results\\dma_plots\\{folder_name}"
        plt.savefig(f"{plots_path}\\{filename}.png")
        plt.close()

        # save_directed_dma_plots(x_open=open_eyes_record.cop.x, y_open=open_eyes_record.cop.y, \
        #         x_closed=closed_eyes_record.cop.x, y_closed=closed_eyes_record.cop.y, \
        #         title=f"Directed DMA, {filename}", folder_name=folder_name, filename=filename)

    fig = plt.figure()
    fig.suptitle(title_text)
    fig.set_size_inches((24, 10)) 
    
    plt.subplot(1,2,1)
    log_F_open_x_mean = np.mean(log_F_open_x_list, axis=0)
    plot_dma(log_n_for_mean_1d, log_F_open_x_mean, f'frontal plane')
    log_F_open_y_mean = np.mean(log_F_open_y_list, axis=0)
    plot_dma(log_n_for_mean_1d, log_F_open_y_mean, f'sagittal plane')
    log_F_open_2d_mean = np.mean(log_F_open_2d_list, axis=0)
    plot_dma(log_n_for_mean_2d, log_F_open_2d_mean, f'2D')
    plt.legend(); plt.grid(); plt.title("Open eyes"); plt.xlabel('log(n)'); plt.ylabel('log(F)')
    
    plt.subplot(1,2,2)
    log_F_closed_x_mean = np.mean(log_F_closed_x_list, axis=0)
    plot_dma(log_n_for_mean_1d, log_F_closed_x_mean, f'frontal plane')
    log_F_closed_y_mean = np.mean(log_F_closed_y_list, axis=0)
    plot_dma(log_n_for_mean_1d, log_F_closed_y_mean, f'sagittal plane')
    log_F_closed_2d_mean = np.mean(log_F_closed_2d_list, axis=0)
    plot_dma(log_n_for_mean_2d, log_F_closed_2d_mean, f'2D')
    plt.legend(); plt.grid(); plt.title("Closed eyes"); plt.xlabel('log(n)'); plt.ylabel('log(F)')
    
    plots_path = "C:\\Users\\bohdank\\Documents\\MyProjects\\stabilometry_analysis\\dma_first_results\\dma_plots"
    plt.savefig(f"{plots_path}\\{folder_name}.png")
    plt.close()

        