import os
from scipy.io import loadmat
import json
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import numpy as np

# import sys
# sys.path.append('C:\\Users\\bohdank\\Documents\\MyProjects\\science\\stabilometry_analysis\\core')
from core.record_model import Record
from core.feature_extractor_model import FeatureExtractor
from core.record_features_model import RecordFeatures
import core.io_helpers as io

import dma.dma_wrapper as DMA


def extract_records_from_mat_files(mat_file_pathes):
    records_dict = {}
    for file_path in mat_file_pathes:
        mat_data = loadmat(file_path).get('s')[0]
        file_name = io.extract_name_from_path(file_path)

        patient_records = []
        for mat_data_element in mat_data:
            record = Record(mat_data_element, file_name)
            if "ОС ЗГ" in record.record_name or "ОС ОГ" in record.record_name:
                patient_records.append(record)
        records_dict[file_name] = patient_records

    return records_dict


def plot_dma(log_n, log_F, title):
        (alpha, b, index) = DMA.calc_scaling_exponent_by_bending_point(log_n, log_F)

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
        (angle, alpha) = DMA.dma_directed(x_open, y_open)
        plt.plot(angle, alpha); plt.grid(); plt.title('Open eyes')
        plt.subplot(2,1,2)
        (angle, alpha) = DMA.dma_directed(x_closed, y_closed)
        plt.plot(angle, alpha); plt.grid(); plt.title('Closed eyes')
        plots_path = f"C:\\Users\\bohdank\\Documents\\MyProjects\\stabilometry_analysis\\dma\\dma_plots\\{folder_name}"
        plt.savefig(f"{plots_path}\\{filename}-directed-dma.png")
        plt.close()

def plot_cop_and_dma(record, path_part):    
        fig = plt.figure()
        fig.set_size_inches((30, 12))
        plt.rcParams.update({'font.size': 17})

        fig.suptitle(f'{record.patient_name}, {record.record_name}')
        
        plt.subplot(1,2,1)
        plt.plot(record.get_time_vector(), record.cop.x, color='r', label='X(t)')
        plt.plot(record.get_time_vector(), record.cop.y, color='b', label='Y(t)')
        plt.legend(loc='upper right')
        plt.ylim([-10, 10])
        plt.grid()
        
        plt.subplot(1,2,2)
        
        (log_n, log_F) = DMA.dma_d1(record.cop.x)
        (alpha, _, index) = DMA.calc_scaling_exponent_by_bending_point(log_n, log_F)
        p = plt.plot(log_n, log_F, color='r', label=f'DMA(X), alpha={alpha:.3f}')
        plt.plot(log_n[index], log_F[index], color=p[0].get_color(), marker='o')

        (log_n, log_F) = DMA.dma_d1(record.cop.y)
        (alpha, _, index) = DMA.calc_scaling_exponent_by_bending_point(log_n, log_F)
        p = plt.plot(log_n, log_F, color='b', label=f'DMA(Y), alpha={alpha:.3f}')
        plt.plot(log_n[index], log_F[index], color=p[0].get_color(), marker='o')
        
        # (log_n, log_F) = DMA.dma_d2(record.cop.x, record.cop.y)
        # (alpha, _, index) = DMA.calc_scaling_exponent(log_n, log_F)
        # p = plt.plot(log_n, log_F, color='g', label=f'DMA(2D), alpha={alpha:.3f}')
        # plt.plot(log_n[index], log_F[index], color=p[0].get_color(), marker='o')
        
        plt.legend(loc='upper right')
        plt.grid()

        folder_path = r'C:\Users\bohdank\Documents\MyProjects\science\stabilometry_analysis\dma\dma_plots\cop_dma'
        plt.savefig(f'{folder_path}\\{path_part}')
        # plt.show()



# data = pd.read_csv("C:\\Users\\bohdank\\Documents\\MyProjects\\stabilometry_analysis\\dma\\samp0.csv")
# x = data.iloc[:, 0]
# y = data.iloc[:, 1]
# (angle, alpha) = DMA.dma_directed_debug(x, y)
# plt.plot(angle, alpha); plt.grid(); plt.title('samp0.csv')
# plt.show()


folder_names = {"rowing": "Rowing athletes", "diving": "Diving athletes", "healthy" : "Non sportsmen" }
import os

plots_path =  f"{os.getcwd()}\dma_plots"


for folder_name, title_text in folder_names.items():
    mat_file_pathes = io.get_file_pathes_in_dir(f"C:/Users/BohdanK/Dropbox/StabiloData/{folder_name}", extension="mat")
    group_records_dict = extract_records_from_mat_files(mat_file_pathes)

    for filename, records_array in group_records_dict.items(): 
        open_eyes_record = records_array[0]
        closed_eyes_record = records_array[1]
        plot_cop_and_dma(open_eyes_record, f'{folder_name}\\{filename}_open.png')
        plot_cop_and_dma(closed_eyes_record, f'{folder_name}\\{filename}_closed.png')


for folder_name, title_text in folder_names.items():
    mat_file_pathes = io.get_file_pathes_in_dir(f"C:/Users/BohdanK/Dropbox/StabiloData/{folder_name}", extension="mat")
    group_records_dict = extract_records_from_mat_files(mat_file_pathes)

    log_F_open_x_list = []; log_F_open_y_list = []; log_F_open_2d_list = []
    log_F_closed_x_list = []; log_F_closed_y_list = []; log_F_closed_2d_list = []
    log_n_for_mean_1d = []; log_n_for_mean_2d = []

    dir_dma_angle_open_list = []; dir_dma_angle_close_list = []

    count = 0
    for filename, records_array in group_records_dict.items(): 
        # count = count + 1
        # if count == 2: break

        # fig = plt.figure()
        # fig.suptitle(f"{folder_name}: {filename}")
        # fig.set_size_inches((10, 24)) 

        open_eyes_record = records_array[0]
        
        (log_n, log_F) = DMA.dma_d1(open_eyes_record.cop.x); log_n_for_mean_1d = log_n
        if log_F.size in (24,25): log_F_open_x_list.append(log_F)

                
        (log_n, log_F) = DMA.dma_d1(open_eyes_record.cop.y)
        if log_F.size in (24,25): log_F_open_y_list.append(log_F)
        
        
        (log_n, log_F) = DMA.dma_d2(open_eyes_record.cop.x, open_eyes_record.cop.y); log_n_for_mean_2d = log_n
        if log_F.size == 27: log_F_open_2d_list.append(log_F)
        
        # plt.subplot(2,1,1)
        # plot_dma(log_n, log_F, 'frontal plane')
        # plot_dma(log_n, log_F, 'sagittal plane')
        # plot_dma(log_n, log_F, '2D')
        # plt.legend(); plt.grid(); plt.title("Open eyes"); plt.xlabel('log(n)'); plt.ylabel('log(F)')
        # plt.ylim((-2, 0.6))

        closed_eyes_record = records_array[1]
        (log_n, log_F) = DMA.dma_d1(closed_eyes_record.cop.x) 
        if log_F.size in (24,25): log_F_closed_x_list.append(log_F)
        (log_n, log_F) = DMA.dma_d1(closed_eyes_record.cop.y)
        if log_F.size in (24,25): log_F_closed_y_list.append(log_F)
        (log_n, log_F) = DMA.dma_d2(closed_eyes_record.cop.x, closed_eyes_record.cop.y)
        if log_F.size == 27: log_F_closed_2d_list.append(log_F)
        
        # plt.subplot(2,1,2)
        # plot_dma(log_n, log_F, 'frontal plane') 
        # plot_dma(log_n, log_F, 'sagittal plane')
        # plot_dma(log_n, log_F, '2D')
        # plt.legend(); plt.grid(); plt.title("Closed eyes"); plt.xlabel('log(n)'); plt.ylabel('log(F)')
        # plt.ylim((-2, 0.6))

        # plt.savefig(f"{plots_path}\\{folder_name}\\{filename}.png")
        # plt.close()

        # (angle, alpha) = DMA.dma_directed(open_eyes_record.cop.x, open_eyes_record.cop.y)
        # dir_dma_angle_open_list.append([angle, alpha])
        # (angle, alpha) = DMA.dma_directed(closed_eyes_record.cop.x, closed_eyes_record.cop.y)
        # dir_dma_angle_close_list.append([angle, alpha])
        
        # save_directed_dma_plots(x_open=open_eyes_record.cop.x, y_open=open_eyes_record.cop.y, \
        #         x_closed=closed_eyes_record.cop.x, y_closed=closed_eyes_record.cop.y, \
        #         title=f"Directed DMA, {filename}", folder_name=folder_name, filename=filename)



#     average_dir_dma_angle_open = np.mean([el[1] for el in dir_dma_angle_open_list], 0)
#     average_dir_dma_angle_close = np.mean([el[1] for el in dir_dma_angle_close_list], 0)
    
#     fig = plt.figure()
#     fig.suptitle(folder_name)
#     fig.set_size_inches((24, 10)) 

#     plt.subplot(1,2,1)
#     plt.plot(angle, average_dir_dma_angle_open)
#     plt.xlabel('alpha'); plt.ylabel('angle')
#     plt.title('EO'); plt.grid()

#     plt.subplot(1,2,2)
#     plt.plot(angle, average_dir_dma_angle_close)
#     plt.xlabel('alpha'); plt.ylabel('angle')
#     plt.title('EC'); plt.grid()
    
#     plt.savefig(f"{plots_path}\\directed-dma-{folder_name}.png")
#     plt.close()


    fig = plt.figure()
    fig.suptitle(title_text)
    fig.set_size_inches((24, 10)) 
    
    log_F_open_x_mean = np.mean(log_F_open_x_list, axis=0)
    log_F_open_y_mean = np.mean(log_F_open_y_list, axis=0)
    log_F_open_2d_mean = np.mean(log_F_open_2d_list, axis=0)
    
    plt.subplot(1,2,1)
    plot_dma(log_n_for_mean_1d, log_F_open_x_mean, f'frontal plane')
    plot_dma(log_n_for_mean_1d, log_F_open_y_mean, f'sagittal plane')
    plot_dma(log_n_for_mean_2d, log_F_open_2d_mean, f'2D')
    plt.legend(); plt.grid(); plt.title("Open eyes"); plt.xlabel('log(n)'); plt.ylabel('log(F)')
    plt.ylim((-2, 0.6))

    log_F_closed_x_mean = np.mean(log_F_closed_x_list, axis=0)
    log_F_closed_y_mean = np.mean(log_F_closed_y_list, axis=0)
    log_F_closed_2d_mean = np.mean(log_F_closed_2d_list, axis=0)
    
    plt.subplot(1,2,2)
    plot_dma(log_n_for_mean_1d, log_F_closed_x_mean, f'frontal plane')
    plot_dma(log_n_for_mean_1d, log_F_closed_y_mean, f'sagittal plane')
    plot_dma(log_n_for_mean_2d, log_F_closed_2d_mean, f'2D')
    plt.legend(); plt.grid(); plt.title("Closed eyes"); plt.xlabel('log(n)'); plt.ylabel('log(F)')
    plt.ylim((-2, 0.6))
    
    plt.savefig(f"{plots_path}\\{folder_name}.png")
    plt.close()

        