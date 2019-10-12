import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

from .dfa_wrapper import get_log_log_plot_bending_point, calculate_alpha_exp_from_bending_point, get_log_log_plot_bending_point_with_debug

def get_current_folder():
    from pathlib import Path
    return str(Path(__file__).parent)
   

def calc_scaling_exponent_in_range(log_n, log_F, start_index=None, end_index=None):
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(log_n)

    x = np.expand_dims(np.asarray(log_n[start_index:end_index]), axis=1)
    y = np.asarray(log_F[start_index:end_index])

    reg = LinearRegression().fit(x, y)
    reg.score(x, y)

    return ( reg.coef_[0], reg.intercept_ )

def calc_scaling_exponent_by_bending_point(log_n, log_F):
    (_, index) = get_log_log_plot_bending_point(log_n, log_F)
    (alpha, b) = calculate_alpha_exp_from_bending_point(log_n, log_F, index)
    return (alpha, b, index)

def __exract_vectors_from_dma_data(data: str):
    rows = data.split('\n')
    array = [__row_to_float_array(row) for row in rows if ',' in row]
    log_n = np.array([row[2] for row in array])#0
    log_F = np.array([row[1] for row in array])
    return (log_n, log_F)

def __exract_vectors_from_direct_dma_data(data: str):
    rows = data.split('\n')
    array = [__row_to_float_array(row) for row in rows if ',' in row]
    angle_in_rads = np.array([row[1] for row in array])
    log_n = np.array([row[4] for row in array])#2
    log_F = np.array([row[3] for row in array])
    return (angle_in_rads, log_n, log_F)

def __row_to_float_array(row):
    return [float(el) for el in row.split(',')]

def __execute_program_on_file(cmd, data_path):
    cmd = "{} {}".format(cmd, data_path)
    out = subprocess.getoutput(cmd)
    (data, metadata) = out.split('Input')
    return __exract_vectors_from_dma_data(data)


def dma_d1(vector, order=0, make_intergration=True):
    current_path = f"{get_current_folder()}"

    data_file_path = f"{current_path}\\temp_data.csv"
    df = pd.DataFrame(vector)
    df.transpose()
    df.to_csv(data_file_path, index=False, header=False)

    # input_Str = "\n".join([str(el) for el in vector])
    # cmd = "{}\\cDMA0_updated.exe -d {}".format(current_path, input_Str)

    integration_mark = " -i " if make_intergration is False else " " 
    cmd = f"{current_path}\\cDMA{order}.exe {integration_mark} -c 1 {data_file_path}"
    out = subprocess.getoutput(cmd)
    (data, metadata) = out.split('Input')
    return __exract_vectors_from_dma_data(data)

def dma_d2(x_vector, y_vector, order=0, make_intergration=True):
    current_path = f"{get_current_folder()}"
    data_file_path = f"{current_path}\\temp_data.csv"
    np_array = np.column_stack((x_vector, y_vector))
    df = pd.DataFrame(np_array)
    df.transpose()
    df.to_csv(data_file_path, index=False, header=False)

    # zip_str_list = ["{},{}".format(el[0], el[1]) for el in zip(x_vector, y_vector)]
    # data_str = "\n".join([str(el) for el in zip_str_list])
    # cmd = "2d_DMA0_updated.exe -d {}".format(data_str)
    integration_mark = " -i " if make_intergration is False else " " 
    cmd = f"{current_path}\\2d_DMA{order}.exe {integration_mark} {data_file_path}"
    out = subprocess.getoutput(cmd)
    (data, metadata) = out.split('Input')
    return __exract_vectors_from_dma_data(data)


def __dma_directed_raw(x_vector, y_vector):
    current_path = f"{get_current_folder()}"
    data_file_path = f"{current_path}\\temp_data.csv"
    np_array = np.column_stack((x_vector, y_vector))
    df = pd.DataFrame(np_array)
    df.transpose()
    df.to_csv(data_file_path, index=False, header=False)
    cmd = f"{current_path}\\direcDMA0.exe -c 1 2 -b 64 {data_file_path}"
    out = subprocess.getoutput(cmd)
    (data, metadata) = out.split('Input')
    return __exract_vectors_from_direct_dma_data(data)

class DirectedDmaItem:
    def __init__(self, alpha, log_n, log_F):
        self.alpha = alpha
        self.log_n = log_n
        self.log_F = log_F

def dma_directed_for_angles(x_vector, y_vector):
    (angle_in_rads, log_n, log_F) = __dma_directed_raw(x_vector, y_vector)
    items = []
    for angle in np.unique(angle_in_rads):
        angle_log_n = log_n[angle_in_rads == angle]
        angle_log_F = log_F[angle_in_rads == angle]
        items.append(DirectedDmaItem(angle, angle_log_n, angle_log_F))
    return items

def dma_directed(x_vector, y_vector):
    (angle_in_rads, log_n, log_F) = __dma_directed_raw(x_vector, y_vector)
    unique_angle_vector = np.unique(angle_in_rads)

    alpha_vector = []
    for angle in unique_angle_vector:
        index_mask = angle_in_rads == angle
        angle_log_n = log_n[index_mask]
        angle_log_F = log_F[index_mask]
        (alpha, _, _) = calc_scaling_exponent_by_bending_point(angle_log_n, angle_log_F)
        alpha_vector.append(alpha)

    return (unique_angle_vector, alpha_vector)

def dma_directed_in_range(x_vector, y_vector, start_index, end_index):
    (angle_in_rads, log_n, log_F) = __dma_directed_raw(x_vector, y_vector)
    unique_angle_vector = np.unique(angle_in_rads)

    alpha_vector = []
    for angle in unique_angle_vector:
        index_mask = angle_in_rads == angle
        angle_log_n = log_n[index_mask]
        angle_log_F = log_F[index_mask]
        (alpha, _) = calc_scaling_exponent_in_range(angle_log_n, angle_log_F, start_index, end_index)
        alpha_vector.append(alpha)

    return (unique_angle_vector, alpha_vector)


def dma_directed_debug(x_vector, y_vector):
    (angle_in_rads, log_n, log_F) = __dma_directed_raw(x_vector, y_vector)
    unique_angle_vector = np.unique(angle_in_rads)

    alpha_vector = []
    for id, angle in enumerate(unique_angle_vector):
        index_mask = angle_in_rads == angle
        angle_log_n = log_n[index_mask]
        angle_log_F = log_F[index_mask]
        (_, index) = get_log_log_plot_bending_point_with_debug(angle_log_n, angle_log_F, debug=(True, True, "C:\\Users\\bohdank\\Documents\\MyProjects\\stabilometry_analysis\\dma_first_results\\debug\\", id+1))
        alpha = calculate_alpha_exp_from_bending_point(angle_log_n, angle_log_F, index)
        
        alpha_vector.append(alpha)

    return (unique_angle_vector, alpha_vector)


if __name__ == "main":
    dma_d2([1, 2, 3, 4, 5, 6, 7], [11, 22, 33, 44, 55, 66, 77])
    dma_d1([1, 2, 3, 4, 5, 6, 7])