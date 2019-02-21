import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

def __exract_vectors_from_data(data: str):
    rows = data.split('\n')
    array = [__row_to_float_array(row) for row in rows if ',' in row]
    log_n = np.array([row[0] for row in array])
    log_F = np.array([row[1] for row in array])
    return (log_n, log_F)

def __row_to_float_array(row):
    return [float(el) for el in row.split(',')]

def __execute_program_on_file(cmd, data_path):
    cmd = "{} {}".format(cmd, data_path)
    out = subprocess.getoutput(cmd)
    (data, metadata) = out.split('Input')
    return __exract_vectors_from_data(data)


def dma_d1(vector):
    current_path = "{}\\dma_first_results".format(os.getcwd())
    data_file_path = "{}\\temp_data.csv".format(current_path)
    df = pd.DataFrame(vector)
    df.transpose()
    df.to_csv(data_file_path, index=False, header=False)

    # input_Str = "\n".join([str(el) for el in vector])
    # cmd = "{}\\cDMA0_updated.exe -d {}".format(current_path, input_Str)
    cmd = "{}\\cDMA0.exe -c 1 {}".format(current_path, data_file_path)
    out = subprocess.getoutput(cmd)
    (data, metadata) = out.split('Input')
    return __exract_vectors_from_data(data)

def dma_d2(x_vector, y_vector):
    current_path = "{}\\dma_first_results".format(os.getcwd())
    data_file_path = "{}\\temp_data.csv".format(current_path)
    np_array = np.column_stack((x_vector, y_vector))
    df = pd.DataFrame(np_array)
    df.transpose()
    df.to_csv(data_file_path, index=False, header=False)

    # zip_str_list = ["{},{}".format(el[0], el[1]) for el in zip(x_vector, y_vector)]
    # data_str = "\n".join([str(el) for el in zip_str_list])
    # cmd = "2d_DMA0_updated.exe -d {}".format(data_str)
    cmd = "{}\\2d_DMA0.exe {}".format(current_path, data_file_path)
    out = subprocess.getoutput(cmd)
    (data, metadata) = out.split('Input')
    return __exract_vectors_from_data(data)

if __name__ == "main":
    dma_d2([1, 2, 3, 4, 5, 6, 7], [11, 22, 33, 44, 55, 66, 77])
    dma_d1([1, 2, 3, 4, 5, 6, 7])