import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

import sys

inputJson = sys.argv[1]
inputObj =  json.loads(inputJson)
x = inputObj["x"]
y = inputObj["y"]
        
def __exract_vectors_from_dma_data(data: str):
    rows = data.split('\n')
    array = [__row_to_float_array(row) for row in rows if ',' in row]
    log_n = np.array([row[0] for row in array])
    log_F = np.array([row[1] for row in array])
    return (log_n, log_F)

def __exract_vectors_from_direct_dma_data(data: str):
    rows = data.split('\n')
    array = [__row_to_float_array(row) for row in rows if ',' in row]
    angle_in_rads = np.array([row[1] for row in array])
    log_n = np.array([row[2] for row in array])
    log_F = np.array([row[3] for row in array])
    return (angle_in_rads, log_n, log_F)

def __row_to_float_array(row):
    return [float(el) for el in row.split(',')]

def dma_d2(x_vector, y_vector):
    current_path = get_prod_or_dev_curr_path()
    data_file_path = f"{current_path}\\temp_data.csv"
    np_array = np.column_stack((x_vector, y_vector))
    df = pd.DataFrame(np_array)
    df.transpose()
    df.to_csv(data_file_path, index=False, header=False)

    cmd = f"{current_path}\\2d_DMA0.exe {data_file_path}"

    # 3. parse returned json in nodejs

    out = subprocess.getoutput(cmd)
    (data, metadata) = out.split('Input')
    return __exract_vectors_from_dma_data(data)

def get_prod_or_dev_curr_path():
    import os
    current_path = f"{os.getcwd()}\\resources\\app\\python_files"
    if os.path.isdir(current_path) == False:
        current_path = f"{os.getcwd()}\\python_files"
    return current_path

res = dma_d2(x, y)
return_obj = {
    "log_n": res[0].tolist(),
    "log_F": res[1].tolist()
}
print('RESULT: ', json.dumps(return_obj))