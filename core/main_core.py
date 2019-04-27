import os
from scipy.io import loadmat
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

from record_model import Record
from feature_extractor_model import FeatureExtractor
from record_features_model import RecordFeatures
import record_visualization as visualizer

def get_file_pathes_in_dir(path: str, extension: str):
    file_names = []
    for file_name in os.listdir(path):
        if file_name.endswith(".{}".format(extension)):
            file_names.append("{}/{}".format(path, file_name))
    return file_names
    

if __name__ == "__main__":
    # study individual patient
    patient_records = Record.extract_records_from_mat_file(
        file_path = 'C:/Users/BohdanK/Dropbox/StabiloData/rowing/altuhov_26-Nov-2018.mat')
    patient_features = RecordFeatures.extract_features_from_patien_records(patient_records)

    chosen_record = patient_records[3] # for example purposes, first record was selected  

    plt.rcParams.update({'font.size': 20})
    plt.figure(0); visualizer.plot_force_signals(plt, chosen_record.get_time_vector(), chosen_record.force_signals)
    plt.figure(1); visualizer.plot_cop_signal(plt, chosen_record.cop.x, chosen_record.cop.y)

    chosen_extractor = FeatureExtractor(chosen_record)
    plt.figure(2); visualizer.plot_fft(chosen_extractor.fft_x_vect, chosen_extractor.f_x_vect, 'X')
    plt.figure(3); visualizer.plot_fft(chosen_extractor.fft_y_vect, chosen_extractor.f_y_vect, 'Y')

    plt.figure(4)
    plt.plot(chosen_record.get_time_vector(), chosen_record.cop.x, label="X")
    plt.plot(chosen_record.get_time_vector(), chosen_record.cop.y, label="Y")
    plt.title("CoP Oscillations on X and Y axes"); plt.legend()
    plt.xlabel("time, s"); plt.ylabel("distance from center, cm"); plt.grid()

    visualizer.plot_cop_with_ellipse(chosen_record.cop.x, chosen_record.cop.y, chosen_extractor.prediction_ellipse(), 'cm')

    plt.show()

    # study patient groups
    group_folder_names = ['rowing', 'diving', 'healthy']

    for group_folder_name in group_folder_names:
        group_mat_file_pathes = get_file_pathes_in_dir(f'C:/Users/BohdanK/Dropbox/StabiloData/{group_folder_name}', extension='mat')
        group_features = RecordFeatures.extract_features_from_mat_files(group_mat_file_pathes)

        #save features
        features_frame = pd.DataFrame.from_records([f.to_export_dict() for f in group_features])
        features_frame.to_excel(f'{group_folder_name}.xlsx')
