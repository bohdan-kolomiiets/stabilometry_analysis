import os
from scipy.io import loadmat
import json
import pandas as pd

from record_model import Record
from feature_extractor_model import FeatureExtractor
from record_features_model import RecordFeatures


def get_file_pathes_in_dir(path: str, extension: str):
    file_names = []
    for file_name in os.listdir(path):
        if file_name.endswith(".{}".format(extension)):
            file_names.append("{}/{}".format(path, file_name))
    return file_names


def extract_features_from_mat_files(mat_file_pathes):
    record_features_set = []
    for file_path in mat_file_pathes:
        mat_data = loadmat(file_path).get('s')[0]
        file_name = extract_name_from_path(file_path)
        for mat_data_element in mat_data:
            record = Record(mat_data_element, file_name)
            if "ОС ЗГ" in record.record_name or "ОС ОГ" in record.record_name:
                extractor = FeatureExtractor(record)
                record_features = RecordFeatures.init_from_feature_extractor(extractor)
                record_features_set.append(record_features)
    return record_features_set


def extract_name_from_path(path):
    file_name_with_extension = path.split('/')[-1]
    return file_name_with_extension.split('.')[0]


mat_file_pathes = get_file_pathes_in_dir('C:/Users/BohdanK/Dropbox/StabiloData/rowing', 'mat')
#  rowing, water_jumps, healthy
features = extract_features_from_mat_files(mat_file_pathes)

# json0 = json.dumps(features[0].__dict__, sort_keys=False)
# print(json0)

features_frame = pd.DataFrame.from_records([f.to_export_dict() for f in features])
features_frame.to_excel("rowing.xlsx")
print(features_frame)

# import matplotlib.pyplot as plt
# import record_visualization as visualizer

# plt.rcParams.update({'font.size': 22})

# mat_data = loadmat('Violeta_Sverchkova_01-Nov-2018.mat').get('s')[0]
# record = Record(mat_data[9], 'Violeta_Sverchkova_01-Nov-2018')
# extractor = FeatureExtractor(record)

# plt.figure(0)
# visualizer.plot_force_signals(plt, record.get_time_vector(), record.force_signals)
# plt.figure(1)
# visualizer.plot_cop_signal(plt, record.cop.x, record.cop.y)
# plt.figure(2)
# visualizer.plot_fft(extractor.fft_x_vect, extractor.f_x_vect, 'X')
# plt.figure(3)
# visualizer.plot_fft(extractor.fft_y_vect, extractor.f_y_vect, 'Y')
# plt.figure(4)
# plt.plot(record.get_time_vector(), record.cop.x, label="X")
# plt.plot(record.get_time_vector(), record.cop.y, label="Y")
# plt.title("CoP Oscillations in X and Y planes")
# plt.xlabel("time, s")
# plt.ylabel("distance from center, cm")
# plt.grid(True)
#plt.figure(5)
# visualizer.plot_cop_with_ellipse(record.cop.x, record.cop.y, extractor.prediction_ellipse(), 'cm')
# plt.show()