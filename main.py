import os
from scipy.io import loadmat
import json
import pandas as pd

from record_model import Record
from feature_extractor_model import FeatureExtractor
from feature_storage_model import FeatureStorage


def get_file_names_in_dir(path: str, extension: str):
    file_names = []
    for file in os.listdir(path):
        if file.endswith(".{}".format(extension)):
            file_names.append(file)
    return file_names


def extract_features_from_mat_files(mat_file_names):
    features = []
    for file_name in mat_file_names:
        mat_data = loadmat(file_name).get('s')[0]
        for mat_data_element in mat_data:
            record = Record(mat_data_element, file_name)
            extractor = FeatureExtractor(record)
            feature_set = FeatureStorage(extractor)
            features.append(feature_set)
    return features


mat_file_names = get_file_names_in_dir('.', 'mat')
features = extract_features_from_mat_files(mat_file_names)

# json0 = json.dumps(features[0].__dict__, sort_keys=False)
# print(json0)

features_frame = pd.DataFrame.from_records([f.to_export_dict() for f in features])
features_frame.to_excel("features.xlsx")
# print(features_frame)