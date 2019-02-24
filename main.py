import os
from scipy.io import loadmat
import json
import pandas as pd

from core.record_model import Record
from core.feature_extractor_model import FeatureExtractor
from core.record_features_model import RecordFeatures

import dma.dma_wrapper as DMA

def get_file_pathes_in_dir(path: str, extension: str):
    file_names = []
    for file_name in os.listdir(path):
        if file_name.endswith(".{}".format(extension)):
            file_names.append("{}/{}".format(path, file_name))
    return file_names
