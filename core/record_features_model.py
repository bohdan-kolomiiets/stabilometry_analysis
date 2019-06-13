from core.record_model import Record
from core.feature_extractor_model import FeatureExtractor

from typing import List
from collections import OrderedDict
import numpy as np

class RecordFeatures:

    @staticmethod
    def extract_features_from_patien_records(patient_records: List[Record]):
        return map(lambda rec: RecordFeatures.init_from_feature_extractor(FeatureExtractor(rec)), 
                patient_records)

    @staticmethod
    def extract_features_from_mat_files(mat_file_pathes, selected_test_names=None):
        features_set = []
        for file_path in mat_file_pathes:
            patient_records = Record.extract_records_from_mat_file(file_path, selected_test_names)
            patient_features = RecordFeatures.extract_features_from_patien_records(patient_records)
            features_set.append(patient_features)
        return features_set

    @classmethod
    def init_from_feature_extractor(cls, extractor: FeatureExtractor):

        feactures = RecordFeatures()
        record = extractor.record

        feactures.patient_name = record.patient_name
        feactures.record_name = record.record_name

        ellipse_data = extractor.prediction_ellipse(0.95)
        feactures.ellipse_area = ellipse_data.area
        feactures.ellipse_big_semi_axis_len = ellipse_data.saxes[0]
        feactures.ellipse_small_semi_axis_len = ellipse_data.saxes[1]
        feactures.ellipse_big_axis_angle = ellipse_data.angles[0]
        feactures.ellipse_small_axis_angle = ellipse_data.angles[1]

        feactures.mean_x = np.mean(record.cop.x)
        feactures.median_x = np.median(record.cop.x)
        feactures.std_x = np.std(record.cop.x)
        feactures.mean_y = np.mean(record.cop.y)
        feactures.median_y = np.median(record.cop.y)
        feactures.std_y = np.std(record.cop.y)
        feactures.turns_index = extractor.modified_turns_index()

        feactures.mean_vx, feactures.mean_vy = extractor.mean_axes_velocity()

        feactures.mean_path_v = extractor.mean_path_velocity()
        feactures.median_path_v = extractor.meadian_path_velocity()

        feactures.mean_angle = extractor.mean_angle()
        feactures.mean_weighted_angle = extractor.mean_weighted_angle()
        feactures.angle_quartile25 = extractor.angles_quartile(0.25)
        feactures.angle_quartile50 = extractor.angles_quartile(0.50)
        feactures.angle_quartile75 = extractor.angles_quartile(0.75)
        feactures.angle_quartile90 = extractor.angles_quartile(0.90)
        feactures.angle_quartile95 = extractor.angles_quartile(0.95)

        feactures.psd_angle = extractor.psd(extractor.fft_angle_vect, extractor.f_angle_vect)
        feactures.psd_welch_angle = extractor.psd_welch(extractor.angle_vect, extractor.record.f_hz)
        feactures.edge_freq_angle99, _, _ = extractor.edge_freq(extractor.fft_angle_vect, extractor.f_angle_vect, 0.99)
        feactures.spect_centroid_angle_freq = extractor.spectral_centroid_freq(extractor.fft_angle_vect, extractor.f_angle_vect)

        feactures.psd_x = extractor.psd(extractor.fft_x_vect, extractor.f_x_vect)
        feactures.psd_welch_x = extractor.psd_welch(extractor.record.cop.x, extractor.record.f_hz)
        feactures.edge_freq_x99, _, _ = extractor.edge_freq(extractor.fft_x_vect, extractor.f_x_vect, 0.99)
        feactures.spect_centroid_x_freq = extractor.spectral_centroid_freq(extractor.fft_x_vect, extractor.f_x_vect)

        feactures.psd_y = extractor.psd(extractor.fft_y_vect, extractor.f_y_vect)
        feactures.psd_welch_y = extractor.psd_welch(extractor.record.cop.y, extractor.record.f_hz)
        feactures.edge_freq_y99, _, _ = extractor.edge_freq(extractor.fft_y_vect, extractor.f_y_vect, 0.99)
        feactures.spect_centroid_y_freq = extractor.spectral_centroid_freq(extractor.fft_y_vect, extractor.f_y_vect)

        feactures.psd_path = extractor.psd(extractor.fft_path_vect, extractor.f_path_vect)
        feactures.psd_welch_path = extractor.psd_welch(extractor.path_vect, extractor.record.f_hz)
        feactures.edge_freq_path99, _, _ = extractor.edge_freq(extractor.fft_path_vect, extractor.f_path_vect, 0.99)
        feactures.spect_centroid_path_freq = extractor.spectral_centroid_freq(extractor.fft_path_vect, extractor.f_path_vect)

        return feactures


    @classmethod
    def init_from_pandas_row(cls, row):

        features = RecordFeatures()

        features.patient_name = row["patient"]
        features.record_name = row["record"]

        features.ellipse_area = row["ellipse_area"]
        features.ellipse_big_semi_axis_len = row["ellipse_big_semi_axis_len"]
        features.ellipse_small_semi_axis_len = row["ellipse_small_semi_axis_len"]
        features.ellipse_big_axis_angle = row["ellipse_big_axis_angle"]
        features.ellipse_small_axis_angle = row["ellipse_small_axis_angle"]

        features.mean_x = row["mean_x"]
        features.median_x = row["median_x"]
        features.std_x = row["std_x"]
        features.mean_y = row["mean_y"]
        features.median_y = row["median_y"]
        features.std_y = row["std_y"]
        features.turns_index = row["turns_index"]

        features.mean_vx = row["mean_vx"]
        features.mean_vy = row["mean_vy"]
        features.mean_path_v = row["mean_path_v"]
        features.median_path_v = row["median_path_v"]

        features.mean_angle = row["mean_angle"]
        features.mean_weighted_angle = row["mean_weighted_angle"]
        features.angle_quartile25 = row["angle_quartile25"]
        features.angle_quartile50 = row["angle_quartile50"]
        features.angle_quartile75 = row["angle_quartile75"]
        features.angle_quartile90 = row["angle_quartile90"]
        features.angle_quartile95 = row["angle_quartile95"]

        features.psd_angle = row["psd_angle"]
        features.psd_welch_angle = row["psd_welch_angle"]
        features.edge_freq_angle99 = row["edge_freq_angle99"]
        features.spect_centroid_angle_freq = row["spect_centroid_angle_freq"]

        features.psd_x = row["psd_x"]
        features.psd_welch_x = row["psd_welch_x"]
        features.edge_freq_x99 = row["edge_freq_x99"]
        features.spect_centroid_x_freq = row["spect_centroid_x_freq"]

        features.psd_y = row["psd_y"]
        features.psd_welch_y = row["psd_welch_y"]
        features.edge_freq_y99 = row["edge_freq_y99"]
        features.spect_centroid_y_freq = row["spect_centroid_y_freq"]

        features.psd_path = row["psd_path"]
        features.psd_welch_path = row["psd_welch_path"]
        features.edge_freq_path99 = row["edge_freq_path99"]
        features.spect_centroid_path_freq = row["spect_centroid_path_freq"]

        return features

    def to_export_dict(self):
        return OrderedDict([
            ('patient', self.patient_name),
            ('record', self.record_name),

            ('ellipse_area', self.ellipse_area),
            ('ellipse_big_semi_axis_len', self.ellipse_big_semi_axis_len),
            ('ellipse_small_semi_axis_len', self.ellipse_small_semi_axis_len),
            ('ellipse_big_axis_angle', self.ellipse_big_axis_angle),
            ('ellipse_small_axis_angle', self.ellipse_small_axis_angle),

            ('mean_x', self.mean_x),
            ('median_x', self.median_x),
            ('std_x', self.std_x),
            ('mean_y', self.mean_y),
            ('median_y', self.median_y),
            ('std_y', self.std_y),
            ('turns_index', self.turns_index),

            ('mean_vx', self.mean_vx),
            ('mean_vy', self.mean_vy),
            ('mean_path_v', self.mean_path_v),
            ('median_path_v', self.median_path_v),

            ('mean_angle', self.mean_angle),
            ('mean_weighted_angle', self.mean_weighted_angle),
            ('angle_quartile25', self.angle_quartile25),
            ('angle_quartile50', self.angle_quartile50),
            ('angle_quartile75', self.angle_quartile75),
            ('angle_quartile90', self.angle_quartile90),
            ('angle_quartile95', self.angle_quartile95),

            ('psd_angle', self.psd_angle),
            ('psd_welch_angle', self.psd_welch_angle),
            ('edge_freq_angle99', self.edge_freq_angle99),
            ('spect_centroid_angle_freq', self.spect_centroid_angle_freq),

            ('psd_x', self.psd_x),
            ('psd_welch_x', self.psd_welch_x),
            ('edge_freq_x99', self.edge_freq_x99),
            ('spect_centroid_x_freq', self.spect_centroid_x_freq),

            ('psd_y', self.psd_y),
            ('psd_welch_y', self.psd_welch_y),
            ('edge_freq_y99', self.edge_freq_y99),
            ('spect_centroid_y_freq', self.spect_centroid_y_freq),

            ('psd_path', self.psd_path),
            ('psd_welch_path', self.psd_welch_path),
            ('edge_freq_path99', self.edge_freq_path99),
            ('spect_centroid_path_freq', self.spect_centroid_path_freq)
        ])
