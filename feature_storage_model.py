import numpy as np
from collections import OrderedDict

from feature_extractor_model import FeatureExtractor


class FeatureStorage:
    def __init__(self, extractor: FeatureExtractor):
        record = extractor.record

        self.patient_name = record.patient_name
        self.record_name = record.record_name

        ellipse_data = extractor.prediction_ellipse(0.95)
        self.ellipse_area = ellipse_data.area  # cm^2
        self.ellipse_big_semi_axis_len = ellipse_data.saxes[0]  # cm
        self.ellipse_small_semi_axis_len = ellipse_data.saxes[1]  # cm
        self.ellipse_big_axis_angle = ellipse_data.angles[0]  # deg
        self.ellipse_small_axis_angle = ellipse_data.angles[1]  # deg

        self.mean_x = np.mean(record.cop.x)
        self.median_x = np.median(record.cop.x)
        self.std_x = np.std(record.cop.x)
        self.mean_y = np.mean(record.cop.y)
        self.median_y = np.median(record.cop.y)
        self.std_y = np.std(record.cop.y)
        self.turns_index = extractor.modified_turns_index()

        self.mean_vx, self.mean_vy = extractor.mean_axes_velocity()

        self.mean_path_v = extractor.mean_path_velocity()
        self.median_path_v = extractor.meadian_path_velocity()

        self.mean_angle = extractor.mean_angle()
        self.mean_weighted_angle = extractor.mean_weighted_angle()
        self.angle_quartile25 = extractor.angles_quartile(0.25)
        self.angle_quartile50 = extractor.angles_quartile(0.50)
        self.angle_quartile75 = extractor.angles_quartile(0.75)
        self.angle_quartile90 = extractor.angles_quartile(0.90)
        self.angle_quartile95 = extractor.angles_quartile(0.95)

        self.psd_angle = extractor.psd(extractor.fft_angle_vect, extractor.f_vect)
        self.edge_freq_angle50, _, _ = extractor.edge_freq(extractor.fft_angle_vect, extractor.f_vect, 0.5)
        self.edge_freq_angle80, _, _ = extractor.edge_freq(extractor.fft_angle_vect, extractor.f_vect, 0.8)
        self.edge_freq_angle90, _, _ = extractor.edge_freq(extractor.fft_angle_vect, extractor.f_vect, 0.9)
        self.edge_freq_angle95, _, _ = extractor.edge_freq(extractor.fft_angle_vect, extractor.f_vect, 0.95)
        self.edge_freq_angle99, _, _ = extractor.edge_freq(extractor.fft_angle_vect, extractor.f_vect, 0.99)
        self.spect_centroid_angle_freq = extractor.spectral_centroid_freq(extractor.fft_angle_vect, extractor.f_vect)

        self.psd_x = extractor.psd(extractor.fftx_vect, extractor.f_vect)
        self.edge_freq_x50, _, _ = extractor.edge_freq(extractor.fftx_vect, extractor.f_vect, 0.5)
        self.edge_freq_x80, _, _ = extractor.edge_freq(extractor.fftx_vect, extractor.f_vect, 0.8)
        self.edge_freq_x90, _, _ = extractor.edge_freq(extractor.fftx_vect, extractor.f_vect, 0.9)
        self.edge_freq_x95, _, _ = extractor.edge_freq(extractor.fftx_vect, extractor.f_vect, 0.95)
        self.edge_freq_x99, _, _ = extractor.edge_freq(extractor.fftx_vect, extractor.f_vect, 0.99)
        self.spect_centroid_x_freq = extractor.spectral_centroid_freq(extractor.fftx_vect, extractor.f_vect)

        self.psd_y = extractor.psd(extractor.ffty_vect, extractor.f_vect)
        self.edge_freq_y50, _, _ = extractor.edge_freq(extractor.ffty_vect, extractor.f_vect, 0.5)
        self.edge_freq_y80, _, _ = extractor.edge_freq(extractor.ffty_vect, extractor.f_vect, 0.8)
        self.edge_freq_y90, _, _ = extractor.edge_freq(extractor.ffty_vect, extractor.f_vect, 0.9)
        self.edge_freq_y95, _, _ = extractor.edge_freq(extractor.ffty_vect, extractor.f_vect, 0.95)
        self.edge_freq_y99, _, _ = extractor.edge_freq(extractor.ffty_vect, extractor.f_vect, 0.99)
        self.spect_centroid_y_freq = extractor.spectral_centroid_freq(extractor.ffty_vect, extractor.f_vect)

        self.psd_path = extractor.psd(extractor.fft_path_vect, extractor.f_vect)
        self.edge_freq_path50, _, _ = extractor.edge_freq(extractor.fft_path_vect, extractor.f_vect, 0.5)
        self.edge_freq_path80, _, _ = extractor.edge_freq(extractor.fft_path_vect, extractor.f_vect, 0.8)
        self.edge_freq_path90, _, _ = extractor.edge_freq(extractor.fft_path_vect, extractor.f_vect, 0.9)
        self.edge_freq_path95, _, _ = extractor.edge_freq(extractor.fft_path_vect, extractor.f_vect, 0.95)
        self.edge_freq_path99, _, _ = extractor.edge_freq(extractor.fft_path_vect, extractor.f_vect, 0.99)
        self.spect_centroid_path_freq = extractor.spectral_centroid_freq(extractor.fft_path_vect, extractor.f_vect)

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
            ('edge_freq_angle50', self.edge_freq_angle50),
            ('edge_freq_angle80', self.edge_freq_angle80),
            ('edge_freq_angle90', self.edge_freq_angle90),
            ('edge_freq_angle95', self.edge_freq_angle95),
            ('edge_freq_angle99', self.edge_freq_angle99),
            ('spect_centroid_angle_freq', self.spect_centroid_angle_freq),

            ('psd_x', self.psd_x),
            ('edge_freq_x50', self.edge_freq_x50),
            ('edge_freq_x80', self.edge_freq_x80),
            ('edge_freq_x90', self.edge_freq_x90),
            ('edge_freq_x95', self.edge_freq_x95),
            ('edge_freq_x99', self.edge_freq_x99),
            ('spect_centroid_x_freq', self.spect_centroid_x_freq),

            ('psd_y', self.psd_y),
            ('edge_freq_y50', self.edge_freq_y50),
            ('edge_freq_y80', self.edge_freq_y80),
            ('edge_freq_y90', self.edge_freq_y90),
            ('edge_freq_y95', self.edge_freq_y95),
            ('edge_freq_y99', self.edge_freq_y99),
            ('spect_centroid_y_freq', self.spect_centroid_y_freq),

            ('psd_path', self.psd_path),
            ('edge_freq_path50', self.edge_freq_path50),
            ('edge_freq_path80', self.edge_freq_path80),
            ('edge_freq_path90', self.edge_freq_path90),
            ('edge_freq_path95', self.edge_freq_path95),
            ('edge_freq_path99', self.edge_freq_path99),
            ('spect_centroid_path_freq', self.spect_centroid_path_freq)
        ])
