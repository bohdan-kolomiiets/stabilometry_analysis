import numpy as np
import statistics as stat
import scipy.signal as sci_sig
from typing import Tuple

import core.math_helper as math_helper 
from core.record_model import Record


class FeatureExtractor:
    def __init__(self, record: Record):
        self.record = record
        self.dx_vect = np.diff(record.cop.x)
        self.dy_vect = np.diff(record.cop.y)
        self.path_vect = self.__get_path_vect()
        self.angle_vect = self.__get_angle_vect()
        self.f_x_vect, self.fft_x_vect = math_helper.calc_fft(record.cop.x, record.f_hz)
        self.f_y_vect, self.fft_y_vect = math_helper.calc_fft(record.cop.y, record.f_hz)
        self.f_path_vect, self.fft_path_vect = math_helper.calc_fft(self.path_vect, record.f_hz)
        self.f_angle_vect, self.fft_angle_vect = math_helper.calc_fft(self.angle_vect, record.f_hz)

    def __get_path_vect(self):
        return np.sqrt(self.dx_vect * self.dx_vect + self.dy_vect * self.dy_vect)

    def mean_path_velocity(self):
        return self.path_vect.sum() / self.record.duration_sec

    def meadian_path_velocity(self):
        return stat.median(self.path_vect) * self.record.f_hz

    def mean_axes_velocity(self) -> Tuple[float, float]:
        vx = np.sum(abs(self.dx_vect)) / self.record.duration_sec
        vy = np.sum(abs(self.dy_vect)) / self.record.duration_sec
        return vx, vy

    def modified_turns_index(self):
        std_x = np.std(self.record.cop.x)
        std_y = np.std(self.record.cop.y)
        x_el = self.dx_vect / std_x
        y_el = self.dy_vect / std_y
        return np.sum(np.sqrt(x_el * x_el + y_el * y_el)) / self.record.signal_len

    def __get_angle_vect(self):
        def shift_angle(dx_el, dy_el):
            if dx_el > 0 and dy_el > 0:  # 1st quarter
                return 0
            elif dx_el < 0 and dy_el > 0:  # 2nd quarter
                return 90
            elif dx_el < 0 and dy_el < 0:  # 3rd quarter
                return 180
            elif dx_el > 0 and dy_el < 0:  # 4th quarter
                return 270

        shift_angles_vect = [shift_angle(dx_el, dy_el) for dx_el, dy_el in zip(self.dx_vect, self.dy_vect)]
        return np.arcsin(self.dy_vect / self.path_vect) + shift_angles_vect

    def mean_angle(self):
        return np.mean(self.angle_vect)

    def mean_weighted_angle(self):
        return np.sum(self.angle_vect * self.path_vect) / np.sum(self.path_vect) / self.record.signal_len

    def angles_quartile(self, quartile):
        return FeatureExtractor.__quartile(self.angle_vect, quartile)

    @staticmethod
    def __quartile(signal, p):
        sorted_x = np.sort(signal)
        quartile_index = int(signal.size * p)
        return sorted_x[quartile_index]

    @staticmethod
    def __get_fft_part(fft, f, f1, f2):
        mask = (f >= f1) & (f <= f2)
        f_part = f[mask]
        fft_part = fft[mask]
        return f_part, fft_part

    @staticmethod
    def psd(fft, f, f1=None, f2=None):
        if f1 is None:
            f1 = f[0]
        if f2 is None:
            f2 = f[-1]
        f_part, fft_part = FeatureExtractor.__get_fft_part(fft, f, f1, f2)
        df = f[1] - f[0]
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
        return np.sum(pow(fft_part, 2)) / (f2-f1) #  np.mean(pow(fft_part, 2)) / df

    @staticmethod
    def psd_welch(x, fs):
        f, fft = sci_sig.welch(x, fs, scaling='density')
        return np.mean(fft)

    @staticmethod
    def spectral_power(fft, f, f1, f2):
        f_part, fft_part = FeatureExtractor.__get_fft_part(fft, f, f1, f2)
        return np.sum(pow(fft_part, 2))  # (f_part[-1] - f_part[0])  # df = f[1] - f[0]

    @staticmethod
    def edge_freq(fft, f, percent):
        total_squared_sum = np.sum(pow(fft, 2))
        target_squared_sum = total_squared_sum * percent

        temp_squared_sum = 0
        for index, amp in enumerate(fft):
            temp_squared_sum += amp ** 2
            if temp_squared_sum >= target_squared_sum:
                return f[index], fft[index], index

        return -1, -1, -1

    @staticmethod
    def spectral_centroid_freq(fft, f):
        return sum(fft * f) / sum(fft)

    def prediction_ellipse(self, p_val=.95):
        """
        Prediction hyperellipsoid for multivariate data.
        __author__ = 'Marcos Duarte, https://github.com/demotu/BMC'

        Parameters
        ----------
        P : 1-D or 2-D array_like
            For a 1-D array, P is the abscissa values of the [x,y] or [x,y,z] data.
            For a 2-D array, P is the joined values of the multivariate data.
            The shape of the 2-D array should be (n, p) where n is the number of
            observations (rows) and p the number of dimensions (columns).
        p_val : float, optional (default = .95)
            Desired prediction probability of the hyperellipsoid.

        Returns
        -------
        hypervolume : float
            Hypervolume (e.g., area of the ellipse or volume of the ellipsoid).
        axes : 1-D array
            Lengths of the semi-axes hyperellipsoid (largest first).
        angles : 1-D array
            Angles of the semi-axes hyperellipsoid (only for 2D or 3D data).
            For the ellipsoid (3D data), the angles are the Euler angles
            calculated in the XYZ sequence.
        center : 1-D array
            Centroid of the hyperellipsoid.
        rotation : 2-D array
            Rotation matrix for hyperellipsoid semi-axes (only for 2D or 3D data).
        """

        from scipy.stats import f as F
        from scipy.special import gamma

        data = np.array([self.record.cop.x, self.record.cop.y]).transpose()

        # covariance matrix
        cov = np.cov(data, rowvar=0)
        # singular value decomposition
        U, s, Vt = np.linalg.svd(cov)
        p, n = s.size, data.shape[0]
        # F percent point function
        fppf = F.ppf(p_val, p, n - p) * (n - 1) * p * (n + 1) / n / (n - p)
        # semi-axes (largest first)
        saxes = np.sqrt(s * fppf)
        area = np.pi ** (p / 2) / gamma(p / 2 + 1) * np.prod(saxes)
        # rotation matrix
        R = Vt
        angles = np.array([np.rad2deg(np.arctan2(R[1, 0], R[0, 0])), 90 - np.rad2deg(np.arctan2(R[1, 0], -R[0, 0]))])
        # centroid of the ellipse
        center = np.mean(data, axis=0)

        return type('', (object,), {
            'area': area,  # cm^2
            'saxes': saxes,  # cm
            'angles': angles,  # deg
            'center': center,  # cm
            'rot_matrix': R,
            'p_val': p_val
        })
