from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from math_helper import MathHelper

plt.close("all")


def new(name, data):
    return type(name, (object,), data)


class Record:

    def __init__(self, mat_record, experiment_name):
        self.id = mat_record[0]
        self.experiment_name = experiment_name
        self.date = mat_record[1]
        self.calibrate_perc = np.array(mat_record[2][0])
        self.calibrate_scale = np.array(mat_record[2][1])
        self.calibrate_offset = np.array(mat_record[2][2])
        self.record_name = mat_record[3]
        self.record_info = mat_record[4]
        self.f_hz = 10  # Hz - mat_record[5][0][0]
        self.duration_sec = mat_record[6][0][0] - 1 / self.f_hz
        self.raw_mass_signals = np.array(mat_record[8][:, 1:])
        self.signal_len = self.raw_mass_signals.shape[1]
        self.force_signals = self.__get_force_signals(self.raw_mass_signals)
        self.cop = self.__calc_cop()

    def __get_force_signals(self, raw_mass_signals):
        normalized_mass = self.__normalize_mass(raw_mass_signals)
        normalized_force = self.__convert_mass_to_force(normalized_mass)
        return normalized_force

    def __normalize_mass(self, raw_mass_signals):
        raw_mass_full = raw_mass_signals.sum(axis=0)
        normalized_mass = np.empty([4, self.signal_len])
        for i, channel in enumerate(raw_mass_signals):
            normalized_mass[i, :] = raw_mass_signals[i, :] + (25 - self.calibrate_perc[i]) / 100 * raw_mass_full
        return normalized_mass

    def __convert_mass_to_force(self, mass_signals):
        g = 9.8066  # m/s^2
        return mass_signals / 1000 * g

    def __calc_cop(self):
        f = self.force_signals
        f_full = f.sum(axis=0)
        length_x = 22.8 / 2  # cm
        length_y = 22.3 / 2  # cm
        x = length_x * (f[2, :] + f[3, :] - f[0, :] - f[1, :]) / f_full
        y = length_y * (f[1, :] + f[2, :] - f[0, :] - f[3, :]) / f_full
        return new('cop', {
            'x': x, 'y': y
        })

    def get_time_vector(self):
        dt = 1 / self.f_hz
        samples_number = self.force_signals.shape[1]
        return np.linspace(0, self.duration_sec, samples_number)


class RecordVisualizer:
    platform_side_length = 26

    def __init__(self):
        pass

    @staticmethod
    def plot_force_signals(plt_ref, record):
        plt_ref.plot(record.get_time_vector(), record.force_signals[0])
        plt_ref.plot(record.get_time_vector(), record.force_signals[1])
        plt_ref.plot(record.get_time_vector(), record.force_signals[2])
        plt_ref.plot(record.get_time_vector(), record.force_signals[3])
        plt_ref.legend(['LB', 'LF', 'RF', 'RB'])
        plt_ref.title('F(t)')
        plt_ref.xlabel('t, s')
        plt_ref.ylabel('F, N')
        plt_ref.grid(True)

    @classmethod
    def plot_cop_signal(cls, plt_ref, record):
        plt_ref.xlim([-cls.platform_side_length / 2, cls.platform_side_length / 2])
        plt_ref.ylim([-cls.platform_side_length / 2, cls.platform_side_length / 2])
        plt_ref.grid(True)
        plt_ref.xlabel('X, cm')
        plt_ref.ylabel('Y, cm')
        plt_ref.title('COP')
        plt_ref.plot(record.cop.x, record.cop.y)

    @staticmethod
    def plot_fft(fft, f, title=''):
        plt.plot(f, fft)
        plt.xlabel('F, Hz')
        plt.ylabel('A(F), cm')
        plt.title('{} spectrum'.format(title))
        plt.grid()

    @staticmethod
    def plot_cop_with_ellipse(x, y, area, saxes, center, R, pvalue, units = None, ax = None):
        """Plot results of the hyperellipsoid function, see its help."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
        else:
            # code based on https://github.com/minillinim/ellipsoid:
            # parametric equations
            P = np.array([x, y]).transpose()
            u = np.linspace(0, 2 * np.pi, 100)

            x = saxes[0] * np.cos(u)
            y = saxes[1] * np.sin(u)
            # rotate data
            for i in range(len(x)):
                [x[i], y[i]] = np.dot([x[i], y[i]], R) + center

            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # plot raw data
            ax.plot(P[:, 0], P[:, 1], '.-', color=[0, 0, 1, .5])
            # plot ellipse
            ax.plot(x, y, color=[0, 1, 0, 1], linewidth=2)
            # plot axes
            diagonal_axes = np.diag(saxes)
            for i in range(saxes.size):
                axis_vector = diagonal_axes[i]
                a = np.dot(axis_vector, R).reshape(2, 1)  # rotate axes
                # points for the axes extremities
                a = np.dot(a, np.array([-1, 1], ndmin=2)) + center.reshape(2, 1)
                ax.plot(a[0], a[1], color=[1, 0, 0, .6], linewidth=2)
                ax.text(a[0, 1], a[1, 1], '%d' % (i + 1), fontsize=20, color='r')
            plt.axis('equal')
            plt.grid()
            title = r'Prediction ellipse (p=%4.2f): Area=' % pvalue
            if units is not None:
                units2 = ' [%s]' % units
                units = units + r'$^2$'
                title = title + r'%.2f %s' % (area, units)
            else:
                units2 = ''
                title = title + r'%.2f' % area

            ax.set_xlabel('X' + units2, fontsize=18)
            ax.set_ylabel('Y' + units2, fontsize=18)
            plt.title(title)

            return ax


class FeatureExtractor:
    def __init__(self, record):
        self.record = record
        self.dx_vect = np.diff(record.cop.x)
        self.dy_vect = np.diff(record.cop.y)
        self.f_vect, self.fftx_vect = MathHelper.fft(record.cop.x, record.f_hz)
        self.f_vect, self.ffty_vect = MathHelper.fft(record.cop.y, record.f_hz)
        pass

    def mean_x(self):
        return np.mean(self.record.cop.x)

    def mean_y(self):
        return np.mean(self.record.cop.y)

    def std_x(self):
        return np.std(self.record.cop.x)

    def std_y(self):
        return np.std(self.record.cop.y)

    def path_vect(self):
        return np.sqrt(self.dx_vect * self.dx_vect + self.dy_vect * self.dy_vect)

    def path_velocity(self):
        return self.path_vect().sum() / self.record.duration_sec

    def axes_velocity(self):
        vx = np.sum(abs(self.dx_vect)) / record.duration_sec
        vy = np.sum(abs(self.dy_vect)) / record.duration_sec
        return vx, vy

    def angle_vect(self):
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
        return np.arcsin(self.dy_vect / self.path_vect()) + shift_angles_vect

    def mean_angle(self):
        return np.mean(self.angle_vect())

    def mean_weighted_angle(self):
        path_vect = self.path_vect()
        return np.sum(self.angle_vect() * path_vect) / np.sum(path_vect) / self.record.signal_len

    @staticmethod
    def psd(fft, f, f1, f2):
        """
        Power spectral density
        :param fft: amplitude vector [cm[
        :param f: frequency vector [Hz]
        :param f1: start frequency [Hz]
        :param f2: end frequency [Hz]
        :return: power spectral density [cm^2]
        """
        mask = (f >= f1) & (f <= f2)
        f_part = f[mask]
        fft_part = fft[mask]
        return sum(pow(fft_part, 2)) / f_part.size

    @staticmethod
    def spectral_power(fft, f, f1, f2):
        """
        Spectral power
        :param fft: amplitude vector [cm[
        :param f: frequency vector [Hz]
        :param f1: start frequency [Hz]
        :param f2: end frequency [Hz]
        :return: spectral power [cm^2 * Hz]
        """
        mask = (f >= f1) & (f <= f2)
        f_part = f[mask]
        fft_part = fft[mask]
        df = f[1] - f[0]
        return sum(pow(fft_part, 2)) * df  # (f_part[-1] - f_part[0])

    @staticmethod
    def edge_freq(fft, f, percent):
        """
        Spectral edge frequency
        :param fft: amplitude vector [cm[
        :param f: frequency vector [Hz]
        :param percent: value below which the specified percent of total power is located
        :return: edge frequency, amplitude at edge frequency,
        index of edge frequency at the input vector
        """
        total_squared_sum = sum(pow(fft, 2))
        target_squared_sum = total_squared_sum * percent

        temp_squared_sum = 0
        for index, amp in enumerate(fft):
            temp_squared_sum += amp ** 2
            if temp_squared_sum >= target_squared_sum:
                return f[index], fft[index], index

        return -1, -1, -1

    @staticmethod
    def spect_mean(fft, f):
        return

    @staticmethod
    def spect_centroid(fft, f):
        """
        Spectral centroid
        :return: mean frequency weighted by amplitude [Hz]
        """
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
            'area': area,
            'saxes': saxes,
            'angles': angles,
            'center': center,
            'rot_matrix': R
        })


mat_data = loadmat('Violeta_Sverchkova_01-Nov-2018.mat').get('s')
mat_records = mat_data[0]
record = Record(mat_records[0], 'Violeta_Sverchkova_01-Nov-2018')

extractor = FeatureExtractor(record)

plt.figure(0)
RecordVisualizer.plot_force_signals(plt, record)
plt.figure(1)
RecordVisualizer.plot_cop_signal(plt, record)
plt.figure(2)
RecordVisualizer.plot_fft(extractor.fftx_vect, extractor.f_vect, 'X')
plt.figure(3)
RecordVisualizer.plot_fft(extractor.ffty_vect, extractor.f_vect, 'Y')

ellipse_data = extractor.prediction_ellipse()
print('prediction ellipse area', ellipse_data.area, 'cm^2')
plt.figure(4)
#TODO: make plot_cop_with_ellipse(x, y, ellipse_data, 'cm', plt.axes())
RecordVisualizer.plot_cop_with_ellipse(
    record.cop.x, record.cop.y, ellipse_data.area, ellipse_data.saxes, ellipse_data.center, ellipse_data.rot_matrix, 0.95,
    'cm', plt.axes())

print('mean cop x', extractor.mean_x(), 'cm')
print('std cop x', extractor.std_x(), 'cm')
print('mean cop y', extractor.mean_y(), 'cm')
print('std cop y', extractor.std_y(), 'cm')

vx, vy = extractor.axes_velocity()
print('mean vx', vx, 'cm/sec')
print('mean vy', vy, 'cm/sec')
print('path v', extractor.path_velocity(), 'cm/sec')

print('mean angle', extractor.mean_angle(), 'degrees/sec')
print('mean weighted angle', extractor.mean_weighted_angle(), 'degrees/sec')

# f_sef, fft_sef, index_sef = FeatureExtractor.edge_freq(fftx, f, 0.5)
# print("50%: f:{}, a:{}, index:{}".format(f_sef, fft_sef, index_sef))
#
# f_sef, fft_sef, index_sef = FeatureExtractor.edge_freq(fftx, f, 0.8)
# print("80%: f:{}, a:{}, index:{}".format(f_sef, fft_sef, index_sef))
#
# f_sef, fft_sef, index_sef = FeatureExtractor.edge_freq(fftx, f, 0.9)
# print("90%: f:{}, a:{}, index:{}".format(f_sef, fft_sef, index_sef))
#
# f_sef, fft_sef, index_sef = FeatureExtractor.edge_freq(fftx, f, 0.95)
# print("95%: f:{}, a:{}, index:{}".format(f_sef, fft_sef, index_sef))
#
# f_sef, fft_sef, index_sef = FeatureExtractor.edge_freq(fftx, f, 0.99)
# print("99%: f:{}, a:{}, index:{}".format(f_sef, fft_sef, index_sef))
#
# print('sc', FeatureExtractor.spect_centroid(fftx, f), 'Hz')
#

ellipse_data = extractor.prediction_ellipse()
print('prediction ellipse area', ellipse_data.area, 'cm^2')

plt.show()
