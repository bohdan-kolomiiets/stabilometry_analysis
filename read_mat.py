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
        self.duration_sec = mat_record[6]
        self.raw_mass_signals = np.array(mat_record[8])
        self.force_signals = self.__get_force_signals(self.raw_mass_signals)
        self.cop = self.__calc_cop()

    def __get_force_signals(self, raw_mass_signals):
        normalized_mass = self.__normalize_mass(raw_mass_signals)
        normalized_force = self.__convert_mass_to_force(normalized_mass)
        return normalized_force

    def __normalize_mass(self, raw_mass_signals):
        if 0 in raw_mass_signals[:, 0]:
            raw_mass_signals[:, 0] = raw_mass_signals[:, 1]
        raw_mass_full = raw_mass_signals.sum(axis=0)

        normalized_mass = np.empty([4, raw_mass_full.size])
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
        duration = (samples_number - 1) * dt
        return np.linspace(0, duration, samples_number)


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


class FeatureExtractor:
    def __init__(self):
        pass

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
    def spect_centroid(fft, f):
        """
        Spectral centroid
        :return: mean frequency weighted by amplitude [Hz]
        """
        return sum(fft * f) / sum(fft)


mat_data = loadmat('Violeta_Sverchkova_01-Nov-2018.mat').get('s')
mat_records = mat_data[0]
record = Record(mat_records[0], 'Violeta_Sverchkova_01-Nov-2018')

# plt.figure(0)
# RecordVisualizer.plot_force_signals(plt, record)
# plt.figure(1)
# RecordVisualizer.plot_cop_signal(plt, record)

# print('mean cop x', np.mean(record.cop.x), 'cm')
# print('std cop x', np.std(record.cop.x), 'cm')
# print('mean cop y', np.mean(record.cop.y), 'cm')
# print('std cop y', np.std(record.cop.y), 'cm')
# print('mean vx', np.mean(np.diff(record.cop.x))*record.f_hz, 'cm/sec')
# print('mean vy', np.mean(np.diff(record.cop.y))*record.f_hz, 'cm/sec')
# print('mean abs vx', np.mean(abs(np.diff(record.cop.x)))*record.f_hz, 'cm/sec')
# print('mean abs vy', np.mean(abs(np.diff(record.cop.y)))*record.f_hz, 'cm/sec')

f, fftx = MathHelper.fft(record.cop.x, record.f_hz)
plt.plot(f, fftx)
plt.xlabel('F, Hz')
plt.ylabel('A(F), cm')
plt.title('Spectrum')
plt.grid()


sp100 = FeatureExtractor.spectral_power(fftx, f, 0, 5)
print('sp100:', sp100)

f_sef, fft_sef, index_sef = FeatureExtractor.edge_freq(fftx, f, 0.5)
print("50%: f:{}, a:{}, index:{}".format(f_sef, fft_sef, index_sef))

f_sef, fft_sef, index_sef = FeatureExtractor.edge_freq(fftx, f, 0.8)
print("80%: f:{}, a:{}, index:{}".format(f_sef, fft_sef, index_sef))

f_sef, fft_sef, index_sef = FeatureExtractor.edge_freq(fftx, f, 0.9)
print("90%: f:{}, a:{}, index:{}".format(f_sef, fft_sef, index_sef))
sp90 = FeatureExtractor.spectral_power(fftx, f, 0, 0.1006)
print('sp90', sp90, 'sp90/sp100', sp90 / sp100)

f_sef, fft_sef, index_sef = FeatureExtractor.edge_freq(fftx, f, 0.95)
print("95%: f:{}, a:{}, index:{}".format(f_sef, fft_sef, index_sef))
sp95 = FeatureExtractor.spectral_power(fftx, f, 0, 0.436)
print('sp95', sp95, 'sp95/sp100', sp95 / sp100)

f_sef, fft_sef, index_sef = FeatureExtractor.edge_freq(fftx, f, 0.99)
print("99%: f:{}, a:{}, index:{}".format(f_sef, fft_sef, index_sef))
sp99 = FeatureExtractor.spectral_power(fftx, f, 0, 1.073)
print('sp99', sp99, 'sp99/sp100', sp99 / sp100)

print('sc', FeatureExtractor.spect_centroid(fftx, f), 'Hz')

# plt.show()
P = np.array([record.cop.x, record.cop.y]).transpose()
MathHelper.hyperellipsoid(P, units='cm', show=True)