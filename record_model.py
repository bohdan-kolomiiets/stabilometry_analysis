import numpy as np


class Record:

    def __init__(self, mat_record, patient_name):
        self.id = mat_record["Id"][0][0]
        self.patient_name = patient_name
        self.date = mat_record["DateTime"][0]
        self.calibrate_perc = np.array(mat_record["CalibrValues"][0])
        self.calibrate_scale = np.array(mat_record["CalibrValues"][1])
        self.calibrate_offset = np.array(mat_record["CalibrValues"][2])
        self.record_name = mat_record["Name"][0]
        self.record_info = mat_record["Features"]
        self.f_hz = 10  # Hz - mat_record[5][0][0]
        self.duration_sec = mat_record["T_seconds"][0][0] - 1 / self.f_hz
        self.raw_mass_signals = np.array(mat_record["Signals"][:, 1:])
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
        return type('cop', (object,), {
            'x': x, 'y': y
        })

    def get_time_vector(self):
        dt = 1 / self.f_hz
        samples_number = self.force_signals.shape[1]
        return np.linspace(0, self.duration_sec, samples_number)
