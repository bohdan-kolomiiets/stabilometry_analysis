import numpy as np


class WiiBoardData:
    def __init__(self):
        self.time_ms = []
        self.top_left_f_kg = []
        self.top_right_f_kg = []
        self.bottom_left_f_kg = []
        self.bottom_right_f_kg = []
        self.cop_x = []
        self.cop_y = []
        self.total_f = [],
        self.resampled_f_hz = 0
        self.resampled_time_ms = []
        self.resampled_cop_x = []
        self.resampled_cop_y = []

    def get_lowpass_filtered_cop(self, cut_f):
        import core.math_helper as math_helper 
        x_filtered = math_helper.butter_lowpass_filter(self.resampled_cop_x, cut_f, self.resampled_f_hz)
        y_filtered = math_helper.butter_lowpass_filter(self.resampled_cop_y, cut_f, self.resampled_f_hz)
        return (x_filtered, y_filtered)


def __norm_and_resample_time_vect(f_hz, original_time_vect):
    time_vect_norm = (original_time_vect - original_time_vect[0])/1000
    T = time_vect_norm[-1]
    F = f_hz
    dt = 1/F
    time_vect_resampled = np.arange(0, T + dt, dt)
    return time_vect_resampled


def __resample_data(data, time_vect):
    import scipy.signal as sc
    resampled_data, _ = sc.resample(data, len(time_vect), time_vect)
    return resampled_data


def read_wii_board_data(file_path):
    import pandas as pd
    cvs_data = pd.read_csv(file_path, ' ')

    wii_data = WiiBoardData()
    wii_data.time_ms = cvs_data.iloc[:, 0]
    wii_data.top_left_f_kg = cvs_data.iloc[:, 1]
    wii_data.top_right_f_kg = cvs_data.iloc[:, 2]
    wii_data.bottom_left_f_kg = cvs_data.iloc[:, 3]
    wii_data.bottom_right_f_kg = cvs_data.iloc[:, 4]
    wii_data.cop_x = cvs_data.iloc[:, 5]
    wii_data.cop_y = cvs_data.iloc[:, 6] * -1
    wii_data.total_f = cvs_data.iloc[:, 7]

    f_hz = 100
    time_vect_resampled = __norm_and_resample_time_vect(f_hz, original_time_vect=np.array(wii_data.time_ms))

    wii_data.resampled_f_hz = f_hz
    wii_data.resampled_time_ms = time_vect_resampled
    wii_data.resampled_cop_x = __resample_data(np.array(wii_data.cop_x), time_vect_resampled)
    wii_data.resampled_cop_y = __resample_data(np.array(wii_data.cop_y), time_vect_resampled)

    return wii_data


if __name__ == "__main__":
    file_path = r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\trial1\anton-forward-right.csv'
    data = read_wii_board_data(file_path)

    # FFT
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(data.cop_x, data.cop_y, color='r', linestyle='--')
    plt.plot(data.resampled_cop_x, data.resampled_cop_y, color='b', linestyle='--')
    plt.legend(['original (mean F = 90 Hz)', 'resampled (F = 100 Hz)'])
    plt.grid()

    # FFT
    plt.figure()
    plt.title('Sampling frequencies histogram')
    t_diff = np.diff(data.time_ms)
    t_diff = t_diff[t_diff > 0]
    
    plt.hist(1000/t_diff, 50)
    plt.xlabel('F, Hz')
    plt.xticks(range(0, 1000, 50))
    plt.grid()
    plt.show()
