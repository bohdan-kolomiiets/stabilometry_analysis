import numpy as np
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt, freqz

def calc_fft(x, fs):
    n_half = int(x.size / 2)
    f = np.linspace(0, fs / 2, n_half)
    fourier = np.abs(fft(x)[:n_half]) / n_half
    return f, fourier


def butter_lowpass_filter(data, cut_f, fs, order=10):
    b, a = __butter_lowpass_coefs(cut_f, fs, order=order)
    y = filtfilt(b, a, data)
    return y
def __butter_lowpass_coefs(highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a


def __butter_bandpass_coefs(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure()
    fs = 100
    for order in [2, 3, 6, 15]:
        b, a = __butter_lowpass_coefs(highcut=20, fs=fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


    
    N = 300
    T = 30  # sec
    t = np.linspace(0, T, N)
    dt = T / (N - 1)
    fs = 1 / dt
    x1 = 0.5 * np.sin(2 * np.pi * t)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(t, x1)
    plt.subplot(212)
    f, fft = calc_fft(x1, fs)
    plt.plot(f, fft)

    target_a_index = fft.argmax()
    target_a = fft[target_a_index]
    target_f = f[target_a_index]
    print('ampl:', target_a)
    print('freq:', target_f)

    plt.show()