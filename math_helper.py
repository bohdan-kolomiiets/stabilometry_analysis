import numpy as np


class MathHelper:
    def __init__(self):
        pass

    @staticmethod
    def fft(x, fs):
        import numpy as np
        from scipy.fftpack import fft

        n_half = int(x.size / 2)
        f = np.linspace(0, fs / 2, n_half)  # * 2 * np.pi
        fourier = np.abs(fft(x)[:n_half]) / n_half
        return f, fourier


def test_calc_fft():
    import matplotlib.pyplot as plt

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
    f, fft = MathHelper.fft(x1, fs)
    plt.plot(f, fft)

    target_a_index = fft.argmax()
    target_a = fft[target_a_index]
    target_f = f[target_a_index]
    print('ampl:', target_a)
    print('freq:', target_f)

    plt.show()
