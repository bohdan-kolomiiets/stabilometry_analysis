import numpy as np
import matplotlib.pyplot as plt

"""
Usage example:
# mat_data = loadmat('Violeta_Sverchkova_01-Nov-2018.mat').get('s')[0]
# record = Record(mat_data[0], 'Violeta_Sverchkova_01-Nov-2018')
# extractor = FeatureExtractor(record)
# plt.figure(0)
# visualizer.plot_force_signals(plt, record.get_time_vector(), record.force_signals)
# plt.figure(1)
# visualizer.plot_cop_signal(plt, record.cop.x, record.cop.y)
# plt.figure(2)
# visualizer.plot_fft(extractor.fftx_vect, extractor.f_vect, 'X')
# plt.figure(3)
# visualizer.plot_fft(extractor.ffty_vect, extractor.f_vect, 'Y')
# plt.show()
"""

__platform_side_length = 26


def plot_force_signals(plt_ref, time_vect, force_signals):
    plt_ref.plot(time_vect, force_signals[0])
    plt_ref.plot(time_vect, force_signals[1])
    plt_ref.plot(time_vect, force_signals[2])
    plt_ref.plot(time_vect, force_signals[3])
    plt_ref.legend(['LB', 'LF', 'RF', 'RB'])
    plt_ref.title('F(t)')
    plt_ref.xlabel('t, s')
    plt_ref.ylabel('F, N')
    plt_ref.grid(True)


def plot_cop_signal(plt_ref, x, y):
    plt_ref.xlim([-__platform_side_length / 2, __platform_side_length / 2])
    plt_ref.ylim([-__platform_side_length / 2, __platform_side_length / 2])
    plt_ref.grid(True)
    plt_ref.xlabel('X, cm')
    plt_ref.ylabel('Y, cm')
    plt_ref.title('COP')
    plt_ref.plot(x, y)


def plot_fft(fft, f, title=''):
    plt.plot(f, fft)
    plt.xlabel('F, Hz')
    plt.ylabel('A(F), cm')
    plt.title('{} spectrum'.format(title))
    plt.grid()


def plot_cop_with_ellipse(x, y, ellipse_data, units=None, ax=None):
    """Plot results of the hyperellipsoid function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        # code based on https://github.com/minillinim/ellipsoid:
        # parametric equations

        area = ellipse_data.area
        saxes = ellipse_data.saxes
        center = ellipse_data.center
        rot_matrix = ellipse_data.rot_matrix
        p_val = ellipse_data.p_val

        P = np.array([x, y]).transpose()
        u = np.linspace(0, 2 * np.pi, 100)

        x = saxes[0] * np.cos(u)
        y = saxes[1] * np.sin(u)
        # rotate data
        for i in range(len(x)):
            [x[i], y[i]] = np.dot([x[i], y[i]], rot_matrix) + center

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.plot(P[:, 0], P[:, 1], '.-', color=[0, 0, 1, .5])  # plot raw data
        ax.plot(x, y, color=[0, 1, 0, 1], linewidth=2)  # plot ellipse
        # plot axes
        diagonal_axes = np.diag(saxes)
        for i in range(saxes.size):
            axis_vector = diagonal_axes[i]
            a = np.dot(axis_vector, rot_matrix).reshape(2, 1)  # rotate axes
            # points for the axes extremities
            a = np.dot(a, np.array([-1, 1], ndmin=2)) + center.reshape(2, 1)
            ax.plot(a[0], a[1], color=[1, 0, 0, .6], linewidth=2)
            ax.text(a[0, 1], a[1, 1], '%d' % (i + 1), fontsize=20, color='r')
        plt.axis('equal')
        plt.grid()
        title = r'Prediction ellipse (p=%4.2f): Area=' % p_val
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
