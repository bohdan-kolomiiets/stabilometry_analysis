import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

def test_shapiro(plot_number, samples_count):
    # data = np.random.normal(0, 1, samples_count)
    data = np.random.rand(samples_count)
    plt.subplot(2, 3, plot_number)
    plt.hist(data)
    _, p_val = sts.shapiro(data)
    plt.title('{} samples; p-val: {}'.format(samples_count, p_val))


test_shapiro(1, 10)
test_shapiro(2, 50)
test_shapiro(3, 100)
test_shapiro(4, 500)
test_shapiro(5, 1000)
test_shapiro(6, 5000)

plt.show()
