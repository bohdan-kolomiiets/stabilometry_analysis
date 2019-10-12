import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn 

def brownian_motion(N):
    dt = math.sqrt(1/N)
    z = np.random.randn(N)
    z[0] = 0
    return np.cumsum(z * dt)

signal = brownian_motion(200)


def integrate(signal, until_index):
    part = signal[:until_index]
    centered_part = part - np.mean(part)
    return np.sum(centered_part)

class ExampleSegmentsSplitter:
    def __init__(self, segment_size, overlap_size):
        self.segment_size = segment_size
        self.overlap_size = overlap_size

    def split(self, signal):
        return []

class ExampleSegmentsFitter:
    def __init__(self):
        pass

    def fit(self, signal):
        return []


splitter = ExampleSegmentsSplitter(segment_size=)

integrated = [integrate(signal, until_index=current_index) for current_index in range(1, len(signal))]




plt.subplot(1,2,1)
plt.plot(signal)

plt.subplot(1,2,2)
plt.plot(integrated)

plt.show()