import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 0.5

import core.io_helpers as io
import dma.dma_wrapper as dma
import wii.wii_board_data_reader as wii

# np.random.normal()

def construct_figure_from(x, y, file_name):

    fig, axes = plt.subplots(2, 2)
    fig.suptitle(file_name)
    fig.set_size_inches((10, 10))

    # cop_axes
    axes[0,0].plot(x, y)
    axes[0,0].set(title='CoP', xlim=[-15, 15], ylim=[-15, 15], xlabel='X', ylabel='Y')
    axes[0,0].grid()

    # dma axes
    (log_n, log_F) = dma.dma_d1(x, order=2, make_intergration=True)
    (alpha, _) = dma.calc_scaling_exponent_in_range(log_n, log_F, dma_start_index, dma_end_index)
    axes[0,1].plot(log_n, log_F, color='g', label=f'DMA(X), alpha={alpha:.3f}')

    (log_n, log_F) = dma.dma_d1(y, order=2, make_intergration=True)
    (alpha, _) = dma.calc_scaling_exponent_in_range(log_n, log_F, dma_start_index, dma_end_index)
    axes[0,1].plot(log_n, log_F, color='r', label=f'DMA(Y), alpha={alpha:.3f}')

    (log_n, log_F) = dma.dma_d2(x, y, order=2, make_intergration=True)
    (alpha, _) = dma.calc_scaling_exponent_in_range(log_n, log_F, dma_start_index, dma_end_index)
    axes[0,1].plot(log_n, log_F, color='b', label=f'DMA(X-Y), alpha={alpha:.3f}')

    axes[0,1].vlines(log_n[dma_start_index], -3, 3, color='gray', linewidth=1, linestyle='--')
    axes[0,1].vlines(log_n[dma_end_index], -3, 3, color='gray', linewidth=1, linestyle='--')

    axes[0,1].set(title='Log-log curves', xlim=[0.5, 3], ylim=[-2.5, 2.5], xlabel='log(n)', ylabel='log(F)')
    axes[0,1].legend(loc='upper left')
    axes[0,1].grid()

    # Directed DMA axes (bending point)
    (angle, alpha) = dma.dma_directed(x, y)
    axes[1,0].plot(angle, alpha, color='r')

    axes[1,0].hlines(0.5, 0, np.pi, color='gray', linewidth=1.5)
    axes[1,0].hlines(1, 0, np.pi, color='gray', linewidth=1.5)
    axes[1,0].hlines(1.5, 0, np.pi, color='gray', linewidth=1.5)

    axes[1,0].set(title='Directed DMA curve (bending point)', xlim=[0, np.pi], ylim=[0, 2], xlabel='angle, rads', ylabel='alpha')
    axes[1,0].legend(loc='upper right')
    axes[1,0].grid()

    #Directed DMA axes (fixed range)
    (angle, alpha) = dma.dma_directed_in_range(x, y, dma_start_index, dma_end_index)
    axes[1,1].plot(angle, alpha, color='r')

    axes[1,1].hlines(0.5, 0, np.pi, color='gray', linewidth=1.5)
    axes[1,1].hlines(1, 0, np.pi, color='gray', linewidth=1.5)
    axes[1,1].hlines(1.5, 0, np.pi, color='gray', linewidth=1.5)

    axes[1,1].set(title='Directed DMA curve (fixed range)', xlim=[0, np.pi], ylim=[0, 2], xlabel='angle, rads', ylabel='alpha')
    axes[1,1].legend(loc='upper right')
    axes[1,1].grid()

    return fig


# ----------------------------------------

record_paths = [
    r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\trial3\bottles_20kg_3.csv",
    r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\trial1\anton-base.csv",
    r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\trial1\anton-backward-right.csv"
]
save_path = r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\ken_comparison'

dma_start_index = 6 # 1.2
dma_end_index = 22 # 1.8


for path in record_paths:
    data = wii.read_wii_board_data(path)
    corr_coef = np.corrcoef(data.cop_x, data.cop_y)[0, 1]
    print(f'{path}:  {corr_coef}')


    # file_name = io.extract_name_from_path(path)

    # figure = construct_figure_from(data.cop_x, data.cop_y, file_name)
    # plt.savefig(fr"{save_path}\{file_name}.png")
    
    # data_frame = pd.DataFrame({'time_ms': data.time_ms, 'x_cm': data.cop_x, 'y_cm': data.cop_y})
    # data_frame.to_csv(fr"{save_path}\from-function_{file_name}.csv", index=False)


pass