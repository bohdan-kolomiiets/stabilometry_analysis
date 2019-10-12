import core.io_helpers as io
import wii.wii_board_data_reader as wii
import core.math_helper as math_helper
import dma.dma_wrapper as dma

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 0.5


# CHECK FILTRATION ON LOG-LOG PLOTS IN DIFFERENT ANGLES
# folder_path = r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\trial2'
# file_path = f'{folder_path}\\base_close_30_1_legs_together.csv'
# data = wii.read_wii_board_data(file_path)
# cut_freq = 15
# (x_filtered, y_filtered) = data.get_lowpass_filtered_cop(cut_freq)
# dma_items_original = dma.dma_directed_for_angles(
#     data.resampled_cop_x, data.resampled_cop_y)
# dma_items_filtered = dma.dma_directed_for_angles(x_filtered, y_filtered)

# for dma_item_original, dma_item_filtered in zip(dma_items_original, dma_items_filtered):
#     file_name = io.extract_name_from_path(file_path)
#     plt.figure()

#     (alpha, b, index) = dma.calc_scaling_exponent(dma_item_original.log_n, dma_item_original.log_F)
#     plt.plot(dma_item_original.log_n[index], dma_item_original.log_F[index], color='r', marker='o')
#     plt.plot(dma_item_original.log_n, dma_item_original.log_F, color='r', label=f'Original. Alpha:{alpha:.2f}')
    
#     (alpha, b, index) = dma.calc_scaling_exponent(dma_item_filtered.log_n, dma_item_filtered.log_F)
#     plt.plot(dma_item_filtered.log_n[index], dma_item_filtered.log_F[index], color='b', marker='o')
#     plt.plot(dma_item_filtered.log_n, dma_item_filtered.log_F, color='b', label=f'Filtered. Alpha:{alpha:.2f}. Cut freq: {cut_freq}')
    
#     plt.title(f' Directed DMA. Record:{file_name}. Angle:{dma_item_original.alpha}')
#     plt.grid()
#     plt.xlabel('log(n)')
#     plt.ylabel('log(F)')
#     plt.legend(loc='upper left')

#     save_folder = f'{folder_path}\\{file_name}'
#     io.create_folder_if_not_exist(save_folder)
#     plt.savefig(f'{save_folder}\\{file_name}_{dma_item_original.alpha}.png')



# RESAMPLING
# plt.figure(1)
# plt.plot(data.copX, data.copY, color='r', linestyle='--')
# plt.plot(data.resampledCopX, data.resampledCopY, color='b', linestyle='--')
# plt.legend(['original (mean F = 90 Hz)', 'resampled (F = 100 Hz)'])
# plt.grid()


# FILTERING
# plt.figure(2)
# folder_path = r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\trial2'
# file_path = f'{folder_path}\\base_close_30_1_legs_together.csv'
# data = wii.read_wii_board_data(file_path)
# plt.plot(data.resampled_time_ms, data.resampled_cop_y)
# for cut_f in [45, 35, 25, 15, 5]:
#     cop_x_filtered, cop_y_filtered = data.get_lowpass_filtered_cop(cut_f)
#     plt.plot(data.resampled_time_ms, cop_y_filtered, linestyle='--')
# plt.legend(['No filterinng', '45Hz lowpass filter', '35Hz lowpass filter',
#             '25Hz lowpass filter', '15Hz lowpass filter', '5Hz lowpass filter'])
# plt.show()


# DIR_DMA WITH DIFFERENT FILTERING
# plt.figure(3)
# (angle, alpha) = dma.dma_directed(data.resampledCopX, data.resampledCopY)
# plt.plot(angle, alpha)

# for cut_f in [45, 35, 25, 15, 5]:
#     x_filtered, y_filtered = filter_lowpass_cop(data.resampledCopX, data.resampledCopY, cut_f=cut_f, fs=100)
#     (angle, alpha) = dma.dma_directed(x_filtered, y_filtered)
#     plt.plot(angle, alpha)

# plt.legend(['No filterinng', '45Hz lowpass filter', '35Hz lowpass filter',
#             '25Hz lowpass filter', '15Hz lowpass filter', '5Hz lowpass filter'])
# plt.grid()

# CALCULATE 4 PLOTS
cut_f = 15  # Hz
wii_board_lim = 15  # cm
spectrum_x_lim = 30  # Hz
spectrum_y_lim = 10  # cm

data_folders = [
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\acrobats\abramskaya_valya',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\acrobats\duroshuk_sofia',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\acrobats\kucina',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\acrobats\kuzmenko',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\acrobats\mihail_sernachenko',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\acrobats\miroshnizhenko',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\acrobats\plashova',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\acrobats\protas_artem',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\acrobats\schukina',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\acrobats\sofin',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\acrobats\veklig_nastya',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\trial1',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\trial2',
    r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\trial3',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\trial3\1',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\trial3\2',
    # r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\trial3\3',
]
for data_folder in data_folders:
    figures_folder = data_folder + r'\figures'

    file_pathes = io.get_file_pathes_in_dir(data_folder, 'csv')
    for file_path in file_pathes:
        if 'WiiWhitenoise3m' not in file_path:
            continue
        file_name = io.extract_name_from_path(file_path)

        data = wii.read_wii_board_data(file_path)
        (x_filtered, y_filtered) = data.get_lowpass_filtered_cop(cut_f) #(data.resampled_cop_x, data.resampled_cop_y)
        fig = plt.figure()
        fig.set_size_inches((20, 10))
        plt.suptitle(file_name)

        plt.subplot(2, 3, 1)
        plt.plot(data.resampled_time_ms, x_filtered, color='r', label='X(t)')
        plt.plot(data.resampled_time_ms, y_filtered, color='b', label='Y(t)')
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('t, sec')
        plt.ylabel('distrance, cm')
        plt.ylim([-wii_board_lim, wii_board_lim])

        plt.subplot(2, 3, 2)
        plt.plot(x_filtered, y_filtered, color='r', label='CoP')
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('X, cm')
        plt.ylabel('Y, cm')
        plt.xlim([-wii_board_lim, wii_board_lim])
        plt.ylim([-wii_board_lim, wii_board_lim])

        plt.subplot(2, 3, 3)
        # (f, spectrum_x) = math_helper.calc_fft(data.resampled_cop_x - np.mean(data.resampled_cop_x), fs=100)
        # (f, spectrum_y) = math_helper.calc_fft(data.resampled_cop_y - np.mean(data.resampled_cop_y), fs=100)
        (f, spectrum_x) = math_helper.calc_fft(x_filtered - np.mean(x_filtered), fs=100)
        (f, spectrum_y) = math_helper.calc_fft(y_filtered - np.mean(y_filtered), fs=100)
        plt.plot(f, spectrum_x, color='r', label='Ax(f)')
        plt.plot(f, spectrum_y, color='b', label='Ay(f)')
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('f, Hz')
        plt.ylabel('A, cm')
        plt.xlim([-0.05, spectrum_x_lim])
        # plt.ylim([0, spectrum_y_lim])

        plt.subplot(2, 3, 4)
        order = 2
        dma_start_index = 6 # 1.2
        dma_end_index = 22 # 1.8
        (log_n, log_F) = dma.dma_d1(x_filtered, order, make_intergration=True)
        (alpha, _) = dma.calc_scaling_exponent_in_range(log_n, log_F, dma_start_index, dma_end_index)
        plt.plot(log_n, log_F, color='r', label=f'DMA(X), alpha={alpha:.3f}')

        (log_n, log_F) = dma.dma_d1(y_filtered, order, make_intergration=True)
        (alpha, _) = dma.calc_scaling_exponent_in_range(log_n, log_F, dma_start_index, dma_end_index)
        plt.plot(log_n, log_F, color='b', label=f'DMA(Y), alpha={alpha:.3f}')
        
        (log_n, log_F) = dma.dma_d2(x_filtered, y_filtered, order, make_intergration=True)
        (alpha, _) = dma.calc_scaling_exponent_in_range(log_n, log_F, dma_start_index, dma_end_index)
        plt.plot(log_n, log_F, color='g', label=f'DMA(2D), alpha={alpha:.3f}')
        
        plt.vlines(log_n[dma_start_index], -3, 3, color='gray', linewidth=1, linestyle='--')
        plt.vlines(log_n[dma_end_index], -3, 3, color='gray', linewidth=1, linestyle='--')
        
        plt.yticks(np.arange(-3, 3, 0.5))
        plt.ylim([-3, 3])
        plt.xlim([0, 3.5])
        plt.legend(loc='lower right')
        plt.grid()
        plt.xlabel('log(n)')
        plt.ylabel('log(F)')

        plt.subplot(2, 3, 5)
        (angle, alpha) = dma.dma_directed(x_filtered, y_filtered)
        plt.plot(angle, alpha, color='r', label='Directed DMA')
        plt.hlines(0.5, 0, np.pi, color='gray', linewidth=1.5)
        plt.hlines(1, 0, np.pi, color='gray', linewidth=1.5)
        plt.hlines(1.5, 0, np.pi, color='gray', linewidth=1.5)
        # plt.yticks(np.arange(0, 1.6, 0.1))
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('angle, rads')
        plt.ylabel('alpha')
        plt.ylim([0, 2])
        plt.xlim([0, np.pi])

        io.create_folder_if_not_exist(figures_folder)
        #
        plt.savefig(f'{figures_folder}\\{file_name}_order0_corrected_with_spikes.png')
        # plt.show()


# plt.show()
