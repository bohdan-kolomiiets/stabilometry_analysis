import core.io_helpers as io 
import dma.dma_wrapper as dma
import wii.wii_board_data_reader as wii

import core.usachev_algorithms as usachev

import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 17

record_paths = [
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\non-sportsmen\alexey\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\non-sportsmen\alexey\base_close.csv",

        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\non-sportsmen\galina\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\non-sportsmen\galina\base_close.csv",

        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\non-sportsmen\maksim\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\non-sportsmen\maksim\base_close.csv",

        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\non-sportsmen\sasha\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\non-sportsmen\sasha\base_close.csv",

        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\non-sportsmen\yana\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\non-sportsmen\yana\base_close.csv",


        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\1\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\1\base_close.csv",

        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\2\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\2\base_close.csv",

        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\3\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\3\base_close.csv",
        
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\4\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\4\base_close.csv",
        
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\5\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\5\base_close.csv",
        
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\6\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\6\base_close.csv",
        
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\7\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\7\base_close.csv",

        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\8\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\8\base_close.csv",

        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\9\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\9\base_close.csv",

        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\10\base_open.csv", 
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\gandball\10\base_close.csv"
    ]

save_path = r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\usachev-algorithms'
io.create_folder_if_not_exist(save_path)

for path in record_paths:
    data = wii.read_wii_board_data(path)
    cop_x, cop_y = data.get_lowpass_filtered_cop(cut_f=15)
    Fs = data.resampled_f_hz

    record_name = io.extract_file_name_from_path(path)
    folder_name = io.extract_folder_name_from_path(path)
    parent_folder_name = io.extract_parent_folder_name_from_path(path)

    fig = plt.figure()
    fig.set_size_inches((25, 10))
    plt.suptitle(record_name)
    
    plt.subplot(1,2, 1)
    plt.title('CoP')
    plt.grid()
    # plt.xlim([-15, 15])
    # plt.ylim([-15, 15])

    plt.plot(cop_x, cop_y)
    
    plt.subplot(1,2, 2)
    (kfr, cumulative_density) = usachev.calc_kfr(cop_x, cop_y, Fs, zones_count=20)
    usachev.plot_kfr(record_name, cumulative_density, kfr)
    
    plt.savefig(fr'{save_path}\{parent_folder_name}_{folder_name}_{record_name}')
    plt.close()
    pass