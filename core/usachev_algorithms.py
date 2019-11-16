import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_kfr(x_vect, y_vect, Fs, zones_count = 15):
    dt = 1/Fs

    #calculate velocities (length of centered vectors)
    vx_vect = np.diff(x_vect)/dt
    vy_vect = np.diff(y_vect)/dt    
    r_vect = [np.sqrt(p[0]*p[0] + p[1]*p[1]) for p in zip(vx_vect, vy_vect)]
    # points_vect = zip(vx_vect, vy_vect, l_vect)

    # calculate square of single circle
    R_max = np.max(r_vect)
    S_total = np.pi * R_max * R_max
    S_i = S_total / zones_count

    zone_number_vect = range(1, zones_count + 1)
    r_threshold_vect = [np.sqrt(zone_number * S_i / np.pi) for zone_number in zone_number_vect]

    #calculate vectors count per zone
    vectors_per_zone = []
    for zone_number in zone_number_vect:
        index = zone_number - 1
        curr_r = r_threshold_vect[index]
        prev_r = 0 if index - 1 < 0 else r_threshold_vect[index - 1]
        indexes_in_zone = [i for i, l in enumerate(r_vect) if l > prev_r and l < curr_r]
        vectors_per_zone.append(len(indexes_in_zone))
        pass

    cumulative_vectors_sum = np.cumsum(vectors_per_zone) / len(r_vect)
    
    #calculate КФР
    total_square = zones_count * 1
    filled_square = np.sum(cumulative_vectors_sum)
    kfr = filled_square / total_square
    
    return (kfr, cumulative_vectors_sum)


def plot_kfr(record_title, cumulative_density, kfr):
    y = np.insert(cumulative_density, 0, 0)
    x = np.arange(0, len(y))
    plt.title(f'Vectors density distribution, КФР={kfr:.4f}')
    plt.xlabel('Zone number')
    plt.ylabel('Vectors percentile')
    plt.grid()
    plt.xticks(x)
    plt.plot(x, y)
    pass


if __name__ == "__main__":

    from core import io_helpers as io 
    import dma.dma_wrapper as dma
    import wii.wii_board_data_reader as wii

    record_paths = [
        r"C:\Users\bohdank\Dropbox\StabiloData\wii_board\trial1\anton-base.csv",
    ]
    save_path = r'C:\Users\bohdank\Dropbox\StabiloData\wii_board\usachev-algorithms'


    for path in record_paths:
        data = wii.read_wii_board_data(path)
        record_name = io.extract_file_name_from_path(path)
        
        plt.figure()
        plt.suptitle(record_name)
        
        plt.subplot(1,2, 1)
        plt.title('CoP')
        plt.grid()
        # plt.xlim([-15, 15])
        # plt.ylim([-15, 15])
        plt.plot(data.resampled_cop_x, data.resampled_cop_y)
        
        plt.subplot(1,2, 2)
        (kfr, cumulative_density) = calc_kfr(data.resampled_cop_x, data.resampled_cop_y, data.resampled_f_hz, zones_count=15)
        plot_kfr(record_name, cumulative_density, kfr)
        
        plt.show()

        pass