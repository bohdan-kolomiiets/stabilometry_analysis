from collections import OrderedDict

from record_features_model import RecordFeatures
import pandas as pd
import scipy.stats as sts
import numpy as np


# ------------------------------------


def test_normality(data_vector):
    alpha = 0.05

    # null hypothesis - data has normal distribution
    stat, p = sts.shapiro(data_vector)
    shapiro_wilk_res = p > alpha

    # n has to more than 20
    # stat, p = sts.normaltest(data_vector)
    # agostino_res = p > alpha

    # anderson_res_data = sts.anderson(data_vector)
    # anderson_res = np.all(anderson_res_data.critical_values > anderson_res_data.statistic)

    return shapiro_wilk_res  # and anderson_res and agostino_res


def get_features_sample(prop_key, prop_val_lambda, record_name):
    features_array = [prop_val_lambda(record_features) for record_features in record_features_set
                      if record_features.record_name == record_name]
    return {prop_key: features_array}


def construct_features_dict(record_name):
    dict = OrderedDict([])
    dict.update(get_features_sample('ellipse_area', lambda x: x.ellipse_area, record_name))
    dict.update(get_features_sample('ellipse_big_semi_axis_len', lambda x: x.ellipse_big_semi_axis_len, record_name))
    dict.update(
        get_features_sample('ellipse_small_semi_axis_len', lambda x: x.ellipse_small_semi_axis_len, record_name))
    dict.update(get_features_sample('ellipse_big_axis_angle', lambda x: x.ellipse_big_axis_angle, record_name))
    dict.update(get_features_sample('ellipse_small_axis_angle', lambda x: x.ellipse_small_axis_angle, record_name))

    dict.update(get_features_sample('mean_x', lambda x: x.mean_x, record_name))
    dict.update(get_features_sample('median_x', lambda x: x.median_x, record_name))
    dict.update(get_features_sample('std_x', lambda x: x.std_x, record_name))
    dict.update(get_features_sample('mean_y', lambda x: x.mean_y, record_name))
    dict.update(get_features_sample('median_y', lambda x: x.median_y, record_name))
    dict.update(get_features_sample('std_y', lambda x: x.std_y, record_name))
    dict.update(get_features_sample('turns_index', lambda x: x.turns_index, record_name))

    dict.update(get_features_sample('mean_vx', lambda x: x.mean_vx, record_name))
    dict.update(get_features_sample('mean_vy', lambda x: x.mean_vy, record_name))
    dict.update(get_features_sample('mean_path_v', lambda x: x.mean_path_v, record_name))
    dict.update(get_features_sample('median_path_v', lambda x: x.median_path_v, record_name))

    dict.update(get_features_sample('mean_angle', lambda x: x.mean_angle, record_name))
    dict.update(get_features_sample('mean_weighted_angle', lambda x: x.mean_weighted_angle, record_name))
    dict.update(get_features_sample('angle_quartile25', lambda x: x.angle_quartile25, record_name))
    dict.update(get_features_sample('angle_quartile50', lambda x: x.angle_quartile50, record_name))
    dict.update(get_features_sample('angle_quartile75', lambda x: x.angle_quartile75, record_name))
    dict.update(get_features_sample('angle_quartile90', lambda x: x.angle_quartile90, record_name))
    dict.update(get_features_sample('angle_quartile95', lambda x: x.angle_quartile95, record_name))

    dict.update(get_features_sample('psd_angle', lambda x: x.psd_angle, record_name))
    dict.update(get_features_sample('edge_freq_angle99', lambda x: x.edge_freq_angle99, record_name))
    dict.update(get_features_sample('spect_centroid_angle_freq', lambda x: x.spect_centroid_angle_freq, record_name))

    dict.update(get_features_sample('psd_x', lambda x: x.psd_x, record_name))
    dict.update(get_features_sample('edge_freq_x99', lambda x: x.edge_freq_x99, record_name))
    dict.update(get_features_sample('spect_centroid_x_freq', lambda x: x.spect_centroid_x_freq, record_name))

    dict.update(get_features_sample('psd_y', lambda x: x.psd_y, record_name))
    dict.update(get_features_sample('edge_freq_y99', lambda x: x.edge_freq_y99, record_name))
    dict.update(get_features_sample('spect_centroid_y_freq', lambda x: x.spect_centroid_y_freq, record_name))

    dict.update(get_features_sample('psd_path', lambda x: x.psd_path, record_name))
    dict.update(get_features_sample('edge_freq_path99', lambda x: x.edge_freq_path99, record_name))
    dict.update(get_features_sample('spect_centroid_path_freq', lambda x: x.spect_centroid_path_freq, record_name))

    return dict


# ------------------------------------


excel_frame = pd.read_excel("water_jumps.xlsx")

record_features_set = []
for index, row in excel_frame.iterrows():
    record_features_set.append(RecordFeatures.init_from_pandas_row(row))

open_eyes_features_dict = construct_features_dict('ОС ОГ')
close_eyes_features_dict = construct_features_dict('ОС ЗГ')

open_close_similarity_dict = OrderedDict([])
for (feature_key, open_eyes_features), (feature_key, close_eyes_features) \
        in zip(open_eyes_features_dict.items(), close_eyes_features_dict.items()):

    open_len = len(open_eyes_features)
    close_len = len(close_eyes_features)
    is_open_normal = test_normality(open_eyes_features)
    is_close_normal = test_normality(close_eyes_features)

    # if p is bigger than alpha - we cannot reject null hypothesis (samples are similar)
    if is_open_normal and is_close_normal:
        # null hypothesis - the averages of two groups are equal
        ttest_res = sts.ttest_rel(open_eyes_features, close_eyes_features)  # equal_var=True
        p_val = ttest_res.pvalue
        is_normal_distribution = True
    else:
        # null hypothesis - the medians of all groups are equal
        kruskal_res = sts.wilcoxon(open_eyes_features, close_eyes_features)  # kruskal
        p_val = kruskal_res.pvalue
        is_normal_distribution = False

    mean_ration = np.mean(close_eyes_features) / np.mean(open_eyes_features) * 100
    if np.median(open_eyes_features) > 0:
        median_ratio = np.median(close_eyes_features)/np.median(open_eyes_features) * 100
    else:
        median_ratio = 0
    # skewness = sts.skew(close_eyes_features)/sts.skew(open_eyes_features) * 100
    open_close_similarity_dict.update({feature_key: [ p_val, is_normal_distribution, mean_ration, median_ratio, \
                                                      sts.skew(open_eyes_features), sts.skew(close_eyes_features)]})

open_close_not_similar_dict = {key: val for key, val in open_close_similarity_dict.items() if val[0] < 0.05}
print('are similar:', sorted(open_close_not_similar_dict.items(), key=lambda kv: kv[1]))

# import matplotlib.pyplot as plt
#
# index = 0
# for key, val in open_close_similarity_dict.items():
#     open_values = open_eyes_features_dict[key]
#     close_values = close_eyes_features_dict[key]
#
#     plt.figure(index)
#     manager = plt.get_current_fig_manager()
#     manager.window.showMaximized()
#     plt.subplot(1, 2, 1)
#     plt.title("{}: open ({})".format(key, open_close_similarity_dict[key]))
#     plt.hist(open_values)
#     plt.subplot(1, 2, 2)
#     plt.title("{}: close ({})".format(key, open_close_similarity_dict[key]))
#     plt.hist(close_values)
#     index+=1
#
#     # plt.tight_layout()
#     plt.savefig('{}.png'.format(key))

# open_values = open_eyes_features_dict["mean_path_v"]
# close_values = close_eyes_features_dict["mean_path_v"]
# plt.subplot(4,2,1); plt.title("mean_path_v: open ({})".format(open_close_similarity_dict["mean_path_v"])); plt.hist(open_values)
# plt.subplot(4,2,2); plt.title("mean_path_v: close ({})".format(open_close_similarity_dict["mean_path_v"])); plt.hist(close_values)
#
# open_values = open_eyes_features_dict["median_path_v"]
# close_values = close_eyes_features_dict["median_path_v"]
# plt.subplot(4,2,3); plt.title("median_path_v: open ({})".format(open_close_similarity_dict["median_path_v"])); plt.hist(open_values)
# plt.subplot(4,2,4); plt.title("median_path_v: close ({})".format(open_close_similarity_dict["median_path_v"])); plt.hist(close_values)
#
# open_values = open_eyes_features_dict["mean_vy"]
# close_values = close_eyes_features_dict["mean_vy"]
# plt.subplot(4,2,5); plt.title("mean_vy: open ({})".format(open_close_similarity_dict["mean_vy"])); plt.hist(open_values)
# plt.subplot(4,2,6); plt.title("mean_vy: close ({})".format(open_close_similarity_dict["mean_vy"])); plt.hist(close_values)
#
# open_values = open_eyes_features_dict["psd_path"]
# close_values = close_eyes_features_dict["psd_path"]
# plt.subplot(4,2,7); plt.title("psd_path: open ({})".format(open_close_similarity_dict["psd_path"])); plt.hist(open_values)
# plt.subplot(4,2,8); plt.title("psd_path: close ({})".format(open_close_similarity_dict["psd_path"])); plt.hist(close_values)
#
# plt.tight_layout()
# plt.savefig('statistically_different_features.png')
# plt.show()
