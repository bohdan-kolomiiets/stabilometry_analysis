from scipy.signal import butter, lfilter, freqz
import sys, os
# from .nolds_pkg.nolds.measures import dfa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, Wn=[low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)

    # w, h = freqz(b, a, worN=2000)
    # plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    y = lfilter(b, a, data)
    return y


def plot_result_wrapper(vals_dict, x_label="x", y_label="y", data_label="data",
                        reg_label="regression line", fname=None, plot_title=None):
    """
    Helper function to plot the various polyorder curves

    :param vals_dict: input dictionary

        should fit the following example:
            vals_dict = {
                order_num (int) : dfa_output (tuple)
            }

        Ex:
            vals_dict = {
                1 : (poly[0], (np.log(nvals), np.log(fluctuations), poly)),
                2 : (poly[0], (np.log(nvals), np.log(fluctuations), poly)),
                ...
            }

        See dfa return values in nolds package

    :param x_label: name of the x-axis
    :param y_label: name of the y-axis
    :param data_label: label of the data curve to place in legend
    :param reg_label: label of the regretion curve to place in legend
    :param fname: full path+name+extention of the file you want to save the plot
    :param plot_title: plot title

    """

    plt.figure()

    for order in list(vals_dict.keys()):

        xvals = vals_dict[order][0]
        yvals = vals_dict[order][1]
        poly = vals_dict[order][2]

        color = "b"

        # TODO: add random color generation
        if order == 1:
            color = "r"
        elif order == 2:
            color = "b"
        elif order == 3:
            color = "g"
        else:
            color = "y"

        order_str = ",ord:" + str(order)

        plt.plot(xvals, yvals, color + "o", label=data_label + order_str)

        if not (poly is None):
            plt.plot(xvals, np.polyval(poly, xvals), color + "-", label=reg_label + order_str)

    if not (plot_title is None):
        plt.title(plot_title)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # TODO: add automatic limits generation function
    plt.xlim(1, 6)
    plt.ylim(-8, 8)

    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig(fname)

    # plt.close()


def get_polyorder_curves(signal=None, max_order=3, debug_data=False, debug_graphs=False, **kwargs):
    """

    Function fot getting the various polynomial order curves from the original data.

    :param signal: the input signal
    :param max_order: maximum polynomial order to ge the DFA
    :param debug_data:
        if True returns the dictionary with outputs of the DFA
        if False returns nothing

    :param debug_graphs:
        if True to save or show the plot of the graphs with different polyorder curves
        if True need to specify additional params, see kwargs

    :param kwargs:

        Possible arguments:
        :param output_folder: output folder to save the debug plots
        :param plot_name: default name of the plot, could be the name of the feature you're trying to apply the DFA to

    :return:
    """

    # input vars check
    if signal is None:
        raise ValueError("Please, define the 'signal' parameter!")

    overlap = kwargs["overlap"]

    if debug_graphs:

        if "output_folder" in kwargs:
            output_folder = kwargs["output_folder"]
        else:
            raise ValueError("Please specify the output folder!")

        if "plot_name" in kwargs:
            plot_name = kwargs["plot_name"]
        else:
            raise ValueError("Please specify the plot_name!")

    # dict to hold the results of the dfa analysis
    dfa_log_data = {}

    # window sizes (max possible range of the data)
    nvals_full = range(4, int(len(signal) * 0.1))

    # polynomial orders to get the DFA result
    orders = range(1, max_order + 1)

    # for every order do DFA
    for order in orders:
        dfa_log_data.update({order: ()})

        # get the DFA data from the standard package
        exponent_result = dfa(data=signal,
                              nvals=nvals_full,
                              overlap=overlap,
                              debug_plot=False,
                              plot_file="d",
                              order=order,
                              debug_data=True)

        # save result to the dict
        dfa_log_data[order] = dfa_log_data[order] + exponent_result[1]

    if debug_graphs:
        # plot the results of the different polyorder curves
        plot_result_wrapper(vals_dict=dfa_log_data,
                            x_label="log(n)",
                            y_label="log(F(n))",
                            fname=output_folder + plot_name + ".png",
                            plot_title=plot_name)

    if debug_data:
        return dfa_log_data
    else:
        pass


def get_aic(original_signal, regression_type):
    # TODO: add checking len
    # TODO: add checking tuple

    k = 0

    if regression_type == "linear":
        k = 3

    from sklearn import linear_model

    xvals = original_signal[0].reshape((len(original_signal[0]), 1))
    yvals = original_signal[1].reshape((len(original_signal[1]), 1))

    regr = linear_model.LinearRegression()
    regr.fit(xvals, yvals)
    predicted_signal = regr.predict(xvals)

    debug = True
    if debug:
        plt.figure(1)
        plt.plot(xvals, yvals, 'b-')
        plt.plot(xvals, predicted_signal, 'r-')
        plt.grid(True)
        plt.show()

    resid = yvals - predicted_signal
    sse = np.sum(resid ** 2)

    AIC = 2 * k - (2 * np.log(sse))

    return AIC


def aic(y_observed, y_predicted, k):
    return 2 * k + y_predicted.size * np.log(np.sum(np.sqrt(((y_predicted - y_observed) ** 2)))/y_predicted.size)


x0 = 0


def func(x, a0, a1, a2, a3):
    # Function that includes x0 (bending point) as a parameter
    global x0

    I = np.ones_like(x)
    I[np.argwhere(x <= x0)] = 0

    return a0 * I + a1 * I * x + a2 * (1 - I) + a3 * x * (1 - I)

    # return a0 + a1 * x + a2 * (x - x0) * I


def piecewise_linear(x, x0, a0, a1, a2):
    condlist = [x <= x0, x > x0]
    funclist = [lambda x: a0 + a1 * x + a2 * (x - x0) * 0, lambda x: a0 + a1 * x + a2 * (x - x0) * 1]
    return np.piecewise(x, condlist, funclist)


def get_bending_point(debug=[False, None], **kwargs):
    """
    Function for automatically getting the bending point of the DFA log-log plots

    :param kwargs:

        Check the get_polyorder_curves function for understanding what kwargs to pass

    :return:
    """

    dfa_data = get_polyorder_curves(**kwargs)

    # TODO: add Alkaike creterion for estimating the bending point of the log-log plots
    bending_point = 0

    from statsmodels.regression.linear_model import OLS
    # from statsmodels.tools import add_constant

    # get the number of points in the X and Y vector
    # vect_len = len(dfa_data[1][1])

    # popt, pcov = curve_fit(func, dfa_data[2][0], dfa_data[2][1])  # fitting our model to the observed data

    # plt.figure()
    # p, e = curve_fit(piecewise_linear, dfa_data[2][0], dfa_data[2][1])
    # xd = np.linspace(np.min(dfa_data[2][0]), np.max(dfa_data[2][0]), 100)
    # plt.plot(dfa_data[2][0], dfa_data[2][1], "o")
    # plt.plot(xd, piecewise_linear(xd, *p))

    order = 2
    xvals = dfa_data[order][0]
    yvals = dfa_data[order][1]

    aic_vals = []
    x_points = []

    from scipy.optimize import curve_fit

    for i, point in enumerate(xvals):

        if i > 2:
            global x0
            x0 = point
            popt, pcov = curve_fit(func, xvals, yvals)  # fitting our model to the observed data
            y_predicted = func(xvals, *popt)  # calculating predictions, given by our fitted model
            aic_vals.append(aic(y_observed=yvals, y_predicted=y_predicted, k=5))
            x_points.append(point)

    bending_point = np.min(aic_vals)
    x_bend_p = aic_vals.index(bending_point)


    if debug[0]:
        plt.figure()
        plt.plot(x_points, aic_vals, 'b-', label='AIC(x)')
        plt.plot(x_points[x_bend_p], bending_point, '-gD', label="x_pos=%5.3f, b_val=%5.3f" %(x_points[x_bend_p], bending_point))
        # y_predicted = func(dfa_data[2][0], *popt)  # calculating predictions, given by our fitted model
        # plt.stem(dfa_data[2][0], y_predicted, 'r-',
        #          label='fit: a0=%5.3f, a1=%5.3f, a2=%5.3f, a3=%5.3f' % tuple(popt))
        plt.xlabel('x')
        plt.ylabel('AIC')
        plt.xlim(np.min(xvals), )
        plt.legend()

        if debug[1] is not None:
            plt.savefig(debug[1]+":AIC(x)")
            plt.close()
        else:
            plt.show()


        x_ind = np.where(xvals == x_points[x_bend_p])

        plt.figure()
        plt.plot(xvals, yvals, 'b-', label='Observed data')
        plt.plot(xvals[x_ind], yvals[x_ind], '-gD', label="Bending point")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        if debug[1] is not None:
            plt.savefig(debug[1]+",point")
            plt.close()
        else:
            plt.show()

    return int(np.exp(x_points[x_bend_p]))


def get_dfa_raw_signal_bands(signal=None, bands=None, fs=None, filt_order=5, poly_order=1, nvals=[]):
    raw_sig_data = {}
    dfa_raw_sig_data = {}
    nvals_full = range(4, int(len(signal) * 0.1))

    for i, band in enumerate(list(bands.keys())):
        filt_sig = butter_bandpass_filter(data=signal,
                                          lowcut=bands[band],
                                          highcut=bands[band],
                                          fs=fs,
                                          order=filt_order)

        raw_sig_data.update({band: filt_sig})

        dfa_value = dfa(data=filt_sig,
                        nvals=nvals,
                        overlap=True,
                        debug_plot=False,
                        plot_file=band,
                        order=poly_order,
                        debug_data=True)

        dfa_raw_sig_data.update({band: dfa_value})

    return raw_sig_data, dfa_raw_sig_data

def perform_dfa(**kwargs):

    signal = kwargs["signal"]
    overlap = kwargs["overlap"]

    bending_point = get_bending_point(**kwargs)

    nvals_new = range(bending_point, int(len(signal) * 0.1))  # signal part after exp(2.5)

    dfa_value = dfa(data=signal,
                          nvals=nvals_new,
                          overlap=overlap,
                          debug_plot=False,
                          plot_file="d",
                          order=2,
                          debug_data=False)

    return dfa_value

def perform_new_dfa(**kwargs):

    signal = kwargs["signal"]
    overlap = kwargs["overlap"]
    order = 2

    # window sizes (max possible range of the data)
    nvals_full = range(4, int(len(signal) * 0.1))

    dfa_tuple = dfa(data=signal,
                    nvals=nvals_full,
                    overlap=overlap,
                    debug_plot=False,
                    plot_file="d",
                    order=order,
                    debug_data=True)

    log_n = dfa_tuple[1][0]
    log_F = dfa_tuple[1][1]

    exp_p, point_loc = get_log_log_plot_bending_point(xvals=log_n, yvals=log_F)

    alpha = calculate_alpha_exp(log_n=log_n, log_F=log_F, bending_point = point_loc)

    return alpha[0]

def calculate_alpha_exp(log_n=None, log_F=None, bending_point=None):

    bend_point_x = bending_point

    x = np.expand_dims(np.asarray(log_n[bend_point_x:len(log_n)]), axis=1)
    y = np.asarray(log_F[bend_point_x:len(log_F)])
    # x = np.expand_dims(np.asarray(log_n[0:bend_point_x+1]), axis=1)
    # y = np.asarray(log_F[0:bend_point_x+1])


    reg = LinearRegression().fit(x, y)
    reg.score(x, y)

    print("Coeffs: ")
    print(reg.coef_)

    return ( reg.coef_[0], reg.intercept_ )

def get_log_log_plot_bending_point(xvals=None, yvals=None):

    aic_vals = []
    x_points = []

    from scipy.optimize import curve_fit

    for i, point in enumerate(xvals):
        if i > 2:
            global x0
            x0 = point
            popt, pcov = curve_fit(func, xvals, yvals)  # fitting our model to the observed data
            y_predicted = func(xvals, *popt)  # calculating predictions, given by our fitted model
            aic_vals.append(aic(y_observed=yvals, y_predicted=y_predicted, k=5))
            x_points.append(point)

    bending_point = np.min(aic_vals)
    x_bend_p = aic_vals.index(bending_point)

    # if debug[0]:
    #     plt.figure()
    #     plt.plot(x_points, aic_vals, 'b-', label='AIC(x)')
    #     plt.plot(x_points[x_bend_p], bending_point, '-gD',
    #              label="x_pos=%5.3f, b_val=%5.3f" % (x_points[x_bend_p], bending_point))
    #     # y_predicted = func(dfa_data[2][0], *popt)  # calculating predictions, given by our fitted model
    #     # plt.stem(dfa_data[2][0], y_predicted, 'r-',
    #     #          label='fit: a0=%5.3f, a1=%5.3f, a2=%5.3f, a3=%5.3f' % tuple(popt))
    #     plt.xlabel('x')
    #     plt.ylabel('AIC')
    #     plt.xlim(np.min(xvals), )
    #     plt.legend()
    #
    #     if debug[1] is not None:
    #         plt.savefig(debug[1] + ":AIC(x)")
    #         plt.close()
    #     else:
    #         plt.show()
    #
    #     x_ind = np.where(xvals == x_points[x_bend_p])
    #
    #     plt.figure()
    #     plt.plot(xvals, yvals, 'b-', label='Observed data')
    #     plt.plot(xvals[x_ind], yvals[x_ind], '-gD', label="Bending point")
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.legend()
    #
    #     if debug[1] is not None:
    #         plt.savefig(debug[1] + ",point")
    #         plt.close()
    #     else:
    #         plt.show()

    return int(np.exp(x_points[x_bend_p])), x_bend_p




def get_log_log_plot_bending_point_with_debug(xvals=None, yvals=None, debug=(None, None, None, None)):

    to_debug, to_save, output_folder, plot_num = debug

    aic_vals = []
    x_points = []

    from scipy.optimize import curve_fit

    for i, point in enumerate(xvals):

        if i > 2:
            global x0
            x0 = point
            popt, pcov = curve_fit(func, xvals, yvals)  # fitting our model to the observed data
            y_predicted = func(xvals, *popt)  # calculating predictions, given by our fitted model
            aic_vals.append(aic(y_observed=yvals, y_predicted=y_predicted, k=5))
            x_points.append(point)

    bending_point = np.min(aic_vals)
    x_bend_p = aic_vals.index(bending_point)

    if to_debug:
        plt.figure()
        plt.plot(x_points, aic_vals, 'b-', label='AIC(x)')
        plt.plot(x_points[x_bend_p], bending_point, '-gD',
                 label="x_pos=%5.3f, b_val=%5.3f" % (x_points[x_bend_p], bending_point))
        plt.xlabel('x')
        plt.ylabel('AIC')
        plt.xlim(np.min(xvals), )
        plt.legend()

        if to_save is not None:
            plt.savefig(output_folder + "AIC(x)."+str(plot_num)+".png")
            plt.close()

        x_ind = np.where(xvals == x_points[x_bend_p])

        plt.figure()
        plt.plot(xvals, yvals, 'b-', label='Observed data')
        plt.plot(xvals[x_ind], yvals[x_ind], '-gD', label="Bending point")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        if to_save is not None:
            plt.savefig(output_folder+"bending_point."+str(plot_num)+".png")
            plt.close()
        else:
            plt.show()

    return int(np.exp(x_points[x_bend_p])), x_bend_p