#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 22 16:43 2025
Created in PyCharm
Created as picosec/get_timing_test

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
import uproot
from scipy.interpolate import CubicSpline


def main():
    file_path = '/local/home/dn277127/Bureau/picosec/Run299-Pool2_TESTBEAM_tree.root'
    tree_name = 'RawDataTree'
    branches = ['amplC1', 'amplC2']
    dt = 0.1  # ns
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        print(tree.keys())

        # Read the data into a numpy array
        data = tree.arrays(branches, library='np')
        amplC1, amplC2 = data['amplC1'], data['amplC2']

    ax = plot_waveform(amplC1[0], dt)
    plot_waveform(amplC2[0], dt, ax_in=ax)
    c1_time = get_timing(amplC1[0], dt, npt_fit=10, plot=True)
    c2_time = get_timing(amplC2[0], dt, npt_fit=20, plot=True)
    print(c1_time, c2_time, c2_time - c1_time)

    c1_time = get_timing_peak_fit(amplC1[0], dt, npt_fit=10, npt_from_peak=3, plot=True)
    c2_time = get_timing_peak_fit(amplC2[0], dt, npt_fit=20, npt_from_peak=5, plot=True)
    plt.show()

    timings, timings_spline, timings_npts_left, timings_sig_spline_max = [], [], [], []
    for event_i in range(len(amplC2)):
        print(f'Event {event_i}/{len(amplC2)}')
        c1_time = get_timing(amplC1[event_i], dt, npt_fit=10)
        c2_time = get_timing(amplC2[event_i], dt, npt_fit=20)
        c1_time_npt_left = get_timing(amplC1[event_i], dt, npt_fit=10, npt_from_peak=-1)
        c2_time_npt_left = get_timing(amplC2[event_i], dt, npt_fit=20, npt_from_peak=-2)
        c1_time_spline = get_timing_cubic_spline(amplC1[event_i], dt, npt_fit=10, npt_spline=5000)
        c2_time_spline = get_timing_cubic_spline(amplC2[event_i], dt, npt_fit=20, npt_spline=2000)
        c1_time_sig_spline_max = get_timing_sigmoid_spline_max(amplC1[event_i], dt, npt_fit=10, npt_spline=5000)
        c2_time_sig_spline_max = get_timing_sigmoid_spline_max(amplC2[event_i], dt, npt_fit=20, npt_spline=2000)
        if c1_time is not None and c2_time is not None:
            timings.append(c2_time - c1_time)
        if c1_time_npt_left is not None and c2_time_npt_left is not None:
            timings_npts_left.append(c2_time_npt_left - c1_time_npt_left)
        if c1_time_spline is not None and c2_time_spline is not None:
            timings_spline.append(c2_time_spline - c1_time_spline)
        if c1_time_sig_spline_max is not None and c2_time_sig_spline_max is not None:
            timings_sig_spline_max.append(c2_time_sig_spline_max - c1_time_sig_spline_max)

    fig, ax = plt.subplots()
    binning = np.linspace(4, 6, 200)
    # binning = 200
    ax.hist(timings, bins=binning, histtype='step', label='Sigmoid')
    ax.hist(timings_spline, bins=binning, histtype='step', label='Spline')
    ax.hist(timings_npts_left, bins=binning, histtype='step', label='Sigmoid Npt Left')
    ax.hist(timings_sig_spline_max, bins=binning, histtype='step', label='Sigmoid Spline Max')
    ax.set_title('Timing Difference Histogram')
    ax.set_xlabel('Timing Difference (ns)')
    ax.set_ylabel('Counts')
    ax.legend()

    plt.show()

    print('donzo')


def get_timing_2(y, dx=0.1):
    """ Get the timing of a waveform. """
    # Find the maximum value and its index
    max_index = np.argmax(y)
    max_value = y[max_index]

    # Find the half maximum value
    half_max = max_value / 2

    # Find the indices where the waveform crosses the half maximum
    crossing_indices = np.where(np.diff(np.sign(y - half_max)))[0]

    # Calculate the time at which the waveform crosses the half maximum
    if len(crossing_indices) >= 2:
        t1 = crossing_indices[0] * dx
        t2 = crossing_indices[1] * dx
        t_half = (t1 + t2) / 2
    else:
        t_half = None

    return t_half, max_value, max_index * dx


def get_timing(y, dx=0.1, npt_fit=20, npt_from_peak=1, plot=False):
    """ Get the timing of a waveform. """
    # Find the maximum value and its index
    min_index = np.argmin(y)
    min_value = y[min_index]

    # Fit the previous npt_fit points to a sigmoid function
    x_fit = np.arange(min_index - npt_fit, min_index + npt_from_peak) * dx
    y_fit = y[min_index - npt_fit:min_index + npt_from_peak]

    try:
        popt, pcov = cf(sigmoid, x_fit, y_fit, p0=[min_value, 5, np.mean(x_fit), 0])
        # Find 20% of the maximum value
        y_max = popt[0] + popt[3]
        y_frac = 0.2 * y_max
        x_timing = inverse_sigmoid(y_frac, *popt)
    except Exception as e:
        print(f"Error in curve fitting. Returning None. {e}")
        popt = [None, None, None]
        x_timing = None

    if plot:
        fig, ax = plt.subplots()
        x_plt = np.linspace(x_fit.min(), x_fit.max(), 200)
        ax.plot(x_fit, y_fit, 'o', label='Data')
        ax.axhline(y=0, color='gray', alpha=0.3, zorder=0)
        if popt[2] is not None:
            ax.plot(x_plt, sigmoid(x_plt, *popt), 'r-', label='Fitted Curve')
            ax.axvline(x=popt[2], color='g', linestyle='--', alpha=0.5, label='Sigmoid Timing')
        print(popt)
        ax.legend()

    return x_timing


def get_timing_peak_fit(y, dx=0.1, npt_fit=20, npt_from_peak=1, plot=False):
    """ Get the timing of a waveform. """
    # Find the maximum value and its index
    min_index = np.argmin(y)
    min_value = y[min_index]

    # Fit the previous npt_fit points to a sigmoid function
    x_fit = np.arange(min_index - npt_fit, min_index + npt_from_peak) * dx
    y_fit = y[min_index - npt_fit:min_index + npt_from_peak]

    try:
        popt, pcov = cf(peak_fit, x_fit, y_fit, p0=[min_value, 5, 1, np.mean(x_fit), 0])
        # Find 20% of the maximum value
        y_max = popt[0] + popt[3]
        y_frac = 0.2 * y_max
        # x_timing = inverse_sigmoid(y_frac, *popt)
        x_timing = None
    except Exception as e:
        print(f"Error in curve fitting. Returning None. {e}")
        popt = [None, None, None]
        x_timing = None

    if plot:
        fig, ax = plt.subplots()
        x_plt = np.linspace(x_fit.min(), x_fit.max(), 200)
        ax.plot(x_fit, y_fit, 'o', label='Data')
        ax.axhline(y=0, color='gray', alpha=0.3, zorder=0)
        if popt[2] is not None:
            ax.plot(x_plt, peak_fit(x_plt, *popt), 'r-', label='Fitted Curve')
            ax.axvline(x=popt[3], color='g', linestyle='--', alpha=0.5, label='Sigmoid Timing')
        print(popt)
        ax.legend()

    return x_timing


def get_timing_cubic_spline(y, dx=0.1, npt_fit=20, npt_spline=1000, plot=False):
    """ Get the timing of a waveform. """
    # Find the maximum value and its index
    min_index = np.argmin(y)

    # Fit the previous npt_fit points to a sigmoid function
    x_fit = np.arange(min_index - npt_fit // 2, min_index + npt_fit // 2) * dx
    y_fit = y[min_index - npt_fit // 2:min_index + npt_fit // 2]

    if len(y_fit) != len(x_fit):
        x_timing = None
    else:
        peak_spline = CubicSpline(x_fit, y_fit)
        x_spline = np.linspace(x_fit.min(), x_fit.max(), npt_spline)
        y_spline = peak_spline(x_spline)
        x_timing = x_spline[np.argmin(y_spline)]

    if plot:
        fig, ax = plt.subplots()
        x_plt = np.linspace(x_fit.min(), x_fit.max(), 200)
        ax.plot(x_fit, y_fit, 'o', label='Data')
        ax.axhline(y=0, color='gray', alpha=0.3, zorder=0)
        if x_timing is not None:
            ax.plot(x_plt, peak_spline(x_plt), 'r-', label='Fitted Curve')
        ax.legend()

    return x_timing


def get_timing_sigmoid_spline_max(y, dx=0.1, npt_fit=20, npt_from_peak=1, npt_spline=1000, plot=False):
    """ Get the timing of a waveform. """
    # Find the maximum value and its index
    min_index = np.argmin(y)
    min_value = y[min_index]

    # Fit the previous npt_fit points to a sigmoid function
    x_fit = np.arange(min_index - npt_fit, min_index + npt_from_peak) * dx
    y_fit = y[min_index - npt_fit:min_index + npt_from_peak]

    try:
        popt, pcov = cf(sigmoid, x_fit, y_fit, p0=[min_value, 5, np.mean(x_fit), 0])

        # Fit the previous npt_fit points to a sigmoid function
        x_spline_peak = np.arange(min_index - npt_fit // 2, min_index + npt_fit // 2) * dx
        y_spline_peak = y[min_index - npt_fit // 2:min_index + npt_fit // 2]

        if len(y_fit) != len(x_fit):
            raise ValueError

        peak_spline = CubicSpline(x_spline_peak, y_spline_peak)
        x_spline = np.linspace(x_spline_peak.min(), x_spline_peak.max(), npt_spline)
        y_spline_min = np.min(peak_spline(x_spline))

        # Find 20% of the maximum value
        y_min = y_spline_min + popt[3]
        y_frac = 0.2 * y_min
        x_timing = inverse_sigmoid(y_frac, *popt)
    except Exception as e:
        print(f"Error in curve fitting. Returning None. {e}")
        popt = [None, None, None]
        x_timing = None

    if plot:
        fig, ax = plt.subplots()
        x_plt = np.linspace(x_fit.min(), x_fit.max(), 200)
        ax.plot(x_fit, y_fit, 'o', label='Data')
        ax.axhline(y=0, color='gray', alpha=0.3, zorder=0)
        if x_timing is not None:
            ax.plot(x_plt, sigmoid(x_plt, *popt), 'r-', label='Fitted Curve')
            ax.axvline(x=x_timing, color='g', linestyle='--', alpha=0.5, label='Sigmoid Timing')
        print(popt)
        ax.legend()

    return x_timing


def plot_waveform(y, dx=0.1, title='Waveform', xlabel='Time (ns)', ylabel='Amplitude (mV)', ax_in=None):
    """ Plot a waveform. """
    if ax_in is None:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax = ax_in
    ax.plot(np.arange(len(y)) * dx, y)

    return ax


def sigmoid(x, a, k, x0, c):
    """ Sigmoid function with offset. """
    return a / (1 + np.exp(-k * (x - x0))) + c


def peak_fit(x, a, k_rise, k_fall, x0, c):
    return a / (1 + np.exp(-k_rise * (x - x0))) / (1 + np.exp(k_fall * (x - x0))) + c


def inverse_sigmoid(y, a, k, x0, c):
    """ Inverse of the sigmoid function with offset. """
    if k == 0:
        raise ValueError("k cannot be zero.")
    if a == 0:
        raise ValueError("a cannot be zero.")

    # Subtract offset
    y_shifted = y - c

    # Invert sigmoid
    log_arg = a / y_shifted - 1
    x = -np.log(log_arg) / k + x0
    return x


if __name__ == '__main__':
    main()
