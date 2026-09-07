#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on May 03 12:45 PM 2025
Created in PyCharm
Created as picosec/double_peak_investigation.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import lecroyparser

from Measure import Measure
from get_timing_test import *


def main():
    trc_dir = '/media/ucla/picosec/Run358/'
    # plot_waveforms(trc_dir)
    # check_c1_c2_xs(trc_dir)
    # check_c1_c2_xs_files_print(trc_dir)
    # check_c1_c2_xs_files(trc_dir)
    # get_timing_from_trc_files(trc_dir)
    compare_timing_algs(trc_dir)

    print('donzo')


def plot_waveforms(trc_dir):
    trc_file = 'C1--Trace--00153.trc'
    points_per_waveform = 10002
    data = lecroyparser.ScopeData(f'{trc_dir}{trc_file}')
    print(data)
    print(data.horizOffset)
    print(data.waveArrayCount)
    print(data.parseInt16(120))
    print(data.posWAVEDESC)
    print(data.parseInt16(144))
    # Print all data attributes
    xs = data.x.reshape(-1, points_per_waveform)
    ys = data.y.reshape(-1, points_per_waveform)
    print(f'xs shape: {xs.shape}')
    print(f'ys shape: {ys.shape}')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig_waveforms, ax_waveforms = plt.subplots(figsize=(10, 6))
    for i in range(5):
        print(xs[i][0:2])
        ax.plot(xs[i], marker='.', ls='none')
        ax_waveforms.plot(xs[i], ys[i], marker='.', ls='none')
    print([xs[i][0]])
    plot_waveform(ys[0], 0.1)
    plt.show()


def check_c1_c2_xs(trc_dir):
    points_per_waveform = 10002
    file_num = '00152'
    trc_c1_file = f'C1--Trace--{file_num}.trc'
    trc_c2_file = f'C2--Trace--{file_num}.trc'
    data_c1 = lecroyparser.ScopeData(f'{trc_dir}{trc_c1_file}')
    data_c2 = lecroyparser.ScopeData(f'{trc_dir}{trc_c2_file}')
    print(f'data_c1.x: {data_c1.x}')
    print(f'data_c2.x: {data_c2.x}')
    print(f'data_c1.diff: {np.diff(data_c1.x)}')
    print(f'data_c2.diff: {np.diff(data_c2.x)}')
    print(f'data_c1.diff hist: {np.histogram(np.diff(data_c1.x), bins=10)}')

    # print(f'data_c1.diff mean and std: {np.mean(np.diff(data_c1.x))} +- {np.std(np.diff(data_c1.x))}')

    print(f'data_c2.x - data_c1.x: {data_c2.x - data_c1.x}')
    print(np.unique(data_c2.x - data_c1.x, return_counts=True))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(np.diff(data_c1.x), bins=100, histtype='step', label='C1')
    ax.hist(np.diff(data_c2.x), bins=100, histtype='step', label='C2')
    ax.set_xlabel('x diff')
    ax.set_ylabel('Counts')
    ax.set_title('C1 and C2 x diff')
    ax.legend()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(data_c2.x - data_c1.x, bins=100, histtype='step', label='C2 - C1')
    ax2.set_xlabel('C2 - C1')
    ax2.set_ylabel('Counts')
    ax2.set_title('C2 - C1')
    ax2.legend()

    plt.show()


def check_c1_c2_xs_files_print(trc_dir):
    n_files = 265
    file_nums = [str(i).zfill(5) for i in range(0, n_files + 1)]
    for file_num in file_nums:
        trc_c1_file = f'C1--Trace--{file_num}.trc'
        trc_c2_file = f'C4--Trace--{file_num}.trc'
        data_c1 = lecroyparser.ScopeData(f'{trc_dir}{trc_c1_file}')
        data_c2 = lecroyparser.ScopeData(f'{trc_dir}{trc_c2_file}')
        c2_c1_diffs_i = (data_c2.x - data_c1.x) * 1e12  # Convert to ps
        uniques, unique_counts = np.unique(c2_c1_diffs_i, return_counts=True)
        print(f'file_num: {file_num}, min-max: {uniques[0]} - {uniques[-1]}ps')


def check_c1_c2_xs_files(trc_dir):
    n_files = 265
    file_nums = [str(i).zfill(5) for i in range(0, n_files + 1)]
    c1_diffs, c2_diffs, c2_c1_diffs = [], [], []
    for file_num in file_nums:
        print(f'file_num: {file_num}')
        trc_c1_file = f'C1--Trace--{file_num}.trc'
        trc_c2_file = f'C4--Trace--{file_num}.trc'
        data_c1 = lecroyparser.ScopeData(f'{trc_dir}{trc_c1_file}')
        data_c2 = lecroyparser.ScopeData(f'{trc_dir}{trc_c2_file}')
        c1_diffs_i = np.diff(data_c1.x)
        c2_diffs_i = np.diff(data_c2.x)
        c2_c1_diffs_i = data_c2.x - data_c1.x

        c1_diffs.append(Measure(c1_diffs_i.mean(), c1_diffs_i.std()) * 1e12)  # Convert to ps
        c2_diffs.append(Measure(c2_diffs_i.mean(), c2_diffs_i.std()) * 1e12)  # Convert to ps
        c2_c1_diffs.append(Measure(c2_c1_diffs_i.mean(), c2_c1_diffs_i.std()) * 1e12)  # Convert to ps

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(np.arange(len(c1_diffs)), [c.val for c in c1_diffs], yerr=[c.err for c in c1_diffs], ls='none',
                marker='o', alpha=0.4, label='C1')
    ax.errorbar(np.arange(len(c2_diffs)), [c.val for c in c2_diffs], yerr=[c.err for c in c2_diffs], ls='none',
                marker='o', alpha=0.4, label='C2')
    ax.set_xlabel('File Number')
    ax.set_ylabel('Time Step Between Points (ps)')
    ax.set_title('C1 and C2 x diff')
    ax.legend()
    fig.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.errorbar(np.arange(len(c2_c1_diffs)), [c.val for c in c2_c1_diffs], yerr=[c.err for c in c2_c1_diffs],
                 marker='o', ls='none', label='C2 - C1')
    ax2.set_xlabel('File Number')
    ax2.set_ylabel('Channel Time Offset (ps)')
    ax2.set_title('C2 - C1')
    ax2.legend()
    fig2.tight_layout()
    plt.show()


def get_timing_from_trc_files(trc_dir):
    n_files = 265
    c1_threshold, c2_threshold = -0.1, -0.1
    file_nums = [str(i).zfill(5) for i in range(0, n_files + 1)]
    time_diffs_good, time_diffs_bad, time_diffs_very_good, time_diffs_matlab = [], [], [], []
    for file_num in file_nums:
        print(f'file_num: {file_num}')
        trc_c1_file = f'C1--Trace--{file_num}.trc'
        trc_c2_file = f'C4--Trace--{file_num}.trc'
        data_c1 = lecroyparser.ScopeData(f'{trc_dir}{trc_c1_file}')
        data_c2 = lecroyparser.ScopeData(f'{trc_dir}{trc_c2_file}')
        c1_events = get_lecroy_nsegments(data_c1)
        c2_events = get_lecroy_nsegments(data_c2)

        c1_x = data_c1.x.reshape(c1_events, -1) * 1e9  # Convert to ns
        c1_y = data_c1.y.reshape(c1_events, -1)
        c2_x = data_c2.x.reshape(c2_events, -1) * 1e9  # Convert to ns
        c2_y = data_c2.y.reshape(c2_events, -1)

        # print(data_c1.trigTimeArray)
        # print(data_c1.parseDouble(data_c1.trigTimeArray))
        # print(data_c1.triggerTime)
        c1_time, c1_offset = get_trigger_offset(data_c1)
        c2_time, c2_offset = get_trigger_offset(data_c2)
        c1_offset, c2_offset = c1_offset * 1e9, c2_offset * 1e9  # Convert to ns
        c1_time, c2_time = c1_time * 1e9, c2_time * 1e9  # Convert to ns

        if c1_events != c2_events:
            print(f'Warning: C1 and C2 have different number of events: {c1_events} vs {c2_events}')
            continue
        for i in range(c1_events):
            if c1_y[i].min() > c1_threshold or c2_y[i].min() > c2_threshold:
                # print(f'Event {i}: C1 min: {c1_y[i].min()}, C2 min: {c2_y[i].min()}')
                continue
            c1_timing_good = get_timing(c1_y[i], c1_x[i] - c1_x[i][0], npt_fit=10, npt_from_peak=1)
            c2_timing_good = get_timing(c2_y[i], c2_x[i] - c1_x[i][0], npt_fit=20, npt_from_peak=1)
            c1_timing_bad = get_timing(c1_y[i], c1_x[i] - c1_x[i][0], npt_fit=10, npt_from_peak=1)
            c2_timing_bad = get_timing(c2_y[i], c2_x[i] - c2_x[i][0], npt_fit=20, npt_from_peak=1)

            c1_t, c2_t = c1_x[i] - c1_x[i][0] + c1_offset[i], c2_x[i] - c2_x[i][0] + c2_offset[i]
            earliest_time = min(c1_t[0], c2_t[0])
            c1_time_very_good = get_timing(c1_y[i], c1_t - earliest_time, npt_fit=10, npt_from_peak=1)
            c2_time_very_good = get_timing(c2_y[i], c2_t - earliest_time, npt_fit=20, npt_from_peak=1)

            c1_dt, c2_dt = c1_x[i][1] - c1_x[i][0], c2_x[i][1] - c2_x[i][0]
            c1_t, c2_t = np.arange(len(c1_y[i])) * c1_dt + c1_offset[i], np.arange(len(c2_y[i])) * c2_dt + c2_offset[i]
            earliest_time = min(c1_t[0], c2_t[0])
            c1_time_matlab = get_timing(c1_y[i], c1_t - earliest_time, npt_fit=10, npt_from_peak=1)
            c2_time_matlab = get_timing(c2_y[i], c2_t - earliest_time, npt_fit=20, npt_from_peak=1)

            if (c1_timing_good is None or c2_timing_good is None or c1_timing_bad is None or c2_timing_bad is None or
                    c1_time_very_good is None or c2_time_very_good is None or c1_time_matlab is None or c2_time_matlab is None):
                print(f'Error in timing calculation for event {i}. Skipping.')
                continue
            time_diffs_good.append(c2_timing_good - c1_timing_good)
            time_diffs_bad.append(c2_timing_bad - c1_timing_bad)
            time_diffs_very_good.append(c2_time_very_good - c1_time_very_good)
            time_diffs_matlab.append(c2_time_matlab - c1_time_matlab)
            # print(f'Event {i}: C1 timing: {c1_timing}, C2 timing: {c2_timing}')
            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.plot(c1_x[i], c1_y[i], marker='.', ls='none', color='green', label=f'C1')
            # ax.axvline(c1_timing, c='green', ls='--')
            # ax.plot(c2_x[i], c2_y[i], marker='.', ls='none', color='red', label=f'C2')
            # ax.axvline(c2_timing, c='red', ls='--')
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.set_title('C1 and C2 waveforms')
            # ax.legend()
            # plt.show()
    fig, ax = plt.subplots(figsize=(10, 6))
    binning = np.linspace(-8.5, -7.5, 200)
    # ax.hist(time_diffs_good, bins=binning, histtype='stepfilled', alpha=0.4, label='Good Events')
    # ax.hist(time_diffs_bad, bins=binning, histtype='stepfilled', alpha=0.4, label='Bad Events')
    ax.hist(time_diffs_very_good, bins=binning, histtype='stepfilled', alpha=0.4, label='Very Good Events')
    ax.hist(time_diffs_matlab, bins=binning, histtype='stepfilled', alpha=0.4, label='Matlab Events')
    ax.set_xlabel('C2 - C1 Timing Difference (ps)')
    ax.set_ylabel('Counts')
    ax.set_title('C2 - C1 Timing Difference')
    ax.legend()
    fig.tight_layout()
    plt.show()


def compare_timing_algs(trc_dir):
    n_files = 265
    c1_threshold, c2_threshold = -0.1, -0.1
    file_nums = [str(i).zfill(5) for i in range(0, n_files + 1)]
    time_diffs, time_diffs_spline_max = [], []
    for file_num in file_nums:
        print(f'file_num: {file_num}')
        trc_c1_file = f'C1--Trace--{file_num}.trc'
        trc_c2_file = f'C4--Trace--{file_num}.trc'
        data_c1 = lecroyparser.ScopeData(f'{trc_dir}{trc_c1_file}')
        data_c2 = lecroyparser.ScopeData(f'{trc_dir}{trc_c2_file}')
        c1_events = get_lecroy_nsegments(data_c1)
        c2_events = get_lecroy_nsegments(data_c2)

        c1_x = data_c1.x.reshape(c1_events, -1) * 1e9  # Convert to ns
        c1_y = data_c1.y.reshape(c1_events, -1)
        c2_x = data_c2.x.reshape(c2_events, -1) * 1e9  # Convert to ns
        c2_y = data_c2.y.reshape(c2_events, -1)

        c1_time, c1_offset = get_trigger_offset(data_c1)
        c2_time, c2_offset = get_trigger_offset(data_c2)
        c1_offset, c2_offset = c1_offset * 1e9, c2_offset * 1e9  # Convert to ns

        if c1_events != c2_events:
            print(f'Warning: C1 and C2 have different number of events: {c1_events} vs {c2_events}')
            continue
        for i in range(c1_events):
            if c1_y[i].min() > c1_threshold or c2_y[i].min() > c2_threshold:
                continue
            c1_t, c2_t = c1_x[i] - c1_x[i][0] + c1_offset[i], c2_x[i] - c2_x[i][0] + c2_offset[i]
            earliest_time = min(c1_t[0], c2_t[0])
            c1_time = get_timing(c1_y[i], c1_t - earliest_time, npt_fit=10, npt_from_peak=1)
            c2_time = get_timing(c2_y[i], c2_t - earliest_time, npt_fit=20, npt_from_peak=1)

            c1_time_cubic = get_timing_sigmoid_spline_max(c1_y[i], c1_t - earliest_time, npt_fit=10, npt_from_peak=1, plot=False)
            c2_time_cubic = get_timing_sigmoid_spline_max(c2_y[i], c2_t - earliest_time, npt_fit=20, npt_from_peak=1, plot=False)

            if c1_time and c2_time:
                time_diffs.append(c2_time - c1_time)
            if c1_time_cubic and c2_time_cubic:
                time_diffs_spline_max.append(c2_time_cubic - c1_time_cubic)

    fig, ax = plt.subplots(figsize=(10, 6))
    binning = np.linspace(-8.5, -7.5, 200)
    ax.hist(time_diffs, bins=binning, histtype='step', alpha=1, label='Nominal')
    ax.hist(time_diffs_spline_max, bins=binning, histtype='step', alpha=1, label='Cubic Spline Max')
    ax.set_xlabel('C2 - C1 Timing Difference (ps)')
    ax.set_ylabel('Counts')
    ax.set_title('C2 - C1 Timing Difference')
    ax.legend()
    fig.tight_layout()
    plt.show()


def get_lecroy_nsegments(scope_data):
    """
    Get the number of segments in the Lecroy file. --> Number of files
    :param scope_data: ScopeData object
    :return: number of segments
    """
    n_segments = scope_data.parseInt16(144)
    return n_segments


def get_trigger_offset(scope_data):
    """Returns the precise trigger offset as a float in seconds,
    if the TRIGTIME_ARRAY section is present."""
    if scope_data.trigTimeArray == 0:
        raise ValueError("No TRIGTIME_ARRAY section found in the file.")

    pos = scope_data.posWAVEDESC + scope_data.waveDescriptor + scope_data.userText

    # Trigger time is a double at the start of the TRIGTIME_ARRAY
    n_segments = get_lecroy_nsegments(scope_data)
    r = np.frombuffer(scope_data.data, dtype=scope_data.endianness + "f8", count=n_segments * 2, offset=pos)

    # r = np.frombuffer(scope_data.data[pos:pos + length], dtype=np.dtype((scope_data.endianness + "f8", 1600)), count=1)[0]
    # trigger_offset = np.frombuffer(scope_data.data[pos:pos + 8], dtype=scope_data.endianness + "f8")[0]
    trigger_time = r[::2]
    trigger_offset = r[1::2]
    return trigger_time, trigger_offset



def get_timing(y, x, npt_fit=20, npt_from_peak=1, plot=False, return_fit=False):
    """ Get the timing of a waveform. """
    # Find the maximum value and its index
    min_index = np.argmin(y)
    min_value = y[min_index]

    # Fit the previous npt_fit points to a sigmoid function
    x_fit = x[min_index - npt_fit:min_index + npt_from_peak]
    y_fit = y[min_index - npt_fit:min_index + npt_from_peak]

    try:
        popt, pcov = cf(sigmoid, x_fit, y_fit, p0=[min_value, 5, np.mean(x_fit), 0], maxfev=10000)
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

    if return_fit:
        return x_timing, popt
    return x_timing


def get_timing_sigmoid_spline_max(y, x, npt_fit=20, npt_from_peak=1, npt_spline=1000, plot=False):
    """ Get the timing of a waveform. """
    # Find the maximum value and its index
    min_index = np.argmin(y)
    min_value = y[min_index]

    # Fit the previous npt_fit points to a sigmoid function
    x_fit = x[min_index - npt_fit:min_index + npt_from_peak]
    y_fit = y[min_index - npt_fit:min_index + npt_from_peak]

    try:
        popt, pcov = cf(sigmoid, x_fit, y_fit, p0=[min_value, 5, np.mean(x_fit), 0])

        # Fit the previous npt_fit points to a sigmoid function
        x_spline_peak = x[min_index - npt_fit // 2:min_index + npt_fit // 2]
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
        ax.set_xlabel('Time (ns)')
        fig.tight_layout()

        fig_spline, ax_spline = plt.subplots()
        ax_spline.plot(x_spline_peak, y_spline_peak, 'o', label='Data')
        ax_spline.plot(x_spline, peak_spline(x_spline), 'r-', label='Spline')
        ax_spline.axhline(y=0, color='gray', alpha=0.3, zorder=0)
        if x_timing is not None:
            ax_spline.axvline(x=x_timing, color='g', linestyle='--', alpha=0.5, label='Sigmoid Timing')
        ax_spline.legend()
        ax_spline.set_xlabel('Time (ns)')
        fig.tight_layout()

    return x_timing


if __name__ == '__main__':
    main()
