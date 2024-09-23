#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 21 17:04 2024
Created in PyCharm
Created as picosec/read_decoded

@author: Dylan Neff, dn277127
"""

import numpy as np
from scipy.optimize import curve_fit as cf
import matplotlib.pyplot as plt
import pandas as pd

import uproot

from Measure import Measure


def main():
    root_path = '/local/home/dn277127/Bureau/picosec/dataTrees/Run081-Pool3_TESTBEAMraw_tree.root'
    chunk_size = 1000  # Number of events per chunk

    read_root_file(root_path, chunk_size)
    # sigmoid_plot_test()

    print('donzo')


def read_root_file(root_path, chunk_size):
    branches = ['amplC1', 'amplC2', 'amplC3', 'amplC4', 't0', 'dt', 'eventNo']

    start_event = 800 * 15
    stop_event = 800 * 20

    file_data = []
    with uproot.open(root_path) as file:
        tree = file[file.keys()[0]]
        print(tree.keys())

        for data in tree.iterate(branches, library='np', step_size=chunk_size, entry_start=start_event, entry_stop=stop_event):
            file_data.append(analyze_chunk(data))

    analyze_timing(file_data)


def analyze_timing(data):
    waveform_fits_C1, waveform_fits_C2 = [], []
    for chunk in data:
        waveform_fits_C1_chunk, waveform_fits_C2_chunk = chunk
        if waveform_fits_C1_chunk is not None:
            waveform_fits_C1.extend(waveform_fits_C1_chunk)
        if waveform_fits_C2_chunk is not None:
            waveform_fits_C2.extend(waveform_fits_C2_chunk)

    df_C1 = pd.DataFrame(waveform_fits_C1)
    df_C2 = pd.DataFrame(waveform_fits_C2)

    # Plot histogram of df_C1['frac_time'] - df_C2['frac_time']
    # time_avg_diff = np.mean(df_C1['frac_time']) - np.mean(df_C2['frac_time'])
    time_diff = df_C1['frac_time'] - df_C2['frac_time']
    # time_diff = time_diff - np.mean(time_diff)
    print(f'Mean time difference: {np.mean(time_diff)}')
    time_diff = time_diff[abs(time_diff) < 20]

    fig, ax = plt.subplots()
    ax.hist(time_diff, bins=100)
    ax.set_xlabel('C1 - C2 Time Difference (ns)')
    ax.set_ylabel('Counts')
    ax.set_title('Filtered Time Difference')
    fig.tight_layout()

    time_diff_shifted = time_diff - np.mean(time_diff)
    time_diff_shifted = time_diff_shifted[abs(time_diff_shifted) < 2]
    time_diff_shifted = time_diff_shifted - np.mean(time_diff_shifted)

    counts, bins = np.histogram(time_diff_shifted, bins=100)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    p0 = [np.max(counts), 0, 0.1]
    popt, pcov = cf(gaus, bin_centers, counts, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    pmeas = [Measure(val, err) for val, err in zip(popt, perr)]

    p02 = [np.max(counts) * 0.9, 0, 0.1, np.max(counts) * 0.1, 0, 1]
    popt2, pcov2 = cf(double_gaus, bin_centers, counts, p0=p02)
    perr2 = np.sqrt(np.diag(pcov2))
    pmeas2 = [Measure(val, err) for val, err in zip(popt2, perr2)]

    time_plot = np.linspace(-2, 2, 1000)

    fig, ax = plt.subplots()
    ax.hist(time_diff_shifted, bins=100)
    ax.plot(time_plot, gaus(time_plot, *p0), color='gray', ls='--', alpha=0.6)
    ax.plot(time_plot, gaus(time_plot, *popt), color='green')
    ax.plot(time_plot, double_gaus(time_plot, *p02), color='gray', ls=':', alpha=0.6)
    ax.plot(time_plot, double_gaus(time_plot, *popt2), color='red')
    ax.set_xlabel('C1 - C2 Time Difference (ns)')
    ax.set_ylabel('Counts')
    ax.set_title('Filtered and Shifted Time Difference')
    ax.annotate(f'Width: {pmeas[2] * 1e3} ps', xy=(0.01, 0.99), xycoords='axes fraction', ha='left', va='top',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    gaus2_widths_str = f'Width1: {pmeas2[2] * 1e3} ps\nWidth2: {pmeas2[5] * 1e3} ps'
    ax.annotate(gaus2_widths_str, xy=(0.01, 0.92), xycoords='axes fraction', ha='left', va='top',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.tight_layout()

    plt.show()


def analyze_chunk(data):
    # start_event = 800 * 15
    # stop_event = 800 * 20
    print(f'Event range: {np.min(data["eventNo"])} - {np.max(data["eventNo"])}')
    # if np.max(data['eventNo']) < start_event or np.min(data['eventNo']) > stop_event:
    #     return None, None
    amplC1, amplC2, amplC3, amplC4 = data['amplC1'], data['amplC2'], data['amplC3'], data['amplC4']
    t0, dt = data['t0'], data['dt']

    waveform_fits_C1, waveform_fits_C2 = [], []
    for i in range(len(amplC1)):
        waveform_fit_dict_C1 = fit_waveform(amplC1[i], dt[i])
        waveform_fit_dict_C2 = fit_waveform(amplC2[i], dt[i])
        binary_num = decode_event_num(amplC3[i], dt[i], True)
        plt.show()
        if waveform_fit_dict_C1 is not None and waveform_fit_dict_C2 is not None:
            waveform_fits_C1.append(waveform_fit_dict_C1)
            waveform_fits_C2.append(waveform_fit_dict_C2)
        # waveforms = np.array([amplC1[i], amplC2[i]])
        # plot_waveform(waveforms, t0=t0[i], dt=dt[i])
        # plt.show()

    return waveform_fits_C1, waveform_fits_C2


def decode_event_num(wave, dt, plot=False):
    bit_size = 25  # ns width of a bit in the waveform
    dt = dt * 1e9  # Convert s to ns
    times = np.arange(len(wave)) * dt

    min_voltage = np.min(wave)
    max_voltage = np.max(wave)
    thresh_voltage = (min_voltage + max_voltage) / 2
    # Find first point below threshold
    start_index = np.where(wave < thresh_voltage)[0][0]
    filtered_wave = wave[start_index:]
    filtered_times = times[start_index:]

    # Break into steps
    step_size = int(bit_size / dt)
    steps = np.arange(0, len(filtered_wave), step_size)

    # Get the average of each step
    step_medians = [np.median(filtered_wave[step:step+step_size]) for step in steps]
    step_time_centers = [np.mean(filtered_times[step:step+step_size]) for step in steps]

    # Build the binary number
    binary_num = ''
    for step_median in step_medians:
        binary_num += '0' if step_median > thresh_voltage else '1'
    binary_num = binary_num[1:17]
    integer_number = int(binary_num, 2)
    print(f'Binary number: {binary_num}, integer: {integer_number}')

    if plot:
        fig, ax = plt.subplots()
        ax.plot(times, wave)
        # Plot the steps and the average of each step
        ax.plot(step_time_centers, step_medians, 'o', color='red')
        for step in steps:
            ax.axvline(filtered_times[step], color='red', ls='-', alpha=0.2)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title('Event Num Waveform')
        ax.annotate(f'Binary: {binary_num}\nInteger: {integer_number}', xy=(0.01, 0.1), xycoords='axes fraction',
                    ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        fig.tight_layout()

    return binary_num


def plot_waveform(waves, t0=None, dt=0.2):
    fig, ax = plt.subplots()
    frac_amp = 0.2
    extra_fit_points = 5
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, (wave, color) in enumerate(zip(waves, colors)):
        fit_points = 10 if i == 0 else 40
        times = np.arange(len(wave)) * dt
        # print(t0[i], dt)
        # if t0 is not None:
        #     times += t0[i]
        times *= 1e9  # Convert s to ns
        min_voltage = np.min(wave)
        min_v_time_point = np.argmin(wave)
        median_voltage = np.median(wave)
        thresh_stop_voltage = median_voltage + (min_voltage - median_voltage) * 0.1

        # Get the index of the first point to the left of the minimum that is less than the median voltage
        left_median = np.where(wave[:min_v_time_point] > thresh_stop_voltage)[0]
        left_median = left_median[-1] if len(left_median) > 0 else 0

        slope = (wave[min_v_time_point] - wave[left_median]) / (times[min_v_time_point] - times[left_median]) * 2
        print(f'slope: {slope}')
        # slope = a * c / 4 --> c = 4 * slope / a

        # times_fit = times[min_v_time_point-fit_points:min_v_time_point]
        # wave_fit = wave[min_v_time_point-fit_points:min_v_time_point]

        times_fit = times[left_median - extra_fit_points:min_v_time_point + 1]
        wave_fit = wave[left_median - extra_fit_points:min_v_time_point + 1]

        times_plot = times[left_median - extra_fit_points * 2:min_v_time_point + extra_fit_points * 2]
        wave_plot = wave[left_median - extra_fit_points * 2:min_v_time_point + extra_fit_points * 2]

        ax.scatter(times_plot, wave_plot, marker='.', color=color, label=f'C{i+1}')

        p0 = [min_voltage, median_voltage, 4 * slope / min_voltage, (times[left_median] + 2 * times[min_v_time_point]) / 3]
        try:
            popt, pcov = cf(sigmoid, times_fit, wave_fit, p0=p0)
            print(f'p0: {p0}')
            print(f'popt: {popt}')
            frac_time = get_sigmoid_frac_x(*popt, fraction=frac_amp)
            frac_voltage = sigmoid(frac_time, *popt)

            times_fit_plot = np.linspace(times_fit[0], times_fit[-1], 1000)
            ax.plot(times_fit_plot, sigmoid(times_fit_plot, *p0), color='r', alpha=0.4, ls='--')
            ax.plot(times_fit_plot, sigmoid(times_fit_plot, *popt), color='r', ls='-')
            ax.axvline(frac_time, color=color, ls='--')
            ax.axhline(frac_voltage, color=color, ls='--')
        except:
            print(f'Fit failed for C{i+1}.')
        # ax.axvline(frac_time, color=color, ls='-')
        # ax.axhline(frac_voltage, color=color, ls='-')

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Voltage (mV)')
    ax.legend()
    fig.tight_layout()


def fit_waveform(wave, dt, frac_amp=0.2, extra_fit_points=5):
    times = np.arange(len(wave)) * dt
    times *= 1e9  # Convert s to ns
    min_voltage = np.min(wave)
    min_v_time_point = np.argmin(wave)
    median_voltage = np.median(wave)
    thresh_stop_voltage = median_voltage + (min_voltage - median_voltage) * 0.1

    # Get the index of the first point to the left of the minimum that is less than the median voltage
    left_median = np.where(wave[:min_v_time_point] > thresh_stop_voltage)[0]
    left_median = left_median[-1] if len(left_median) > 0 else 0

    slope = (wave[min_v_time_point] - wave[left_median]) / (times[min_v_time_point] - times[left_median]) * 2

    times_fit = times[left_median - extra_fit_points:min_v_time_point + 1]
    wave_fit = wave[left_median - extra_fit_points:min_v_time_point + 1]

    p0 = [min_voltage, median_voltage, 4 * slope / min_voltage,
          (times[left_median] + 2 * times[min_v_time_point]) / 3]
    try:
        popt, pcov = cf(sigmoid, times_fit, wave_fit, p0=p0)
        frac_time = get_sigmoid_frac_x(*popt, fraction=frac_amp)
        frac_voltage = sigmoid(frac_time, *popt)
        sigmoid_amp = get_sigmoid_max(*popt)
        baseline = popt[1]
        time_offset = popt[3]
        sigmoid_slope = popt[2]
        fit_dict = {
            'frac_time': frac_time,
            'frac_voltage': frac_voltage,
            'sigmoid_amp': sigmoid_amp,
            'baseline': baseline,
            'time_offset': time_offset,
            'sigmoid_slope': sigmoid_slope
        }
        return fit_dict
    except:
        return None


def sigmoid_plot_test():
    fig, ax = plt.subplots()
    x = np.linspace(180, 182, 1000)
    y1 = sigmoid(x, -1, 0, 100, 181)
    y2 = sigmoid(x, -1, 0, 10, 181)

    ax.plot(x, y1, label='c=1')
    ax.plot(x, y2, label='c=0.1')
    ax.axhline(0, color='k', ls='-')
    ax.legend()
    plt.show()


def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-c * (x - d))) + b


def get_sigmoid_frac_x(a, b, c, d, fraction=0.5):
    return -np.log(a / (fraction * (a - b)) - 1) / c + d


def get_sigmoid_max(a, b, c, d):
    return a + b


def gaus(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))


def double_gaus(x, a1, b1, c1, a2, b2, c2):
    return gaus(x, a1, b1, c1) + gaus(x, a2, b2, c2)


if __name__ == '__main__':
    main()
