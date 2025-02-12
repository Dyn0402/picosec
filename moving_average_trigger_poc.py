#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 11 10:26 AM 2025
Created in PyCharm
Created as picosec/moving_average_trigger_poc.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit as cf
from scipy.stats import norm

from Measure import Measure


def main():
    # plot_square_wave_moving_averages()
    # plot_mv_avg_gaus_noise_threshold_scaling()
    # gaus_noise_std_vs_mv_avg_points()
    plot_false_positive_vs_mv_avg_points()
    # fit_thresholds_vs_mv_avg_points()
    # generate_gaus_noise()
    print('donzo')


def generate_gaus_noise():
    # General parameters
    n_points = 1000
    x = np.linspace(0, 400, n_points)  # x in picoseconds
    moving_average_points = [10, 5, 1]

    # Baseline parameters
    baseline = 0
    sigma_noise = 0.1
    y_noise = baseline + np.random.normal(0, sigma_noise, n_points)

    # Signal parameters
    # signal_func = gaus_signal
    # signal_params = {'amp': -1, 'mu': 200, 'sigma': 5}

    signal_func = zero_signal
    signal_params = {}

    y_signal = signal_func(x, *signal_params.values())

    y = y_noise + y_signal

    for points in moving_average_points:
        x_avg, y_avg = moving_average(x, y, points)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(x_avg, y_avg, color="black", label=f"Moving Average {points} Points")
        ax.set_title("Moving Average Trigger")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Signal")
        ax.legend()
        fig.tight_layout()

    plt.show()


def plot_square_wave_moving_averages():
    """
    Make a plot showing a simple square wave signal and various moving averages.
    Demonstrate that when the moving average window is smaller than the signal period, the maximum of the moving average
    is the same as the maximum of the signal. When the window is larger than the signal period, the maximum of the
    moving average is strictly smaller than the square wave.
    :return:
    """
    mv_avg_ns = [1, 5, 10, 20, 30, 50]
    amp = 1
    width = 20
    center = 50
    n_points = 100
    x = np.linspace(0, 100, n_points)

    y = square_wave_signal(x, amp, width, center)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(x, y, color='black', label='Signal')
    for mv_avg_n in mv_avg_ns:
        x_avg, y_avg = moving_average_numpy(x, y, mv_avg_n)
        print(f'Moving Average {mv_avg_n}\nx: {x_avg}\ny: {y_avg}')
        ax.plot(x_avg, y_avg, label=f'Moving Average {mv_avg_n}', linewidth=4)
    ax.set_title('Square Wave Moving Averages')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Signal')
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()

    plt.show()


def plot_mv_avg_gaus_noise_threshold_scaling():
    """
    Generate normally distributed noise data and set a threshold at 3 sigma. Then take n-point moving averages of the
    noise data and scale the threshold by 1/sqrt(n) to maintain the same rejection rate. Plot the data and thresholds
    to illustrate the scaling.
    :return:
    """
    n_points = 1000
    x = np.linspace(0, 400, n_points)  # x in picoseconds
    sigma_threshold = -3
    moving_average_points = [1, 5, 10, 20, 50]

    # Get gaus cdf for threshold

    # Baseline parameters
    baseline = 0
    sigma_noise = 0.1
    y_noise = baseline + np.random.normal(0, sigma_noise, n_points)

    fig, axs = plt.subplots(len(moving_average_points), 1, figsize=(8, 8), sharex='all', sharey='all')
    for n_mv_avg, ax in zip(moving_average_points, axs):
        x_avg, y_avg = moving_average(x, y_noise, n_mv_avg)
        ax.scatter(x_avg, y_avg, color="black", label=f'{n_mv_avg}-Point Moving Average', s=1)
        scaled_threshold = sigma_threshold * sigma_noise / np.sqrt(n_mv_avg)
        if n_mv_avg == 1:
            ax.axhline(sigma_threshold * sigma_noise, color='red', label=f"{sigma_threshold}σ for 1 Point")
        else:
            ax.axhline(sigma_threshold * sigma_noise, color='red', alpha=0.3)
            ax.axhline(scaled_threshold, color='red', label=f"{sigma_threshold}σ for {n_mv_avg} Point", linestyle='--')
        ax.axhline(0, color='black', linewidth=1)
        ax.legend(loc='upper right')
        # Count the number of points below threshold and print number and fraction
        n_below = len(y_avg[y_avg < scaled_threshold])
        fraction_below = n_below / len(y_avg)
        print(f'{n_mv_avg}-Point Moving Average\n'
              f'Points Below Threshold: {n_below}\n'
              f'Fraction Below Threshold: {fraction_below}')
    axs[-1].set_xlabel("Time (ps)")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)

    plt.show()


def gaus_noise_std_vs_mv_avg_points():
    """
    Generate normally distributed noise data. Take n-point moving averages of the noise data and calculate the standard
    deviation of the moving average. Plot the standard deviation of the moving average vs the number of points in the
    moving average.
    :return:
    """
    n_points = 1000
    n_waveforms = 1000
    x = np.linspace(0, 400, n_points)  # x in picoseconds
    moving_average_points = np.arange(1, 101, 1)

    # Baseline parameters
    baseline = 0
    sigma_noise = 0.1

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Generate noise waveforms
    y_noises = np.random.normal(0, sigma_noise, (n_waveforms, n_points))

    i = 0
    for y_noise in y_noises:
        print(f'Waveform {i}')
        i += 1
        stds = []
        for n_mv_avg in moving_average_points:
            x_avg, y_avg = moving_average_numpy(x, y_noise, n_mv_avg)
            stds.append(np.std(y_avg))

        ax.plot(moving_average_points, stds, color='black', alpha=0.3)

    # Calculate and plot a 1/sqrt(n) scaling line
    x_plt = np.linspace(min(moving_average_points), max(moving_average_points), 1000)
    y_sqrt_n = sigma_noise / np.sqrt(x_plt)

    ax.plot(x_plt, y_sqrt_n, color='red', label='1/sqrt(n)', linestyle='--')
    ax.set_title('Standard Deviation of Moving Average vs Number of Points in Moving Average')
    ax.set_xlabel('Number of Points in Moving Average')
    ax.set_ylabel('Standard Deviation')
    ax.legend()
    fig.tight_layout()

    plt.show()


def plot_false_positive_vs_mv_avg_points():
    """
    Generate m waveforms of normally distributed noise data. Take n-point moving averages of the noise data and
    decide if any point is below a set threshold. Plot the number of false positives vs the number of points in the
    moving average.
    :return:
    """
    n_points = 10000
    n_waveforms = 1000
    x = np.linspace(0, 1000, n_points)  # x in picoseconds
    moving_average_points = np.arange(1, 2000, 20)
    rejection_rate = 0.999
    single_point_rate = 1 - rejection_rate ** (1 / n_points)
    print(f'Single Point Rejection Rate: {1 - single_point_rate}')
    single_point_sigmas = norm.ppf(single_point_rate)

    # sigma_threshold = -3
    sigma_threshold = single_point_sigmas
    print(f'Sigma Threshold: {sigma_threshold}')

    # Baseline parameters
    baseline = 0
    sigma_noise = 0.2

    # Generate noise waveforms
    y_noises = np.random.normal(0, sigma_noise, (n_waveforms, n_points))

    # Read correction factor from file
    df = pd.read_csv('correction_factor.csv')
    correction_factor_dict = {row['n_points']: row['correction_factor'] for index, row in df.iterrows()}
    for n_mv_avg in moving_average_points:
        print(f'Correction Factor for {n_mv_avg} Points: {correction_factor_dict[n_mv_avg]}')
    input()

    i = 0
    false_positives = np.zeros(len(moving_average_points))
    waveform_mins = {n_mv_avg: [] for n_mv_avg in moving_average_points}
    for y_noise in y_noises:
        print(f'Waveform {i}')
        i += 1
        for j, n_mv_avg in enumerate(moving_average_points):
            x_avg, y_avg = moving_average_numpy(x, y_noise, n_mv_avg)
            threshold_scaled = sigma_threshold * sigma_noise / (n_mv_avg ** 0.5) * correction_factor_dict[n_mv_avg]
            # threshold_scaled = sigma_threshold * sigma_noise / (n_mv_avg ** 0.5)
            waveform_mins[n_mv_avg].append(np.min(y_avg))
            n_below = len(y_avg[y_avg < threshold_scaled])
            if n_below > 0:
                false_positives[j] += 1

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(moving_average_points, false_positives, color='black')
    # ax.plot(moving_average_points, exponential(moving_average_points, *popt), color='red', linestyle='--')
    ax.set_title('False Positives vs Number of Points in Moving Average')
    ax.set_xlabel('Number of Points in Moving Average')
    ax.set_ylabel('False Positives')
    fig.tight_layout()

    false_positives_n1 = false_positives[0]
    # With waveform mins, for each n_mov_avg use the distribution to determine the threshold necessary to get the same
    # number of false positives as the 1 point moving average.
    thresholds = []
    for n_mv_avg in moving_average_points:
        threshold = np.percentile(waveform_mins[n_mv_avg], 100 * false_positives_n1 / n_waveforms)
        print(f'For {n_mv_avg} points, threshold: {threshold}')
        thresholds.append(threshold)

    fig_thresholds, ax_thresholds = plt.subplots(1, 1, figsize=(8, 8))
    ax_thresholds.plot(moving_average_points, thresholds, color='black')
    ax_thresholds.plot(moving_average_points, sigma_threshold * sigma_noise / (moving_average_points ** 0.5), color='red', linestyle='--')
    ax_thresholds.set_title('Threshold vs Number of Points in Moving Average')
    ax_thresholds.set_xlabel('Number of Points in Moving Average')
    ax_thresholds.set_ylabel('Threshold')
    fig_thresholds.tight_layout()

    # Plot the ratio of thresholds to the scaled 1 point threshold
    ratio = np.array(thresholds) / (sigma_threshold * sigma_noise / (moving_average_points ** 0.5))
    fig_ratio, ax_ratio = plt.subplots(1, 1, figsize=(8, 8))
    ax_ratio.plot(moving_average_points, ratio, color='black')
    ax_ratio.set_title('Threshold Ratio vs Number of Points in Moving Average')
    ax_ratio.set_xlabel('Number of Points in Moving Average')
    ax_ratio.set_ylabel('Threshold Ratio')
    fig_ratio.tight_layout()


    # # Write false positive fraction and number of moving average points to file
    # df = pd.DataFrame({'n_points': moving_average_points, 'correction_factor': ratio})
    # df.to_csv('correction_factor.csv', index=False)
    #
    # # Write thresholds to file
    # df = pd.DataFrame({'n_points': moving_average_points, 'threshold': thresholds})
    # df.to_csv('thresholds.csv', index=False)

    plt.show()


def fit_thresholds_vs_mv_avg_points():
    """
    Try to fit the thresholds vs moving average points with some function.
    :return: 
    """
    df = pd.read_csv('thresholds.csv')
    x = df['n_points']
    y = df['threshold']

    threshold_n1 = y[0]

    # p0 = [threshold_n1, 0.5, 0.01]
    # popt, pcov = cf(exponential, x, y, p0=p0)

    p0_power = [threshold_n1, -0.5, 0]
    popt_power, pcov_power = cf(power_law, x, y, p0=p0_power)
    perr = np.sqrt(np.diag(pcov_power))
    meases = [Measure(val, err) for val, err in zip(popt_power, perr)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(x, y, color='black')
    ax.plot(x, threshold_n1 / (x ** 0.5), color='blue', linestyle='--', label='1/sqrt(n)')
    # ax.plot(x, exponential(x, *p0), color='red', linestyle='-', label='Exponential Guess', alpha=0.4)
    # ax.plot(x, exponential(x, *popt), color='red', linestyle='-', label='Exponential Fit')
    ax.plot(x, power_law(x, *p0_power), color='green', linestyle='--', label='Power Law Guess', alpha=0.4)
    ax.plot(x, power_law(x, *popt_power), color='green', linestyle='--', label='Power Law Fit')
    ax.set_title('Threshold vs Number of Points in Moving Average')
    ax.set_xlabel('Number of Points in Moving Average')
    ax.set_ylabel('Threshold')
    # Write out equation in annotation and then another one to list parameter values
    # Write out equation in latex
    latex_str = rf'$\text{{Threshold}}(n) = {meases[0]} \cdot n^{{{meases[1]}}} + {meases[2]}$'
    # latex_str = rf'$\text{{Threshold}} = {meases[0]} \cdot n^{{{meases[1]}}}$'
    ax.annotate(latex_str, va='bottom', ha='left',
                xy=(0.1, 0.1), xycoords='axes fraction', fontsize=12, bbox=dict(facecolor='salmon', alpha=0.5))
    ax.legend()
    fig.tight_layout()

    plt.show()


def exponential(x, a, b, c):
    return a * np.exp(-b * x) + c


def power_law(x, a, b, c):
    return a * x ** b + c


def moving_average(x, y, points):
    x_avg = moving_average_1d(x, points)
    y_avg = moving_average_1d(y, points)
    return x_avg, y_avg


def moving_average_1d(data, window_size):
    avg = []
    for i in range(len(data) - window_size + 1):
        avg.append(np.mean(data[i:i + window_size]))
    return np.array(avg)


def moving_average_numpy(x, y, n):
    """
    Calculate the integral of a waveform in both x and y with numpy
    :param x: time points
    :param y: voltage points
    :param n: number of points to integrate
    :return: x, y of the integral waveform
    """
    x_int = np.convolve(x, np.ones(n), 'valid') / n
    y_int = np.convolve(y, np.ones(n), 'valid') / n
    return x_int, y_int


def gaus_signal(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def zero_signal(x):
    return np.zeros(len(x))


def square_wave_signal(x, amp, width, center):
    """
    Single square wave with given amplitude, width, and center.
    :param x:
    :param amp:
    :param width:
    :param center:
    :return:
    """
    y = np.zeros(len(x))
    y[(x > center - width / 2) & (x < center + width / 2)] = amp
    return y


if __name__ == '__main__':
    main()
