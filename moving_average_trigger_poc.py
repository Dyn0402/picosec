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


def main():
    # plot_square_wave_moving_averages()
    plot_mv_avg_gaus_noise_threshold_scaling()
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
        x_avg, y_avg = moving_average(x, y, mv_avg_n)
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

    # Baseline parameters
    baseline = 0
    sigma_noise = 0.1
    y_noise = baseline + np.random.normal(0, sigma_noise, n_points)

    fig, axs = plt.subplots(len(moving_average_points), 1, figsize=(8, 8), sharex='all', sharey='all')
    for n_mv_avg, ax in zip(moving_average_points, axs):
        x_avg, y_avg = moving_average(x, y_noise, n_mv_avg)
        ax.scatter(x_avg, y_avg, color="black", label=f'{n_mv_avg}-Point Moving Average', s=1)
        if n_mv_avg == 1:
            ax.axhline(sigma_threshold * sigma_noise, color='red', label=f"{sigma_threshold}σ for 1 Point")
        else:
            ax.axhline(sigma_threshold * sigma_noise, color='red', alpha=0.3)
            ax.axhline(sigma_threshold * sigma_noise / np.sqrt(n_mv_avg), color='red', label=f"{sigma_threshold}σ for {n_mv_avg} Point", linestyle='--')
        ax.axhline(0, color='black', linewidth=1)
        ax.legend(loc='upper right')
    axs[-1].set_xlabel("Time (ps)")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)

    plt.show()


def moving_average(x, y, points):
    x_avg = moving_average_1d(x, points)
    y_avg = moving_average_1d(y, points)
    return x_avg, y_avg


def moving_average_1d(data, window_size):
    avg = []
    for i in range(len(data) - window_size + 1):
        avg.append(np.mean(data[i:i + window_size]))
    return np.array(avg)


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
