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

    print('donzo')


def moving_average(x, y, points):
    x_avg, y_avg = np.zeros(len(x)), np.zeros(len(y))
    for i in range(len(y)):
        x_avg[i] = np.mean(x[max(0, i - points):min(len(x), i + points)])
        y_avg[i] = np.mean(y[max(0, i - points):min(len(y), i + points)])
    return x_avg, y_avg


def gaus_signal(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def zero_signal(x):
    return np.zeros(len(x))


if __name__ == '__main__':
    main()
