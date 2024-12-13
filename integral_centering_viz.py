#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 13 12:34 PM 2024
Created in PyCharm
Created as picosec/integral_centering_viz.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Parameters
    x = np.linspace(0, 400, 1000)  # x in picoseconds
    signal = -np.exp(-0.5 * ((x - 200) / 5) ** 2)  # Sharp negative signal at 200ps

    # Integrate for different window sizes
    window_sizes = [20, 50, 100, 200]
    results_end = []
    results_center = []

    for window_size in window_sizes:
        positions_end, integrated_signal_end = moving_integral(signal, window_size, x, center_window=False)
        positions_center, integrated_signal_center = moving_integral(signal, window_size, x, center_window=True)
        results_end.append((positions_end, integrated_signal_end))
        results_center.append((positions_center, integrated_signal_center))

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot with x at the end of the window
    axes[0].plot(x, signal, label="Original Signal", color="black", linestyle="--")
    for idx, (positions, integrated_signal) in enumerate(results_end):
        axes[0].plot(positions, integrated_signal, label=f"n = {window_sizes[idx]}")
    axes[0].set_title("Integral with x at End of Window")
    axes[0].set_ylabel("Integrated Signal")
    axes[0].legend()

    # Plot with x at the center of the window
    axes[1].plot(x, signal, label="Original Signal", color="black", linestyle="--")
    for idx, (positions, integrated_signal) in enumerate(results_center):
        axes[1].plot(positions, integrated_signal, label=f"n = {window_sizes[idx]}")
    axes[1].set_title("Integral with x at Center of Window")
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel("Integrated Signal")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print('donzo')

# Function to compute moving integral
def moving_integral(signal, window_size, x_values, center_window=False):
    half_window = window_size // 2
    integrated_signal = []
    positions = []
    for i in range(len(signal) - window_size + 1):
        integral = np.sum(signal[i:i + window_size])
        integrated_signal.append(integral)
        if center_window:
            positions.append(x_values[i + half_window])  # Center of window
        else:
            positions.append(x_values[i + window_size - 1])  # End of window
    return np.array(positions), np.array(integrated_signal)




if __name__ == '__main__':
    main()
