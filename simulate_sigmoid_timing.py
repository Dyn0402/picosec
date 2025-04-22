#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 22 17:59 2025
Created in PyCharm
Created as picosec/simulate_sigmoid_timing

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
import uproot

from scipy.interpolate import CubicSpline
from get_timing_test import plot_waveform, get_timing


def main():
    file_path = '/local/home/dn277127/Bureau/picosec/Run299-Pool2_TESTBEAM_tree.root'
    tree_name = 'RawDataTree'
    branches = ['amplC1', 'amplC2']
    dt = 0.1  # ns
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        print(tree.keys())

        # Read the data into a numpy array
        data = tree.arrays(branches, library='np', entry_start=0, entry_stop=1)
        amplC1, amplC2 = data['amplC1'], data['amplC2']

    interp_c1 = create_cubic_spline(np.arange(len(amplC1[0])) * dt, amplC1[0])
    interp_c2 = create_cubic_spline(np.arange(len(amplC2[0])) * dt, amplC2[0])

    xs_plt = np.linspace(0, len(amplC1[0])*dt, 10 * len(amplC1[0]))

    ax_c1 = plot_waveform(amplC1[0], dt)
    plot_waveform(interp_c1(xs_plt), dt / 10, ax_in=ax_c1)

    ax_c2 = plot_waveform(amplC2[0], dt)
    plot_waveform(interp_c2(xs_plt), dt / 10, ax_in=ax_c2)

    # Sample c1 and c2 cubic splines with random shifts in x0 and calculate timing and time differences
    n_waveforms = 2000
    time_diffs = []
    dt_test = 0.5
    ts = np.arange(len(amplC1[0])) * dt_test
    for i in range(n_waveforms):
        print(f'Event {i}/{n_waveforms}')
        ts_i = ts + np.random.uniform(-2.5, 2.5)

        t_sigmoid_c1 = get_timing(interp_c1(ts_i), dt_test, npt_fit=10)
        t_sigmoid_c2 = get_timing(interp_c2(ts_i), dt_test, npt_fit=20)
        if t_sigmoid_c1 is None or t_sigmoid_c2 is None:
            continue

        time_diffs.append(t_sigmoid_c2 - t_sigmoid_c1)

    fig, ax = plt.subplots()
    # binning = np.linspace(-5, 5, 200)
    ax.hist(time_diffs, bins=20, histtype='step', color='blue', label='Timing Difference')
    ax.set_title('Timing Difference Histogram')
    ax.set_xlabel('Timing Difference (ns)')
    ax.set_ylabel('Counts')


    plt.show()


    print('donzo')


def create_cubic_spline(x_values, y_values):
    """
    Create a cubic spline interpolation from x and y values.

    Parameters:
    - x_values (array-like): The x coordinates of the data points.
    - y_values (array-like): The y coordinates of the data points.

    Returns:
    - CubicSpline object
    """
    return CubicSpline(x_values, y_values)


def find_extrema_in_range(spline, x_min, x_max, num_subdivisions=100):
    """
    Find the minimum and maximum of a spline in the given range.

    Parameters:
    - spline (CubicSpline): The spline function.
    - x_min (float): Start of the interval.
    - x_max (float): End of the interval.
    - num_subdivisions (int): Number of sub-intervals for root finding.

    Returns:
    - (x_min_val, min_val, x_max_val, max_val): Tuple of x and y values for min and max
    """
    # First derivative of the spline
    spline_derivative = spline.derivative()

    # Find critical points by root-finding the derivative in small intervals
    x_candidates = [x_min, x_max]
    x_samples = np.linspace(x_min, x_max, num_subdivisions)

    for i in range(len(x_samples) - 1):
        a, b = x_samples[i], x_samples[i + 1]
        if spline_derivative(a) * spline_derivative(b) < 0:  # sign change → root
            try:
                root = scipy.optimize.brentq(spline_derivative, a, b)
                x_candidates.append(root)
            except ValueError:
                continue

    # Evaluate spline at all candidate points
    y_candidates = [spline(x) for x in x_candidates]

    # Find min and max
    min_idx = np.argmin(y_candidates)
    max_idx = np.argmax(y_candidates)

    return (x_candidates[min_idx], y_candidates[min_idx],
            x_candidates[max_idx], y_candidates[max_idx])


if __name__ == '__main__':
    main()
