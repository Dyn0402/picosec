#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 31 2:38 PM 2025
Created in PyCharm
Created as picosec/sigmoid_timing_poc.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    a = -5
    k = 0.1
    x0 = 200
    c = -2
    # c = a * 0.24

    t_frac = 0.9

    y_max = a + c - 1.5
    y_frac_alex = (y_max - c) * t_frac + c
    t_trad = get_timing_trad(f=t_frac, a=a, k=k, x0=x0, c=c)
    t_sigmoid = get_timing_sigmoid(f=t_frac, k=k, x0=x0)
    t_sigmoid_alex = get_timing_sigmoid_alex(y_frac_alex, a, k, x0, c)

    fig, ax = plt.subplots()
    x = np.linspace(0, 400, 1000)
    y = sigmoid_offset(x, a, k, x0, c)
    ax.plot(x, y, label='Sigmoid Offset')
    ax.set_title('Sigmoid Offset Function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axhline(y=0, color='gray', alpha=0.3, zorder=0)
    ax.axhline(y=y_max, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=c, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=t_trad, color='g', linestyle='--', alpha=0.5, label='Traditional Timing')
    ax.axvline(x=t_sigmoid, color='b', linestyle='--', alpha=0.5, label='Sigmoid Timing')
    ax.axvline(x=t_sigmoid_alex, color='orange', linestyle='--', alpha=0.5, label='Sigmoid Alex Timing')
    ax.legend()
    fig.tight_layout()

    plt.show()

    print('donzo')



def sigmoid_offset(x, a, k, x0, c):
    """ Sigmoid function with offset. """
    return a / (1 + np.exp(-k * (x - x0))) + c


def get_timing_trad(f, a, k, x0, c):
    """ Get timing as fraction of the maximum value of the sigmoid. """
    y_max = a + c
    y_frac = f * y_max
    t = x0 - (1 / k) * np.log((a / (y_frac - c)) - 1)
    return t


def get_timing_sigmoid(f, k, x0):
    """ Get timing as fraction of the maximum value of the sigmoid. """
    t = x0 - (1 / k) * np.log(1 / f - 1)
    return t


def get_timing_sigmoid_alex(y, a, k, x0, c):
    """ Get timing as fraction of the maximum value of the sigmoid. """
    t = x0 - (1 / k) * np.log(a / (y - c) - 1)
    return t


if __name__ == '__main__':
    main()
