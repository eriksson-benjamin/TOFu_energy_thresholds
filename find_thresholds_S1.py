# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 08:37:04 2022

@author: bener807
"""

'''
Calculates the energy thresholds for S1 detectors.

Note
----
Data is stored locally on PC and on backup of PC. Na-22 data is used for the S1
detectors.
'''

import sys
try:
    sys.path.insert(0, 'C:/python/useful_definitions/')
except Exception:
    pass
import useful_defs as udfs
try:
    sys.path.insert(
        0, '/home/beriksso/TOFu/analysis/benjamin/github/TOFu/functions/')
except Exception:
    sys.path.insert(0, 'C:/python/TOFu/functions/')
import tofu_functions as dfs
import numpy as np
import matplotlib.pyplot as plt
udfs.set_nes_plot_style()
import scipy


def read_data(detector):
    """Read binned Na-22 data from energy calibration."""
    p = udfs.unpickle(f'data/Na-22/light_yield/binned/{detector}.pickle')

    return p['bin centres'], p['counts']


def model(a, b, l, k, x0, x):
    """Logistic function multiplied by exponential."""
    L = l / (1 + np.exp(-k**3 * (x - x0)))
    e = a**5 * np.exp(-b**5 * x)

    offset = 0
    return (e + offset) * L


def plot_model(parameters, x, y, yerr, detector):
    """Plot model on top of data points."""
    plt.figure(detector)

    # Plot data points
    plt.plot(x, y, 'k.', markersize=2)
    plt.errorbar(x, y, yerr=yerr, linestyle='None', color='k')

    # Plot model
    plt.plot(x, model(*parameters, x), 'r--')

    plt.xlabel('$E_{ee}$ (MeV$_{ee}$)')
    plt.ylabel('counts (a.u.)')
    plt.title(detector.replace('_', '-'), loc='left')


def fit_function(parameters, x, y, detector):
    """Minimize chi^2 for given fitting region."""
    fit_range = get_fit_range(detector)
    bool_range = (x >= fit_range[0]) & (x <= fit_range[1])
    y_model = model(*parameters, x)[bool_range]
    y_data = y[bool_range]

    # Calculate chi^2
    dof = len(y_data) - len(parameters)
    # dof = 1
    chi2 = np.sum((y_data - y_model)**2 / y_model) / dof

    return chi2


def get_fit_range(detector):
    """Return detector specific fit ranges."""
    lines = np.loadtxt('data/fit_ranges.txt', dtype='str')
    ranges = {l[0]: [float(l[1]), float(l[2])] for l in lines}

    return ranges[detector]


def get_bin_edges(detector):
    """Return detector specific binning."""
    lines = np.loadtxt('data/bin_edges.txt', dtype='str')
    edges = {l[0]: [float(l[1]), float(l[2]), float(l[3])] for l in lines}
    e = edges[detector]

    return np.arange(e[0], e[1], e[2])


def plot_threshold(detector, threshold):
    """Plot thresholds."""
    plt.figure(detector)
    plt.axvline(threshold, color='k', linestyle='--')
    plt.title(f'$E_{{thr}}$ = {threshold:.3f} MeV$_{{ee}}$', loc='right')


def starting_guesses(detector):
    """Return starting guesses for given detector."""
    p = np.loadtxt('data/starting_guesses.txt', dtype='str')
    arg = np.where(p[:, 0] == detector)[0]
    a = float(p[arg, 1])
    b = float(p[arg, 2])
    l = float(p[arg, 3])
    k = float(p[arg, 4])
    x0 = float(p[arg, 5])
    return a, b, l, k, x0,


def main(detector):
    """Perform fit, plot, and save to file."""
    print(detector)
    print('-----')

    # Read the binned data and normalize
    bin_centres, events = read_data(detector)
    u_events = np.sqrt(events) / events.max()
    events = events / events.max()

    # Plot initial guess
    plot_model(starting_guesses(detector), x=bin_centres,
               y=events, yerr=u_events, detector=f'Init. guess {detector}')

    # Minimize test statistic with given bounds
    bnds = ((0, None), (0, None), (0, 1), (0, None), (0, 0.2))
    parameters = starting_guesses(detector)
    popt = scipy.optimize.minimize(fit_function, parameters, bounds=bnds,
                                   args=(bin_centres, events, detector))
    print(f'a = {popt.x[0]}')
    print(f'b = {popt.x[1]}')
    print(f'l = {popt.x[2]}')
    print(f'k = {popt.x[3]}')
    print(f'x0 = {popt.x[4]}')
    print('')
    plot_model(popt.x, bin_centres, events, u_events, detector)

    # Plot thresholds
    plot_threshold(detector, popt.x[4])

    # Save thresholds to file
    with open('thresholds_S1.txt', 'a') as handle:
        handle.write(f'{detector} {popt.x[4]:.4f}\n')

    # Save fit parameters to file
    with open(f'{detector}.txt', 'w') as handle:
        handle.write(f'a  {popt.x[0]}\n')
        handle.write(f'b  {popt.x[1]}\n')
        handle.write(f'l  {popt.x[2]}\n')
        handle.write(f'k  {popt.x[3]}\n')
        handle.write(f'x0 {popt.x[4]}\n')

    return popt


def print_popt(popt):
    """Print fit parameters."""
    print(f'{popt.x[0]:.4f} {popt.x[1]:.4f} {popt.x[2]:.4f} {popt.x[3]:.4f} '
          f'{popt.x[4]:.4f}')


if __name__ == '__main__':
    # Store thresholds in txt file
    with open('thresholds_S1.txt', 'w') as handle:
        handle.write('# Detector Threshold (MeVee)\n')

    detectors = ['S1_01', 'S1_02', 'S1_03', 'S1_04', 'S1_05']
    for detector in detectors:
        popt = main(detector)
        print_popt(popt)
