# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 08:54:32 2022

@author: bener807
"""

'''
Calculates the energy thresholds for S1/S2 detectors.

Note
----
Data is stored locally on PC and on backup of PC. Neutron data with coincidence
requirement is used for S2 data.
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


def merge_data(shot_numbers, directory):
    """Merge data from shots to single array."""
    energies = dfs.get_dictionaries('merged')
    for shot_number in shot_numbers:
        # Import energies
        f_name = f'data/coincident_energies/{directory}/{shot_number}.pickle'
        erg = udfs.unpickle(f_name)

        for key, item in erg.items():
            energies[key] = np.append(energies[key], item)

    return energies


def model(a, b, l, k, x0, x):
    """Logistic function multiplied by exponential."""
    L = l / (1 + np.exp(-k**3 * (x - x0)))
    e = a**5 * np.exp(-b**5 * x)

    return e * L


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
    plt.ylim(0, 1.5)
    plt.xlim(0, 0.4)


def fit_function(parameters, x, y, detector, directory):
    """Minimize chi^2 for given fitting region."""
    fit_range = get_fit_range(detector, directory)
    bool_range = (x >= fit_range[0]) & (x <= fit_range[1])
    y_model = model(*parameters, x)[bool_range]
    y_data = y[bool_range]

    # Calculate chi^2
    dof = len(y_data) - len(parameters)
    chi2 = np.sum((y_data - y_model)**2 / y_model) / dof
    return chi2


def get_fit_range(detector, directory):
    """Return detector specific fit ranges."""
    lines = np.loadtxt(f'data/starting_guesses/{directory}/fit_ranges.txt',
                       dtype='str')
    ranges = {l[0]: [float(l[1]), float(l[2])] for l in lines}

    return ranges[detector]


def get_bin_edges(detector, directory):
    """Return detector specific binning."""
    lines = np.loadtxt(f'data/starting_guesses/{directory}/bin_edges.txt',
                       dtype='str')
    edges = {l[0]: [float(l[1]), float(l[2]), float(l[3])] for l in lines}
    e = edges[detector]

    return np.arange(e[0], e[1], e[2])


def plot_threshold(detector, threshold):
    """Plot thresholds."""
    plt.figure(detector)
    plt.axvline(threshold, color='k', linestyle='--')
    plt.title(f'$E_{{thr}}$ = {threshold:.3f} MeV$_{{ee}}$', loc='right')


def starting_guesses(detector, directory):
    """Return starting guesses for given detector."""
    p = np.loadtxt(f'data/starting_guesses/{directory}/starting_guesses.txt',
                   dtype='str')
    arg = np.where(p[:, 0] == detector)[0]
    a = float(p[arg, 1])
    b = float(p[arg, 2])
    l = float(p[arg, 3])
    k = float(p[arg, 4])
    x0 = float(p[arg, 5])

    return a, b, l, k, x0


def print_popt(popt):
    """Print fit parameters."""
    print((f'{popt.x[0]:.4f} {popt.x[1]:.4f} '
           f'{popt.x[2]:.4f} {popt.x[3]:.4f} {popt.x[4]:.4f}'))


def main(shot_numbers, detectors, directory):
    """Perform fit, plot, and save to file."""
    # Store thresholds in txt file
    with open('thresholds_S2.txt', 'w') as handle:
        handle.write('# Detector Threshold (MeVee)\n')

    # Merge all data for given shot numbers
    energies = merge_data(shot_numbers, directory)

    for detector in detectors:
        print(detector)
        print('-----')
        bin_edges = get_bin_edges(detector, directory)
        bin_centres = bin_edges[1:] - np.diff(bin_edges) / 2

        # Histogram the energy spectrum and normalize
        events, _ = np.histogram(energies[detector], bin_edges)
        u_events = np.sqrt(events) / events.max()
        events = events / events.max()

        # Plot initial guess
        plot_model(starting_guesses(detector, directory), x=bin_centres,
                   y=events, yerr=u_events, detector=f'Init. guess {detector}')

        # Minimize test statistic with given bounds
        bnds = ((0, None), (0, None), (0, 1.2), (0, None), (0, 0.5))
        parameters = starting_guesses(detector, directory)
        args = (bin_centres, events, detector, directory)
        popt = scipy.optimize.minimize(fit_function, parameters, bounds=bnds,
                                       args=args)
        print(f'a = {popt.x[0]}')
        print(f'b = {popt.x[1]}')
        print(f'l = {popt.x[2]}')
        print(f'k = {popt.x[3]}')
        print(f'x0 = {popt.x[4]}')
        print('')
        print_popt(popt)
        plot_model(popt.x, bin_centres, events, u_events, detector)

        # Plot thresholds
        plot_threshold(detector, popt.x[4])

        # Save thresholds to file
        with open('thresholds_S2.txt', 'a') as handle:
            handle.write(f'{detector} {popt.x[4]:.4f}\n')

        # Save fit parameters to file
        with open(f'{detector}.txt', 'w') as handle:
            handle.write(f'a  {popt.x[0]}\n')
            handle.write(f'b  {popt.x[1]}\n')
            handle.write(f'l  {popt.x[2]}\n')
            handle.write(f'k  {popt.x[3]}\n')
            handle.write(f'x0 {popt.x[4]}\n')

    return popt


if __name__ == '__main__':
    # Analysed shots
    shot_numbers = [100054, 100055, 100056, 100057, 100058, 100059, 100060,
                    100061, 100062, 100063, 100064, 100068, 100069, 100070,
                    100072, 100073, 100074, 100075, 100077, 100078, 100079,
                    100080, 100081, 100082, 100083, 100084, 100085, 100086,
                    100087, 100088]
    detectors = dfs.get_dictionaries('S2')
    detectors = ['S2_29']
    directory = '20-11-2020'
    popt = main(shot_numbers, detectors, directory)
