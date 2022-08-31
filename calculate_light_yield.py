# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:12:02 2022

@author: bener807
"""

import sys
sys.path.insert(0, 'C:/python/useful_definitions/')
import useful_defs as udfs
sys.path.insert(0, 'C:/TOFu/functions/')
import tofu_functions as dfs
import numpy as np
import matplotlib.pyplot as plt
dfs.set_plot_style()


def calculate_light_yield(detector):
    """Calculate the light yield spectra of Na-22 data."""
    # Import pulse waveforms
    board, channel = dfs.get_board_name(detector)
    if int(board) < 6:
        rec_len = 64
        bias_level = 27000
    else:
        rec_len = 56
        bias_level = 1600

    path = '../gamma_calibration/data/raw_data/20-11-2020/'
    p = udfs.import_pulses(board=board, channel=channel, path=path,
                           record_length=rec_len)

    # Perform baseline reduction
    dat = dfs.baseline_reduction(p)

    # Remove bad pulses
    dat, _ = dfs.cleanup(dat, dx=1, detector_name=detector,
                         bias_level=bias_level)

    # Perform sinc interpolation
    dat_sinc = dfs.sinc_interpolation(dat, np.arange(0, rec_len),
                                      np.arange(0, rec_len, 0.1))

    # Perform integration between 10-30 ns
    dat_sinc = dat_sinc[:, 100:300]

    # # Calculate area
    p_ar = -dfs.get_pulse_area(dat_sinc, 10)

    # Calculate light yield
    p_ly = dfs.get_energy_calibration_(p_ar, detector)

    # Save to file
    to_save = {'light yield': p_ly}
    udfs.pickler(f'{detector}.pickle', to_save)

    return p_ly


def load_bins(detector):
    """Read binning information from text file."""
    f = np.loadtxt('data/Na-22/light_yield/bins.txt', dtype='str')
    row = np.where(f[:, 0] == detector)
    start = int(f[row][0][1])
    stop = int(f[row][0][2])
    width = float(f[row][0][3])

    return np.arange(start, stop, width)


def histogram_data(detector):
    """Bin light yield data and save bins/counts to file."""
    # Read light yield data
    p_ly = udfs.unpickle(f'data/Na-22/light_yield/full/{detector}.pickle')

    # Histogram data
    bins = load_bins(detector)
    bin_centres = bins[1:] - np.diff(bins) / 2
    h_ly, _ = np.histogram(p_ly['light yield'], bins=bins)

    # Plot
    plt.figure(detector)
    plt.plot(bin_centres, h_ly, 'k.', markersize=0.5)
    plt.errorbar(bin_centres, h_ly, yerr=np.sqrt(h_ly), linestyle='None',
                 color='k')
    plt.xlabel(r'E$_{ee}$ (MeV$_{ee}$)')
    plt.ylabel('counts/bins')
    plt.title(detector.replace('_', '-'), loc='right')
    plt.gca().ticklabel_format(style='plain')

    # Save to file
    to_save = {'bin centres': bin_centres,
               'counts': h_ly}
    udfs.pickler(f'{detector}.pickle', to_save, check=False)
    return bins, h_ly


if __name__ == '__main__':
    detectors = dfs.get_dictionaries('merged')
    detectors = ['S2_16']
    for detector in detectors:
        # p_ly = calculate_light_yield(detector)
        bins, h_ly = histogram_data(detector)
