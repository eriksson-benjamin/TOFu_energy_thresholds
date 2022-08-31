# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:16:28 2022

@author: bener807
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, 'C:/python/useful_definitions/')
import useful_defs as udfs
import find_thresholds_S1 as fcns1
import find_thresholds_S2 as fcns2


def plot_for_paper(shot_numbers):
    """Produce plots for paper."""
    detectors = ['S1_01', 'S1_02', 'S2_10', 'S2_14']
    spec_dets = ['S1_01', 'S1_02', 'S1_03', 'S1_04', 'S1_05', 'S2_12', 'S2_16']

    fig, axes = plt.subplots(2, 2, sharey='row')
    plt.subplots_adjust(hspace=0.3)
    letters = ['(a)', '(b)', '(c)', '(d)']
    xranges = [[0, 0.12], [0, 0.12], [0.0, 0.4], [0.0, 0.4]]
    yranges = [[-0.2, 1.1], [-0.2, 1.1], [-0.2, 1.1], [-0.2, 1.1]]
    for i, ax in enumerate(axes.flatten()):
        if detectors[i] in spec_dets:
            bin_centres, events = fcns1.read_data(detectors[i])
        else:
            # Merge all data for given shot numbers
            energies = fcns2.merge_data(shot_numbers, 20, 80)
            bin_edges = fcns2.get_bin_edges(detectors[i])
            bin_centres = bin_edges[1:] - np.diff(bin_edges) / 2

            # Histogram the energy spectrum and normalize
            events, _ = np.histogram(energies[detectors[i]], bin_edges)

        # Normalize
        u_events = np.sqrt(events) / events.max()
        events = events / events.max()

        # Read fit parameters
        parameters = np.loadtxt(f'output/fit_parameters/{detectors[i]}.txt',
                                dtype='float', usecols=1)
        threshold = parameters[-1]

        # Light yield spectrum
        ax.plot(1E3 * bin_centres, events / events.max(), 'k.', markersize=2)
        ax.errorbar(1E3 * bin_centres, events, yerr=u_events,
                    color='k', linestyle='None')

        # Plot model on data
        y_fit = fcns2.model(*parameters, bin_centres)
        ax.plot(1E3 * bin_centres, y_fit, 'r-')

        # Thresholds
        ax.axvline(1E3 * threshold, color='k', linestyle='--')

        # Configure plots
        ax.text(0.85, 0.83, letters[i], transform=ax.transAxes)
        ax.text(1.15 * 1E3 * threshold, -0.1,
                f'E$_{{thr}}$ = {1E3*threshold:.0f} keV$_{{ee}}$')

        ax.set_xlim(1E3 * xranges[i][0], 1E3 * xranges[i][1])
        ax.set_ylim(yranges[i][0], yranges[i][1])
    ax.text(0.47, 0.05, 'E$_{ee}$ (keV$_{ee}$)', transform=fig.transFigure)
    ax.text(0.08, 0.5, 'counts (a.u.)', transform=fig.transFigure,
            rotation='vertical')


if __name__ == '__main__':
    # Analysed shots
    shot_numbers = [100054, 100055, 100056, 100057, 100058, 100059, 100060,
                    100061, 100062, 100063, 100064, 100068, 100069, 100070,
                    100072, 100073, 100074, 100075, 100077, 100078, 100079,
                    100080, 100081, 100082, 100083, 100084, 100085, 100086,
                    100087, 100088]

    plot_for_paper(shot_numbers)
