#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:03:11 2022

@author: beriksso
"""

"""
Extracts the energy spectrum for S2 detectors under the coincidence requirement
with an S1 from data files generated using the --save-data argument in 
create_TOF.py
"""


import numpy as np
import useful_defs as udfs
udfs.set_nes_plot_style()
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/beriksso/TOFu/analysis/benjamin/github/TOFu/functions')
import tofu_functions as dfs


def extract_energies(f_name):
    """Extract the energy spectrum from the file."""
    dat = udfs.unpickle(f_name)
    
    # Create dictionaries
    S1_dict, S2_dict = dfs.get_dictionaries()
    S1_energies = dfs.get_dictionaries('S2')
    S2_energies = dfs.get_dictionaries('S2')
    S2_tof = dfs.get_dictionaries('S2')
    
    # Extract energy/times-of-flight for each S1/S2 combo
    for S1 in S1_dict.keys():
        for S2 in S2_dict.keys():
            S1_energies[S2] = np.append(S1_energies[S2], 
                                        dat['energies'][S1][S2][0, :])
            S2_energies[S2] = np.append(S2_energies[S2], 
                            dat['energies'][S1][S2][1, :])
            S2_tof[S2] = np.append(S2_tof[S2], dat['times_of_flight'][S1][S2])
            
    return S1_energies, S2_energies, S2_tof


def plot_hist(detector, energies):
    """Plot energy spectrum."""
    plt.figure(detector)
    bin_edges = np.arange(0, 5, 0.001)
    bin_centres = udfs.get_bin_centres(bin_edges)
    counts, _ = np.histogram(energies, bins=bin_edges)
    plt.errorbar(bin_centres, counts, yerr=np.sqrt(counts), color='k', 
                 linestyle='None')
    plt.xlabel('$E_{ly}$ (MeVee)')
    plt.ylabel('counts')
    
    
def write_file(S2, energies):
    """Write extracted energies to file."""
    udfs.pickler(f'{S2}.pickle', energies)
    
    
if __name__ == '__main__':
    shots = [100054, 100055, 100056, 100057, 100058, 100059, 100060, 100061, 
             100062, 100063, 100064, 100068, 100069, 100070, 100072, 100073, 
             100074, 100075, 100077, 100078, 100079, 100080, 100081, 100082, 
             100083, 100084, 100085, 100086, 100087, 100088]
    path = '/common/scratch/beriksso/TOFu/data/energy_thresholds/26-11-2022'
    
    S1_energies = dfs.get_dictionaries('S2')
    S2_energies = dfs.get_dictionaries('S2')
    times_of_flight = dfs.get_dictionaries('S2')
    print('Extracting data...')
    for shot in shots:
        f_name = f'{path}/{shot}_40.0_80.0.pickle'
        S1_ergs, S2_ergs, tof = extract_energies(f_name)
        for S2 in S2_energies.keys():
            S1_energies[S2] = np.append(S1_energies[S2], S1_ergs[S2])
            S2_energies[S2] = np.append(S2_energies[S2], S2_ergs[S2])
            times_of_flight[S2] = np.append(times_of_flight[S2], tof[S2])
        print(f'{shot} complete.')
    
    print('Applying kinematic cuts...')
    for S2 in S2_energies.keys():
        # Perform kinematic cuts on data
        _, _, erg_S2_cut = dfs.kinematic_cuts(times_of_flight[S2], 
                                              S1_energies[S2], S2_energies[S2],
                                              cut_factors=(0.7, 1.4, 1.2))
        
        plot_hist(S2, erg_S2_cut)
        print(f'{S2} complete. Writing to file.')
        write_file(S2, erg_S2_cut)
        
        






