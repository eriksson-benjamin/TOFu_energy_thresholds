#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 08:13:57 2022

@author: beriksso
"""

'''
Calculates coincidences between S1/S2 data and saves corresponding energies
to file.
'''

import sys
sys.path.insert(0, '/home/beriksso/python/definitions/')
import useful_defs as udfs
sys.path.insert(1, '/home/beriksso/TOFu/analysis/benjamin/github/TOFu/functions/')
import tofu_functions as dfs
import numpy as np
import multiprocessing as mp

def get_tof(arguments):
    '''
    Used in main function for parallelization of finding 
    coincidences between S1's and S2's
    '''
    s1 = arguments[0:5]
    s2 = arguments[6:]
    
    s1_times = times_S1[s1]
    s2_times = times_S2[s2]
    
    tof, inds = dfs.sTOF4(s1_times, s2_times, t_back=time_window, 
                          t_forward=time_window, return_indices=True, 
                          timer=time_level)
    return tof, inds, s1, s2

def calculate_coincidences(shot_number):
    '''
    Coincidences are used to avoid the low energy peak of random background
    seen when plotting the entire pulse height spectrum.
    '''
    # Arguments to parallelize finding coincidences between S1s and S2s   
    tof_argu = np.array([], dtype='str')
    S1_dict, S2_dict = dfs.get_dictionaries()
    for s1 in S1_dict.keys():
        for s2 in S2_dict.keys():
            tof_argu = np.append(tof_argu, f'{s1} {s2}')
    
    # Calculate coincidences
    available_cpu = mp.cpu_count()
    if available_cpu > 16: available_cpu=16
    pool_tof = mp.Pool(available_cpu)
    tof_info = pool_tof.map(get_tof, tof_argu)
    
    # Return energies corresponding to coincident events
    coincident_energies = dfs.get_dictionaries('merged')
    flight_times = dfs.get_dictionaries('merged')
    for lst in tof_info:
        # Detector names
        det_1 = lst[2]
        det_2 = lst[3]
        
        # Indices to save
        inds_1 = lst[1][0]
        inds_2 = lst[1][1]
    
        # Energies
        coincident_energies[det_1] = np.append(coincident_energies[det_1], 
                                               energy_S1[det_1][inds_1])
        coincident_energies[det_2] = np.append(coincident_energies[det_2], 
                                               energy_S2[det_2][inds_2])
        # Times
        flight_times[det_1] = np.append(flight_times[det_1], 
                                        times_S2[det_2][inds_2] 
                                        - times_S1[det_1][inds_1])
    return coincident_energies, flight_times

def import_data(shot_number):
    path = f'/common/scratch/beriksso/TOFu/data/{shot_number}/{shot_number}_thr.pickle'
    all_data = udfs.unpickle(path)
    times_S1 = all_data['new_times_S1']
    times_S2 = all_data['new_times_S2']
    energy_S1 = all_data['energy_S1'] # MeVee
    energy_S2 = all_data['energy_S2'] # MeVee
    
    return times_S1, times_S2, energy_S1, energy_S2

if __name__ == '__main__':
    # Set globals
    disabled_detectors = []
    time_window = 500
    time_level = 0
    
    # Import data from shots
    shot_numbers = [100054, 100055, 100056, 100057, 100058, 100059, 100060]
    for shot_number in shot_numbers:
        # Import time/energy data    
        times_S1, times_S2, energy_S1, energy_S2 = import_data(shot_number)
    
        # Calculate coincident events
        coin_ergs, flight_times = calculate_coincidences(shot_number)
       
        # Save to file
        udfs.pickler(f'data/coincident_energies/{shot_number}.pickle', 
                      coin_ergs)
        udfs.pickler(f'data/flight_times/{shot_number}.pickle', 
                     flight_times)

