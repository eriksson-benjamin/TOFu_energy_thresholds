#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:15:48 2022

@author: beriksso
"""

import numpy as np
import sys
sys.path.insert(0, '/home/beriksso/TOFu/analysis/benjamin/github/TOFu/functions')
sys.path.insert(0, 'C:/python/TOFu/functions/')
import tofu_functions as dfs

thr = np.loadtxt('output/thresholds.txt', dtype='str')
thr_dict = {t[0]:float(t[1]) for t in thr}
thresholds = list(thr_dict.values())
energies = dfs.inverted_light_yield(thresholds)


with open('thresholds_MeV.txt', 'w') as handle:
    handle.write('# Detector Threshold (MeV)\n')

for key, erg in zip(thr_dict.keys(), energies):
    with open('thresholds_MeV.txt', 'a') as handle:
        handle.write(f'{key} {erg:.4f}\n')


