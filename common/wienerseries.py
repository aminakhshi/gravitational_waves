#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 20:01:41 2022

@author: amin
"""
import numpy as np
from scipy import signal
import math

def nexpow2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

class Wiener_class(object):
    def __init__(self, gw_array, fs = None, nfft = None, nperseg = None, noverlap = None, 
                 window = 'hann', filt_type = 'hrnr'):
        try:
            self.merger_name = gw_array.merger_name
        except:
            print("merger name is unknown")
            pass
        
        assert len(gw_array.strain) > 1, "strain array does not exist"
        try:
            start_time = gw_array.sample_times[0]
            merger_time = gw_array.merger_time
        except:
            raise TypeError('start time and merger time should be defined')
        self.start_time = start_time
        self.merger_time = merger_time
        self.strain = {}
        for ifo in gw_array.strain.keys():
            self.strain[ifo] = np.array(gw_array.strain[ifo])
        if not fs:
            try:
                fs = gw_array.fs
            except:
                raise ValueError('sampling rate is not defined')
        self.fs = fs
        self.delta_t = 1/fs
        try:
            self.sample_times = np.array(gw_array.sample_times)
        except:
            print("GPS Time is not available. Switch to seconds")
            self.merger_time = (self.merger_time - self.start_time) * self.fs
            # self.sample_times = np.arange()        
        if not nperseg:
            nperseg = np.fix(0.06*self.fs).astype(int)
        if np.remainder(nperseg, 2) == 1:
            nperseg = nperseg + 1
        self.nperseg = nperseg
        
        if not noverlap:
            self.noverlap = np.fix(0.5*self.nperseg).astype(int)
            self.offset = self.nperseg - self.noverlap
        else:
            self.noverlap = noverlap
            self.offset = self.nperseg - self.noverlap            
        if not nfft:
            nfft = max([256, nexpow2(self.nperseg)])
        self.nfft = nfft
        # if data_cut:
        self.window = signal.get_window(window, self.nperseg)
    
    def _get_fft(self, cut_sec = None, axis = -1, astype = None):
        # if not cut_sec:
        strain = list(self.strain.values())
        strain_psd = []
        for val in strain:
            _, _, wkn = signal.stft(val, self.fs, window = self.window,
                                    nperseg= self.nperseg, noverlap= self.offset,
                                    nfft= self.nfft)
            strain_psd.append(wkn)
        # if astype == 'init_noise':
        #     wkn = np.mean(abs(wkn), axis = axis)
        return strain_psd   