import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from common.get_wave import gwseries
from common.wienerseries import Wiener_class
from common.utils import *        
from common.get_extraction import wiener_filter

from common.corr_analysis import match_pair


merger_name = 'GW170814'

data = gwseries(merger_name = merger_name)
data = Wiener_class(data, fs = None, nfft = None, nperseg = None, noverlap = None, 
                    window = 'hann', filt_type = 'hrnr')
## TODO: the wiener function should be replaced by a class for automatic search
fs = data.fs
merger_time = data.merger_time
GPSTime = None
if merger_time > 1e5:
    start_time = data.start_time
    merger_time = (merger_time - start_time) * fs
sample_times = data.sample_times
strain = list(data.strain.values()) 
num_detectors = len(strain)
strain_length = len(strain[0])
strain = np.array(strain)
if strain.shape[0] != num_detectors:
    strain = strain.T
noise = get_slice_noise(strain = strain, merger_time = merger_time, axis = -1)
if GPSTime:
    sample_times = get_slice(strain = sample_times, merger_time = merger_time, axis = -1)
    strain = get_slice(strain = strain, merger_time= merger_time, axis = -1)
else:
    strain = get_slice(strain = strain, merger_time= merger_time, axis = -1)
    sample_times = np.linspace(-strain.shape[1]//2,  strain.shape[1]//2, strain.shape[1])/fs


out = wiener_filter(strain, noise, fs = 4096, nfft = None, nperseg = None, noverlap = None, 
             window = 'hann', filt_type = 'hrnr', axis = -1)

first_run = ['GW150914', 'GW151012', 'GW151226']
second_run = ['GW170104', 'GW170608', 'GW170729',
              'GW170809', 'GW170814', 'GW170818', 'GW170823']
third_run = ['GW190412', 'GW190814', 'GW190521']
###
events_key = first_run + second_run + third_run
events_val = [['GW150914.txt', []], ['GW151012.txt', []],
              ['GW151226.txt', []], ['GW170104.txt', []],
              ['GW170608.txt', []], ['GW170729.txt', []],
              ['GW170809.txt', []], ['GW170814.txt', []],
              ['GW170818.txt', []], ['GW170823.txt', []],
              ['GW190412.txt', []], ['GW190814.txt', []],
              ['GW190521.txt', []]]
events_ = dict(zip(events_key, events_val))
###
strain_file_name = events_[merger_name][0]
file_to_eval = np.genfromtxt(f'./results/{strain_file_name}')
out_ = match_pair(merger_name, file_to_eval)
# out_ = match(event_name, out)

plt.figure()
plt.plot(sample_times, out[0,:], label = 'H')
plt.plot(sample_times, out[1,:], label= 'L')
plt.xlim(-0.2, 0.2 )