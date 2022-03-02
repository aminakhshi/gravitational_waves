import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from common.get_wave import gwseries
from common.wienerseries import Wiener_class
from common.utils import *        
from common.get_extraction import wiener_filter


merger_name = 'GW150914'

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

# tau_max=0.02;
# t_vec = np.arange(-np.fix(tau_max*fs).astype(int),np.fix(tau_max*fs).astype(int))/fs
# for key, tau in enumerate((t_vec*fs)):
#     [corr,corr_err]=p_corr(data_mat(:,1),data_mat(:,2),tau);
#     amp_cr(cntr)=corr;
#     amp_cr_err(cntr)=corr_err;
plt.figure()
plt.plot(sample_times, out[0,:], label = 'H')
plt.plot(sample_times, out[1,:], label= 'L')
plt.xlim(-0.2, 0.2 )