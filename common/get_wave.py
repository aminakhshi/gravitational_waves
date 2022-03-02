from pycbc import catalog
import pickle
import numpy 
def get_gwseries(merger, detector : str, fs = 4096, duration = 32):
    # , fpass = 'high',
    #  notch = False, whiten = True, copy = False):
    # assert merger, "Merger object is not valid"
    # assert fs, "sampling rate is not defined"
    # assert duration, "duration is not defined"
    # assert detector, "detector is not defined"
    raw_data = merger.strain(detector, duration = duration, sample_rate = fs)
    raw_std = numpy.std(raw_data)
    return raw_data, raw_std    

class gwseries(object):
    """
    Pre-processed class for a specific merger
    Parameters
    ----------
    merger_name: str
    The name (GW simple fomrat date) of the merger event.
    fs : sampling frequency in Hz
    duration: duration time in sec
    _opts: 
    """   
    def __init__(self, merger_name: str, fs = None, duration = None, _opts = None):

        # for key, val in _opts:
        #     setattr(self, key, self.key)
        get_source = merger_name.split('GW')[1]
        if get_source.startswith(('15', '17')):
            m = catalog.Merger(merger_name, source='gwtc-1')
        else:
            m = catalog.Merger(merger_name, source='gwtc-2')
        if fs:
            self.fs = fs
        else:
            self.fs = 4096
        if duration:
            self.duration = duration
        else:
            self.duration = 32
        
        if not _opts:
            _opts = {'fpass': 'high',
                     'notch' : False,
                     'whiten' : True,
                     'norm': True,
                     'to_save' : False}
            
        self.merger_name = m.common_name
        self.merger_time = m.data['GPS']
        self.data_frame = m.frame
        self.strain = {}
        self.strain_std = {}
        for ifo in ['H1', 'L1', 'V1']:
            try:
                self.strain[ifo], self.strain_std[ifo] = get_gwseries(merger = m, detector = ifo,
                                                             fs = self.fs , duration = self.duration)
            except:
                continue
            if _opts['whiten']:
                self.strain[ifo] = self.strain[ifo].whiten(4, 4)
            if _opts['fpass'] == 'high':
                self.strain[ifo] = self.strain[ifo].highpass_fir(30, 512)
            elif _opts['fpass'] == 'low':
                self.strain[ifo] = self.strain[ifo].lowpass_fir(500, 512)
            elif _opts['fpass'] == 'band':
                self.strain[ifo] = self.strain[ifo].highpass_fir(30, 512)
                self.strain[ifo] = self.strain[ifo].lowpass_fir_fir(500, 512)
            if _opts['notch']:
                self.strain[ifo] = self.strain[ifo].notch_fir(49, 51, order = 30)
            if _opts['norm']:
                self.strain[ifo] = (self.strain[ifo] - numpy.mean(self.strain[ifo]))/numpy.std(self.strain[ifo])
                self.strain[ifo] = self.strain[ifo] * self.strain_std[ifo]
            self.sample_times = self.strain[ifo].get_sample_times()
            
        if _opts['to_save']:
            with open('{}.pickle'.format(self.merger_name), 'wb') as handle:
                pickle.dump(self.strain, handle, protocol=pickle.HIGHEST_PROTOCOL)
           