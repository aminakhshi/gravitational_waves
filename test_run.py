from test_rr import main as match
import numpy as np

event_name = "GW170104"

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
strain_file_name = events_[event_name][0]
file_to_eval = np.genfromtxt(f'./gw_new/{strain_file_name}')
data = file_to_eval
out_ = match(event_name, data)
