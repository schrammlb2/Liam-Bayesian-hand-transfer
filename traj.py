import scipy.io
import sys
import numpy as np 
import pdb
import matplotlib.pyplot as plt

data_type = 'pos' #type of data used for this task
task = 'sim' #Which task we're training. This tells us what file to use
skip_step = 1
held_out = .1

data_type_offset = {'load':2, 'pos':0}
task_offset = {'real_old':14, 'real':10, 'sim':6}
dt_ofs = data_type_offset[data_type]
task_ofs = task_offset[task]

datafile_name = 'data/robotic_hand_simulator/B/sim_data_partial_v13_d4_m1.mat'
DATA = scipy.io.loadmat(datafile_name)['D']
y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+2] - DATA[:, dt_ofs:dt_ofs+2]

pdb.set_trace()
plt.figure(1)
# plt.scatter(y_data[0, 0], marker="*", label='start')
plt.plot(y_data[:, 0]**2, color='blue', label='Ground Truth', marker='.')
plt.show()