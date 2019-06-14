import scipy.io
import sys
import numpy as np 
import pdb
import matplotlib.pyplot as plt
import pickle

data_type = 'pos' #type of data used for this task
task = 'real' #Which task we're training. This tells us what file to use
skip_step = 1
held_out = .1

data_type_offset = {'load':2, 'pos':0}
task_offset = {'real_old':14, 'real':10, 'sim':6}
dt_ofs = data_type_offset[data_type]
task_ofs = task_offset[task]

if task=='sim':
	datafile_name = 'data/robotic_hand_simulator/B/sim_data_partial_v13_d4_m1.mat'
	DATA = scipy.io.loadmat(datafile_name)['D']
elif task == 'real':
	datafile_name = 'data/robotic_hand_real/A/t42_cyl45_right_data_discrete_v0_d4_m1.obj'
	with open(datafile_name, 'rb') as pickle_file:
		data_matrix, state_dim, action_dim, _, _ = pickle.load(pickle_file, encoding='latin1')
	DATA = np.asarray(data_matrix)

elif task == 'real_old':
	datafile_name = 'data/robotic_hand_real/old_data/t42_cyl45_data_discrete_v0_d12_m1.obj'
	with open(datafile_name, 'rb') as pickle_file:
		data_matrix, state_dim, action_dim, _, _ = pickle.load(pickle_file, encoding='latin1')
	DATA = np.asarray(data_matrix)
y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+2] - DATA[:, dt_ofs:dt_ofs+2]

skip = y_data[:,0]**2 > .03
# skip2 = 
# pdb.set_trace()
plt.figure(1)
y_data_mod = y_data*(1-skip)[:, None]
# plt.scatter(y_data[0, 0], marker="*", label='start')
# plt.plot(DATA[:, 0], color='blue', label='Ground Truth', marker='.')
plt.plot(y_data[:,0]**2, color='blue', label='Ground Truth', marker='.')
# plt.plot(y_data_mod[:,0]**2, color='blue', label='Ground Truth', marker='.')
# plt.hist(y_data_mod[:, 0]**2, bins=1000)
plt.show()