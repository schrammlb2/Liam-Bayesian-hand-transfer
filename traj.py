import scipy.io
import sys
import numpy as np 
import pdb
import matplotlib.pyplot as plt
import pickle

data_type = 'load' #type of data used for this task
task = 'real_B' #Which task we're training. This tells us what file to use
skip_step = 1
held_out = .1

data_type_offset = {'load':2, 'pos':0, 'velocity': 2, 'angle': 0}
task_offset = {'real_old':14, 'real_A':10, 'real_B':10, 'sim':6, 'acrobot': 5}
dt_ofs = data_type_offset[data_type]
task_ofs = task_offset[task]

def apply_along_timepoints(f, data):
	lst = [
		f(DATA[:,0]),
		f(DATA[:,1]),
		f(DATA[:,2]),
		f(DATA[:,3])
	]
	return lst

if task=='sim':
	datafile_name = 'data/robotic_hand_simulator/B/sim_data_partial_v13_d4_m1.mat'
	DATA = scipy.io.loadmat(datafile_name)['D']
elif task == 'real_A':
	# datafile_name = 'data/robotic_hand_real/A/t42_cyl45_right_data_discrete_v0_d4_m1.obj'
	datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1.obj'
	with open(datafile_name, 'rb') as pickle_file:
		data_matrix, state_dim, action_dim, _, _ = pickle.load(pickle_file, encoding='latin1')
	DATA = np.asarray(data_matrix)
elif task == 'real_B':
	# datafile_name = 'data/robotic_hand_real/A/t42_cyl45_right_data_discrete_v0_d4_m1.obj'
	datafile_name = 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1.obj'
	with open(datafile_name, 'rb') as pickle_file:
		data_matrix, state_dim, action_dim, _, _ = pickle.load(pickle_file, encoding='latin1')
	DATA = np.asarray(data_matrix)

elif task == 'real_old':
	datafile_name = 'data/robotic_hand_real/old_data/t42_cyl45_data_discrete_v0_d12_m1.obj'
	with open(datafile_name, 'rb') as pickle_file:
		data_matrix, state_dim, action_dim, _, _ = pickle.load(pickle_file, encoding='latin1')
	DATA = np.asarray(data_matrix)

elif task == 'acrobot':
	with open('data/acrobot/acrobot_data_v2_d4', 'rb') as pickle_file:
	# with open('data/acrobot/test_trajectory/acrobot_ao_rrt_plan1', 'rb') as pickle_file:
	    DATA = pickle.load(pickle_file)
	    # DATA = DATA[:10^6]


y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+2] - DATA[:, dt_ofs:dt_ofs+2]

skip = y_data[:,0]**2 > 1
# print(np.apply_along_axis(np.max, DATA, 0))
y_data = y_data*(1-skip)[:, None]
# skip2 = 
# pdb.set_trace()
cutoffa = 0#2200
cutoffb = -1#3000
while True:
	plt.figure(1)
	# plt.scatter(y_data[0, 0], marker="*", label='start')
	pdb.set_trace()
	plt.plot(DATA[cutoffa:cutoffb, 0], DATA[cutoffa:cutoffb, 1], color='blue', label='Ground Truth', marker='.')
	# plt.plot(DATA[:, 4], DATA[:, 5], color='blue', label='Ground Truth', marker='.')
	# plt.plot(y_data[:100,0],y_data[:100,1], color='blue', label='Ground Truth', marker='.')
	# plt.plot(y_data_mod[:,0]**2, color='blue', label='Ground Truth', marker='.')
	# plt.hist(DATA[:, 0], bins=1000)
	# plt.hist(y_data[:, 0]**2, bins=1000)
	plt.show()