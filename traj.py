import scipy.io
import sys
import numpy as np 
import torch
import pdb
import matplotlib.pyplot as plt
import pickle
from common.data_normalization import *
from common.pt_build_model import *
from common.utils import *

data_type = 'pos' #type of data used for this task
task = 'sim_A' #Which task we're training. This tells us what file to use
skip_step = 1
held_out = .9

data_type_offset = {'load':2, 'pos':0, 'velocity': 2, 'angle': 0}
task_offset = {'real_old':14, 'real_A':10, 'real_B':10, 'sim':6, 'sim_A': 6, 'acrobot': 5}
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
elif task == 'sim_A':
	datafile_name = 'data/robotic_hand_simulator/A/sim_data_discrete_v14_d4_m1_episodes.obj'
	with open(datafile_name, 'rb') as pickle_file:
		data_matrix= pickle.load(pickle_file, encoding='latin1')
		DATA = np.concatenate(data_matrix, 0)
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
DATA = DATA[0::1000]

x_data = DATA[:, :task_ofs]
y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+2] - DATA[:, dt_ofs:dt_ofs+2]


x_mean_arr = np.mean(x_data, axis=0)
x_std_arr = np.std(x_data, axis=0)
y_mean_arr = np.mean(y_data, axis=0)
y_std_arr = np.std(y_data, axis=0)
x_data = z_score_normalize(x_data, x_mean_arr, x_std_arr)
y_data = z_score_normalize(y_data, y_mean_arr, y_std_arr)

# pdb.set_trace()

skip = y_data[:,0]**2 > 8
# print(np.apply_along_axis(np.max, DATA, 0))
y_data = y_data*(1-skip)[:, None]
# skip2 = 
# pdb.set_trace()
cutoffa = 0#2200
cutoffb = -1#3000

model_file = 'save_model/robotic_hand_simulator/pytorch/sim_A_heldout' + str(held_out) + '_1.pkl'
with open(model_file, 'rb') as pickle_file:
    print("Running " + model_file)
    model = torch.load(pickle_file, map_location='cpu')

model.eval()

# x_data = x_data[0::1000]

batch_size = 256
batches = [x_data[i: max(len(x_data), i+batch_size)] for i in range(0, len(x_data), batch_size)]

outs = [model(torch.tensor(batch, dtype=torch.float)).detach().numpy()[:, dt_ofs:dt_ofs+2] for batch in batches]

while True:
	plt.figure(1)
	# plt.scatter(y_data[0, 0], marker="*", label='start')
	# plt.plot(DATA[cutoffa:cutoffb, 0], DATA[cutoffa:cutoffb, 1], color='blue', label='Ground Truth', marker='.')
	# plt.plot(DATA[:, 4], DATA[:, 5], color='blue', label='Ground Truth', marker='.')
	plt.plot(y_data[:,0],y_data[:,1], color='blue', label='Ground Truth', marker='.')
	for out in outs:
		plt.plot(out[:,0],out[:,1], color='red', label='Ground Truth', marker='.')
	# plt.plot(y_data[:100,2],y_data[:100,3], color='blue', label='Ground Truth', marker='.')
	# plt.plot(y_data_mod[:,0]**2, color='blue', label='Ground Truth', marker='.')
	# plt.hist(DATA[:, 0], bins=1000)
	# plt.hist(y_data[:, 0]**2, bins=1000)
	plt.show()
	break