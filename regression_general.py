import scipy.io
from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb

data_type = 'load' #type of data used for this task
task = 'real' #Which task we're training. This tells us what file to use
skip_step = 1
data_type_offset = {'load':2, 'pos':0}
# task_offset = {'real_old':14, 'real':6, 'simulator':6}
task_offset = {'real_old':14, 'real':10, 'simulator':6}
dt_ofs = data_type_offset[data_type]
task_ofs = task_offset[task]

if task == 'simulator':
	datafile_name = 'data/robotic_hand_simulator/sim_data_discrete_v13_d4_m' + str(skip_step) + '.mat'
	save_path = 'save_model/robotic_hand_simulator/d4_s' + str(skip_step) + '_' + data_type
	DATA = scipy.io.loadmat(datafile_name)['D']

elif task == 'real': 
	datafile_name = 'data/robotic_hand_real/t42_cyl45_right_data_discrete_v0_d4_m1.obj'
	save_path = 'save_model/robotic_hand_real/' + data_type
	with open(datafile_name, 'rb') as pickle_file:
		data_matrix, state_dim, action_dim, _, _ = pickle.load(pickle_file, encoding='latin1')
	DATA = np.asarray(data_matrix)
	# task_ofs= state_dim+action_dim

elif task == 'real_old': 
	datafile_name = 'data/robotic_hand_real/old_data/t42_cyl45_data_discrete_v0_d12_m1.obj'
	save_path = 'save_model/robotic_hand_real/' + data_type
	with open(datafile_name, 'rb') as pickle_file:
		data_matrix, state_dim, action_dim, _, _ = pickle.load(pickle_file, encoding='latin1')
	DATA = np.asarray(data_matrix)



# DATA = scipy.io.loadmat(datafile_name)['D']
x_data = DATA[:, :task_ofs]
y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+2] - DATA[:, dt_ofs:dt_ofs+2]
# if task == 'real':
# x_data = DATA[:, :task_ofs]
# y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+2] - DATA[:, dt_ofs:dt_ofs+2]
if __name__ == "__main__":
	neural_network = BNN(nn_type='0')
	neural_network.add_dataset(x_data, y_data, held_out_percentage=0.1)
	neural_network.build_neural_net()
	neural_network.train(save_path=save_path, normalization=True, normalization_type='z_score', decay='False')#, load_path=save_path)
    