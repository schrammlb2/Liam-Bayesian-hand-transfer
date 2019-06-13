import scipy.io
from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb

from sys import argv

data_type = 'load' #type of data used for this task
task = 'sim_B' #Which task we're training. This tells us what file to use
skip_step = 1
outfile = None
append = False
held_out = .1
# _ , arg1, arg2, arg3 = argv

# pdb.set_trace()
if len(argv) > 1:
	data_type = argv[1]
if len(argv) > 2:
	task = argv[2]
if len(argv) > 3:
	skip_step = int(argv[3])
if len(argv) > 4:
	outfile = argv[4]
if len(argv) > 5:
	append = (argv[5] in ['True' ,'T', 'true', 't'])
if len(argv) > 6:
	held_out = float(argv[6])

assert data_type in ['pos', 'load']
assert task in ['real_A', 'real_B','sim_A', 'sim_B']
assert skip_step in [1,10]
if (len(argv) > 3 and task != 'sim_A'):
	print('Err: Skip step only appicable to sim_A task. Do not use this argument for other tasks')
	exit(1)


if task == 'sim_A':
	datafile_name = 'data/robotic_hand_simulator/A/sim_data_discrete_v13_d4_m' + str(skip_step) + '.mat'
	save_path = 'save_model/robotic_hand_simulator/A/d4_s' + str(skip_step) + '_' + data_type
	DATA = scipy.io.loadmat(datafile_name)['D']

elif task == 'sim_B':
	datafile_name = 'data/robotic_hand_simulator/B/sim_data_partial_v13_d4_m1.mat'
	save_path = 'save_model/robotic_hand_simulator/B/d4_s1_' + data_type
	DATA = scipy.io.loadmat(datafile_name)['D']

elif task == 'real_A': 
	datafile_name = 'data/robotic_hand_real/A/t42_cyl45_right_data_discrete_v0_d4_m1.obj'
	save_path = 'save_model/robotic_hand_real/A/' + data_type
	with open(datafile_name, 'rb') as pickle_file:
		data_matrix, state_dim, action_dim, _, _ = pickle.load(pickle_file, encoding='latin1')
	DATA = np.asarray(data_matrix)
	# task_ofs= state_dim+action_dim
elif task == 'real_B': 
	datafile_name = 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1.obj'
	save_path = 'save_model/robotic_hand_real/B/' + data_type
	with open(datafile_name, 'rb') as pickle_file:
		data_matrix, state_dim, action_dim, _, _ = pickle.load(pickle_file, encoding='latin1')
	DATA = np.asarray(data_matrix)

elif task == 'real_old': 
	datafile_name = 'data/robotic_hand_real/old_data/t42_cyl45_data_discrete_v0_d12_m1.obj'
	save_path = 'save_model/robotic_hand_real/old/' + data_type
	with open(datafile_name, 'rb') as pickle_file:
		data_matrix, state_dim, action_dim, _, _ = pickle.load(pickle_file, encoding='latin1')
	DATA = np.asarray(data_matrix)


# pdb.set_trace()

data_type_offset = {'load':2, 'pos':0}
task_offset = {'real_old':14, 'real_A':10,'real_B':10, 'sim_A':6, 'sim_B':6}
dt_ofs = data_type_offset[data_type]
task_ofs = task_offset[task]

# DATA = scipy.io.loadmat(datafile_name)['D']
x_data = DATA[:, :task_ofs]
y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+2] - DATA[:, dt_ofs:dt_ofs+2]
# if task == 'real_A':
# x_data = DATA[:, :task_ofs]
# y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+2] - DATA[:, dt_ofs:dt_ofs+2]
# pdb.set_trace()
if __name__ == "__main__":
	neural_network = BNN(nn_type='0')
	neural_network.add_dataset(x_data, y_data, held_out_percentage=held_out)
	neural_network.build_neural_net()
	final_loss = neural_network.train(save_path=save_path, normalization=True, normalization_type='z_score', decay='True')#, load_path=save_path)
	if outfile: 
		if append:
			f = open(outfile, 'a+')
		else:
			f = open(outfile, 'w+')
		out_string= ('cold start\t' + data_type +
					'\t' + task + '\t' + str(held_out) +
					'\t:' + str(final_loss) + '\n')
		f.write(out_string)
		f.close()
    