import scipy.io
from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb

from sys import argv

data_type = 'pos' #type of data used for this task
task = 'real' #Which task we're training. This tells us what file to use
skip_step = 1
outfile = None
append = False
held_out = .1
# _ , arg1, arg2, arg3 = argv

# pdb.set_trace()
if len(argv) > 1:
	data_type = argv[1]

assert data_type in ['pos', 'load', 'finger']
assert task in ['real']
if (len(argv) > 3 and task != 'sim_A'):
	print('Err: Skip step only appicable to sim_A task. Do not use this argument for other tasks')
	exit(1)


elif task == 'real_A': 
	# datafile_name = 'data/robotic_hand_real/A/t42_cyl45_right_data_discrete_v0_d4_m1.obj'

	datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d10_m1.obj'
	save_path = 'save_model/robotic_hand_real/A/' + data_type
	with open(datafile_name, 'rb') as pickle_file:
		data_matrix, state_dim, action_dim, _, _ = pickle.load(pickle_file, encoding='latin1')
	DATA = np.asarray(data_matrix)
	# task_ofs= state_dim+action_dim


# pdb.set_trace()

data_type_offset = {'pos':0, 'load':2, 'finger':4}
state_dim = 10
action_dim = 6
dt_ofs = data_type_offset[data_type]
task_ofs = state_dim + action_dim





n=1.0
p_max_0 = np.percentile(DATA[:,2], 100.0-n)
p_min_0 = np.percentile(DATA[:,2], n)
p_max_1 = np.percentile(DATA[:,3], 100.0-n)
p_min_1 = np.percentile(DATA[:,3], n)
yd_0 = np.clip(DATA[:,2], p_min_0, p_max_0)
yd_1 = np.clip(DATA[:,3], p_min_1, p_max_1)
yd = np.stack([yd_0, yd_1], axis=1)
DATA = np.concatenate([DATA[:,0:2], yd, DATA[:,4:]], axis=1)



x_data = DATA[:, :task_ofs]
y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+2] - DATA[:, dt_ofs:dt_ofs+2]

method = 'clip'



if __name__ == "__main__":
	neural_network = BNN(nn_type='2', dropout_p=0)
	neural_network.add_dataset(x_data, y_data, held_out_percentage=held_out)
	neural_network.build_neural_net()
	final_loss = neural_network.train(save_path=save_path, normalization=True, normalization_type='z_score', decay='True')
	# final_loss = neural_network.train(save_path=save_path, normalization=True, normalization_type='z_score', decay='True', load_path=save_path)
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
    