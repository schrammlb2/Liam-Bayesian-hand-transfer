import scipy.io
from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb

from sys import argv

data_type = 'pos' #type of data used for this task
task = 'real_A' #Which task we're training. This tells us what file to use
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



datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'
save_path = 'save_model/robotic_hand_real/episodic/' + data_type
with open(datafile_name, 'rb') as pickle_file:
	out = pickle.load(pickle_file, encoding='latin1')

state_dim = 4
action_dim = 6
alpha = .4

new_eps = []
for episode in out:
	x_ave = episode[0, 2:4]*0
	y_ave = episode[0, state_dim+action_dim+2:]*0
	x_ave_list_episode = []
	y_ave_list_episode = []
	for i in range(len(episode)):
		x_ave *= alpha
		y_ave *= alpha

		x_ave += episode[i, 2:4]*(1-alpha) 
		y_ave += episode[i, state_dim+action_dim+2:]*(1-alpha)
		x_ave_list_episode.append(x_ave)
		y_ave_list_episode.append(y_ave)
	x_stack = np.stack(x_ave_list_episode, axis=0)	
	y_stack = np.stack(y_ave_list_episode, axis=0)

	new_ep = np.concatenate([episode[:,:4], x_stack, episode[:,4:], y_stack], axis=1)
	new_eps.append(new_ep)
DATA = np.concatenate(new_eps)

new_state_dim = 6
# pdb.set_trace()
# data_matrix, state_dim, action_dim, _, _ = out
# DATA = np.asarray(data_matrix)
# task_ofs= state_dim+action_dim



data_type_offset = {'ave_load':4, 'load':2, 'pos':0}
# task_offset = {'real_old':14, 'real_A':10,'real_B':10, 'sim_A':6, 'sim_B':6}
dt_ofs = data_type_offset[data_type]
# task_ofs = task_offset[task]
task_ofs = new_state_dim + action_dim

# DATA = scipy.io.loadmat(datafile_name)['D']
# x_data = DATA[:, :task_ofs]
# if data_type == 'load':
# 	y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+4] - DATA[:, dt_ofs:dt_ofs+4]
# else:
# 	y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+2] - DATA[:, dt_ofs:dt_ofs+2]

x_data = DATA[:, :task_ofs]
y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+2] - DATA[:, dt_ofs:dt_ofs+2]
method = 'clip'
assert method in ['zero', 'clip']

# if task in ['real_A', 'real_B']:
# 	#Need to get this set up as a per-file basis
# 	#This is currectly for the wrong file
# 	skip = y_data[:,0]**2 > .032
# 	y_data = y_data*(1-skip)[:, None]
if task in ['real_A', 'real_B'] and method == 'clip':
	n=1.0
	p_max_0 = np.percentile(y_data[:,0], 100.0-n)
	p_min_0 = np.percentile(y_data[:,0], n)
	p_max_1 = np.percentile(y_data[:,1], 100.0-n)
	p_min_1 = np.percentile(y_data[:,1], n)
	yd_0 = np.clip(y_data[:,0], p_min_0, p_max_0)
	yd_1 = np.clip(y_data[:,1], p_min_1, p_max_1)
	yd = np.stack([yd_0, yd_1], axis=1)
	# y_data = y_data*(1-skip)[:, None]
	# pdb.set_trace()
	y_data = yd

# if task == 'real_A':
# x_data = DATA[:, :task_ofs]
# y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+2] - DATA[:, dt_ofs:dt_ofs+2]
# pdb.set_trace()
if __name__ == "__main__":
	neural_network = BNN(nn_type='2', dropout_p=.1)
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
    