import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from common.data_normalization import z_score_normalize, z_score_denormalize
import pdb


from sys import argv
test_traj=2

config = tf.ConfigProto(device_count={'GPU': 0})

tf.keras.backend.set_floatx('float32')  # for input weights of NN

task = 'real_A'



if len(argv) > 1:
    test_traj = int(argv[1])

assert task in ['real_A', 'sim_s1', 'sim_s10']

PRE_START = 0  # prediction start point
PRE_END = None  # prediction end point, None means predict whole trajectory
VAR_POS = [0.00001, 0.00001]
VAR_LOAD = [0.00001, 0.00001]
state_dim = 6
act_dim = 6

alpha = .4


#Set up the appropriate file paths for each of the tasks
model_loc_map = {'real_A': ('save_model/robotic_hand_real/episodic/pos', 'save_model/robotic_hand_real/episodic/load'),
	'sim_s1' : ('save_model/robotic_hand_simulator/episodic/d4_s1_pos', 'save_model/robotic_hand_simulator/episodic/d4_s1_load'),
	'sim_s10' : ('save_model/robotic_hand_simulator/episodic/d4_s10_pos', 'save_model/robotic_hand_simulator/episodic/d4_s10_load')
	}
trajectory_path_map = {#'real_A': 'data/robotic_hand_real/episodic/t42_cyl45_right_test_paths.obj', 
	'real_A': 'data/robotic_hand_real/A/testpaths_cyl35_d_v0.pkl', 
	'sim_s1': 'data/robotic_hand_simulator/A/test_trajectory/jt_rollout_1_v14_d8_m1.pkl', 
	'sim_s10' : 'data/robotic_hand_simulator/A/test_trajectory/jt_path'+str(1)+'_v14_m10.pkl'
	}
trajectory_path = trajectory_path_map[task]
pos_model_path = model_loc_map[task][0]  # use to import weights and normalization arrays
load_model_path = model_loc_map[task][1]
#This looks way more complicated than it really is. it's really just an efficient way to list out all the filenames.
with open(trajectory_path, 'rb') as pickle_file:
    trajectory = pickle.load(pickle_file, encoding='latin1')

if task == 'real_A':
    real_positions = trajectory[0][test_traj]
    acts = trajectory[1][test_traj]

    x_ave = real_positions[0, 2:4]*0
    x_ave_list_episode = []
    for i in range(len(real_positions)):
        x_ave *= alpha

        x_ave += real_positions[i, 2:4]*(1-alpha) 
        x_ave_list_episode.append(x_ave)
    x_stack = np.stack(x_ave_list_episode, axis=0)  
    # pdb.set_trace()

    # ground_truth = np.append([real_positions, x_stack, acts], axis=1)

    ground_truth = np.append(real_positions, x_stack, axis=1)
    ground_truth = np.append(ground_truth, acts, axis=1)
else:
	real_positions = trajectory[0][:, :4]
	if task == 'sim_s1':
		acts = trajectory[1]
	elif task == 'sim_s10':
		acts = trajectory[2]

	ground_truth = np.asarray(real_positions[:-1])
	ground_truth = np.append(ground_truth, np.asarray(acts), axis=1)


with open(pos_model_path+'/normalization_arr/normalization_arr', 'rb') as pickle_file:
    x_norm_arr, y_norm_arr_ang = pickle.load(pickle_file)

with open(load_model_path+'/normalization_arr/normalization_arr', 'rb') as pickle_file:
    y_norm_arr_vel = pickle.load(pickle_file)[1]

'''Neural net structure'''
dropout_p1 = .1 
dropout_p2 = .1     

layer_num = 64

neural_net_pos = tf.keras.Sequential([
    tfp.layers.DenseFlipout(layer_num, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.05),
    tfp.layers.DenseFlipout(layer_num, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.05),
    tfp.layers.DenseFlipout(2),
])

neural_net_load = tf.keras.Sequential([
    tfp.layers.DenseFlipout(layer_num, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.05),
    tfp.layers.DenseFlipout(layer_num, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.05),
    tfp.layers.DenseFlipout(2),
])

neural_net_pos = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p1),
    tf.keras.layers.Dense(64, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p1),
    tf.keras.layers.Dense(32, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p1),
    tf.keras.layers.Dense(32, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p1),
    tf.keras.layers.Dense(16, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p1),
    tf.keras.layers.Dense(16, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p1),
    tf.keras.layers.Dense(16, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p1),
    tf.keras.layers.Dense(16, activation=tf.nn.selu),
    # tf.keras.layers.AlphaDropout(rate=dropout_p1),
    # tfp.layers.DenseFlipout(self.output_dim),

    tf.keras.layers.AlphaDropout(rate=dropout_p1),
    tf.keras.layers.Dense(2)
])

neural_net_load = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p2),
    tf.keras.layers.Dense(64, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p2),
    tf.keras.layers.Dense(32, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p2),
    tf.keras.layers.Dense(32, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p2),
    tf.keras.layers.Dense(16, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p2),
    tf.keras.layers.Dense(16, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p2),
    tf.keras.layers.Dense(16, activation=tf.nn.selu),
    tf.keras.layers.AlphaDropout(rate=dropout_p2),
    tf.keras.layers.Dense(16, activation=tf.nn.selu),
    # tf.keras.layers.AlphaDropout(rate=dropout_p2),
    # tfp.layers.DenseFlipout(self.output_dim),

    tf.keras.layers.AlphaDropout(rate=dropout_p2),
    tf.keras.layers.Dense(2)
])


x = tf.placeholder(tf.float32, shape=[None, state_dim+act_dim])

y_pos_mean = neural_net_pos(x)
y_load_mean = neural_net_load(x)
y_pos_distribution = tfp.distributions.Normal(loc=y_pos_mean, scale=VAR_POS)
y_load_distribution = tfp.distributions.Normal(loc=y_load_mean, scale=VAR_LOAD)
y_pos_delta_pre = y_pos_distribution.sample()
y_load_delta_pre = y_load_distribution.sample()


with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    neural_net_pos.load_weights(pos_model_path+"/weights/BNN_weights")  # load NN parameters
    neural_net_load.load_weights(load_model_path+"/weights/BNN_weights")
    poses = []  # prediction in angle space
    loads = []  # prediction in load space
    ave_loads = []  # prediction in averaged load space
    poses.append(ground_truth[0][:2])
    loads.append(ground_truth[0][2:4])
    ave_loads.append(ground_truth[0][4:6])
    state = np.array(ground_truth[0])
    norm_state = z_score_normalize(np.asarray([state]), x_norm_arr[0], x_norm_arr[1])

    print(norm_state)
    alpha = .5
    for i in range(len(ground_truth)-1):
        # pdb.set_trace()
        (pos_delta, load_delta) = sess.run((y_pos_delta_pre, y_load_delta_pre), feed_dict={x: norm_state})
        pos_delta = z_score_denormalize(pos_delta, y_norm_arr_ang[0], y_norm_arr_ang[1])[0]  # denormalize
        load_delta = z_score_denormalize(load_delta, y_norm_arr_vel[0], y_norm_arr_vel[1])[0]
        # load_delta = ground_truth[i+1, 2:4] - ground_truth[i, 2:4]
        next_pos = state[:2] + pos_delta
        next_load = state[2:4] + load_delta
        next_ave_load = state[2:4]*alpha + load_delta*(1-alpha)
        poses.append(next_pos)
        loads.append(next_load)
        ave_loads.append(next_ave_load)
        state = np.append(np.concatenate([next_pos, next_load, next_ave_load]), ground_truth[i + 1][state_dim:state_dim+act_dim])
        norm_state = z_score_normalize(np.asarray([state]), x_norm_arr[0], x_norm_arr[1])
        

poses = np.asarray(poses)
loads = np.asarray(loads)

plt.figure(1)
plt.scatter(ground_truth[0, 0], ground_truth[0, 1], marker="*", label='start')
plt.plot(ground_truth[:, 0], ground_truth[:, 1], color='blue', label='Ground Truth', marker='.')
plt.plot(poses[:, 0], poses[:, 1], color='red', label='NN Prediction')
plt.axis('scaled')
plt.title('Bayesian NN Prediction -- pos Space')
plt.legend()
plt.show()

plt.figure(2)
plt.scatter(ground_truth[0, 2], ground_truth[0, 3], marker="*", label='start')
plt.plot(ground_truth[:, 2], ground_truth[:, 3], color='blue', label='Ground Truth', marker='.')
plt.plot(loads[:, 0], loads[:, 1], color='red', label='NN Prediction')
plt.axis('scaled')
plt.title('Bayesian NN Prediction -- load Space')
plt.legend()
plt.show()

plt.figure(2)
plt.scatter(ground_truth[0, 4], ground_truth[0, 5], marker="*", label='start')
plt.plot(ground_truth[:, 4], ground_truth[:, 5], color='blue', label='Ground Truth', marker='.')
plt.plot(ave_loads[:, 0], ave_loads[:, 1], color='red', label='NN Prediction')
plt.axis('scaled')
plt.title('Bayesian NN Prediction -- running average load Space')
plt.legend()
plt.show()