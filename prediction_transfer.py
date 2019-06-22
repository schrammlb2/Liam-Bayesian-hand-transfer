import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from common.data_normalization import z_score_normalize, z_score_denormalize
from common.build_model import build_model
import pdb


from sys import argv
test_traj=1

config = tf.ConfigProto(device_count={'GPU': 0})

tf.keras.backend.set_floatx('float32')  # for input weights of NN

task = 'real'
transfer_type = 'adaptation'

if len(argv) > 1:
    test_traj = int(argv[1])
if len(argv) > 2:
    transfer_type = argv[2]

assert task in ['real']#, 'sim_s1', 'sim_s10']
assert transfer_type in ['direct','adaptation']

PRE_START = 0  # prediction start point
PRE_END = None  # prediction end point, None means predict whole trajectory
VAR_POS = [0.00001, 0.00001]
VAR_LOAD = [0.00001, 0.00001]
state_dim = 4
if task == 'real':
	act_dim = 6
else:
	act_dim=2


#Set up the appropriate file paths for each of the tasks
direct_loc_map = {'real': ('save_model/robotic_hand_real/A/pos', 'save_model/robotic_hand_real/A/load'),
	'sim_s1' : ('save_model/robotic_hand_simulator/A/d4_s1_pos', 'save_model/robotic_hand_simulator/A/d4_s1_load'),
	'sim_s10' : ('save_model/robotic_hand_simulator/A/d4_s10_pos', 'save_model/robotic_hand_simulator/A/d4_s10_load')
	}
adaptation_loc_map = {'real': ('save_model/robotic_hand_real/Transfer/pos', 'save_model/robotic_hand_real/Transfer/load')#,
    # 'sim_s1' : ('save_model/robotic_hand_simulator/d4_s1_pos', 'save_model/robotic_hand_simulator/d4_s1_load'),
    # 'sim_s10' : ('save_model/robotic_hand_simulator/d4_s10_pos', 'save_model/robotic_hand_simulator/d4_s10_load')
    }
trajectory_path_map = {'real': 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl'
	}
if transfer_type == 'direct': 
    model_loc_map = direct_loc_map
elif transfer_type == 'adaptation':
    model_loc_map = adaptation_loc_map

trajectory_path = trajectory_path_map[task]
pos_model_path = model_loc_map[task][0]  # use to import weights and normalization arrays
load_model_path = model_loc_map[task][1]
#This looks way more complicated than it really is. it's really just an efficient way to list out all the filenames.
with open(trajectory_path, 'rb') as pickle_file:
    trajectory = pickle.load(pickle_file, encoding='latin1')

if task == 'real':
    real_positions = trajectory[0][test_traj]
    acts = trajectory[1][test_traj]
    ground_truth = np.append(real_positions, acts, axis=1)
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
dropout_p = .05     
neural_net_pos = build_model('2', 2, dropout_p)
neural_net_load = build_model('2', 2, dropout_p)

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
    loads = []  # prediction in velocity space
    poses.append(ground_truth[0][:2])
    loads.append(ground_truth[0][2:4])
    state = np.array(ground_truth[0])
    # pdb.set_trace()
    norm_state = z_score_normalize(np.asarray([state]), x_norm_arr[0], x_norm_arr[1])

    print(norm_state)
    for i in range(len(ground_truth)-1):
        (pos_delta, load_delta) = sess.run((y_pos_delta_pre, y_load_delta_pre), feed_dict={x: norm_state})
        pos_delta = z_score_denormalize(pos_delta, y_norm_arr_ang[0], y_norm_arr_ang[1])[0]  # denormalize
        load_delta = z_score_denormalize(load_delta, y_norm_arr_vel[0], y_norm_arr_vel[1])[0]
        # load_delta = ground_truth[i+1, 2:4] - ground_truth[i, 2:4]
        next_pos = state[:2] + pos_delta
        next_load = state[2:4] + load_delta


        # Percentile cutoffs, absolute value, p = 8: 
        p_max_0= 191.90525000000008
        p_min_0= -0.04
        p_max_1= -3.36
        p_min_1= -188.82227027027028



        yd_0 = np.clip(next_load[0], p_min_0, p_max_0)
        yd_1 = np.clip(next_load[1], p_min_1, p_max_1)
        yd = np.stack([yd_0, yd_1], axis=0)
        next_load = yd



        poses.append(next_pos)
        loads.append(next_load)
        state = np.append(np.append(next_pos, next_load), ground_truth[i + 1][state_dim:state_dim+act_dim])
        norm_state = z_score_normalize(np.asarray([state]), x_norm_arr[0], x_norm_arr[1])

#calculate prediction error
data_size = -1

pos_guesses = np.reshape(poses, (data_size, 2))
pos_truths = ground_truth[:, :2]
pos_mse = ((pos_guesses - pos_truths)**2).mean(axis=None)
print("Position MSE: ", pos_mse)
load_guesses = np.reshape(loads, (data_size, 2))
load_truths = ground_truth[:, 2:4]
load_mse = ((load_guesses - load_truths)**2).mean(axis=None)
print("Load MSE: ", load_mse)
state_guesses = np.hstack((pos_guesses, load_guesses))
state_truths = ground_truth[:, :4]
state_mse = ((state_guesses - state_truths)**2).mean(axis=None)
print("State MSE: ", state_mse)

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