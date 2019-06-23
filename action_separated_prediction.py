import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from common.data_normalization import z_score_normalize, z_score_denormalize
from common.build_model import build_model
import pdb


from sys import argv
test_traj=2

config = tf.ConfigProto(device_count={'GPU': 0})

tf.keras.backend.set_floatx('float32')  # for input weights of NN

task = 'real_B'
nn_type = '3'


if len(argv) > 1:
	test_traj = int(argv[1])

assert task in ['real_A', 'real_B', 'sim_s1', 'sim_s10']

PRE_START = 0  # prediction start point
PRE_END = None  # prediction end point, None means predict whole trajectory
VAR_POS = [0.00001, 0.00001]
VAR_LOAD = [0.00001, 0.00001]
state_dim = 4
if task in ['real_A', 'real_B']:
	act_dim = 6
else:
	act_dim=2


def get_action_index(action):
    action_list = [np.asarray([1.5,0]), np.asarray([-1.5,0]), np.asarray([0, 1.5]), np.asarray([0, -1.5]), 
        np.asarray([1, 1]), np.asarray([1, -1]), np.asarray([-1, 1]), np.asarray([-1, -1])]
    for i, act in enumerate(action_list):
        if np.isclose(action,act).all():
            return i
    print("Action is outside scope of networks")
    print(action)
    assert False



#Set up the appropriate file paths for each of the tasks
model_loc_map = {'real_A': ('save_model/robotic_hand_real/A/pos', 'save_model/robotic_hand_real/A/load'),
    'real_B': ('save_model/robotic_hand_real/B/pos', 'save_model/robotic_hand_real/B/load'),
	'sim_s1' : ('save_model/robotic_hand_simulator/A/d4_s1_pos', 'save_model/robotic_hand_simulator/A/d4_s1_load'),
	'sim_s10' : ('save_model/robotic_hand_simulator/A/d4_s10_pos', 'save_model/robotic_hand_simulator/A/d4_s10_load')
	}
trajectory_path_map = {#'real_A': 'data/robotic_hand_real/A/t42_cyl45_right_test_paths.obj', 
	'real_A': 'data/robotic_hand_real/A/testpaths_cyl35_d_v0.pkl', 
    'real_B': 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl',
	'sim_s1': 'data/robotic_hand_simulator/A/test_trajectory/jt_rollout_1_v14_d8_m1.pkl', 
	'sim_s10' : 'data/robotic_hand_simulator/A/test_trajectory/jt_path'+str(1)+'_v14_m10.pkl'
	}
trajectory_path = trajectory_path_map[task]
pos_model_path = model_loc_map[task][0]  # use to import weights and normalization arrays
load_model_path = model_loc_map[task][1]
#This looks way more complicated than it really is. it's really just an efficient way to list out all the filenames.
with open(trajectory_path, 'rb') as pickle_file:
    trajectory = pickle.load(pickle_file, encoding='latin1')

if task in ['real_A', 'real_B']:
    real_positions = trajectory[0][test_traj]
    acts = trajectory[1][test_traj]

    if task == 'real_B' and test_traj == 0:
        start = 199
        real_positions = trajectory[0][test_traj][start:]
        acts = trajectory[1][test_traj][start:]


    
    ground_truth = np.append(real_positions, acts, axis=1)

    # p_max_0= 226.28925614035072
    # p_min_0= 2.12
    # p_max_1= -0.1648863809669727
    # p_min_1= -199.26


    # yd_0 = np.clip(ground_truth[:,2], p_min_0, p_max_0)
    # yd_1 = np.clip(ground_truth[:,3], p_min_1, p_max_1)
    # yd = np.stack([yd_0, yd_1], axis=1)
    # ground_truth = np.concatenate([ground_truth[:,0:2], yd, ground_truth[:,4:]], axis=1)

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
neural_net_pos = [build_model(nn_type, 2, dropout_p)for i in range(8)]
neural_net_load = [build_model(nn_type, 2, dropout_p)for i in range(8)]


x = tf.placeholder(tf.float32, shape=[None, state_dim+act_dim])

# y_pos_mean = neural_net_pos(x)
# y_load_mean = neural_net_load(x)
# y_pos_distribution = tfp.distributions.Normal(loc=y_pos_mean, scale=VAR_POS)
# y_load_distribution = tfp.distributions.Normal(loc=y_load_mean, scale=VAR_LOAD)
# y_pos_delta_pre = y_pos_distribution.sample()
# y_load_delta_pre = y_load_distribution.sample()

# Condensed the above 3 lines (per network) into one line, so I can more easily store them all in a list.
# Functionality is the same, just now there's one operation for each action
y_pos_delta_pre = [tfp.distributions.Normal(loc=nn(x), scale=VAR_POS).sample() for nn in neural_net_pos]
y_load_delta_pre = [tfp.distributions.Normal(loc=nn(x), scale=VAR_LOAD).sample() for nn in neural_net_load]





with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i, net in enumerate(neural_net_pos):
        net.load_weights(pos_model_path+"/weights/BNN_weights" + str(i))  # load NN parameters
    for i, net in enumerate(neural_net_load):        
        net.load_weights(load_model_path+"/weights/BNN_weights" + str(i))
    poses = []  # prediction in angle space
    loads = []  # prediction in velocity space
    poses.append(ground_truth[0][:2])
    loads.append(ground_truth[0][2:4])
    state = np.array(ground_truth[0])
    norm_state = z_score_normalize(np.asarray([state]), x_norm_arr[0], x_norm_arr[1])

    print(norm_state)
    for i in range(len(ground_truth)-1):
        action = ground_truth[i + 1][state_dim:state_dim+2]
        ind = get_action_index(action)
        (pos_delta, load_delta) = sess.run((y_pos_delta_pre[ind], y_load_delta_pre[ind]), feed_dict={x: norm_state})
        pos_delta = z_score_denormalize(pos_delta, y_norm_arr_ang[0], y_norm_arr_ang[1])[0]  # denormalize
        load_delta = z_score_denormalize(load_delta, y_norm_arr_vel[0], y_norm_arr_vel[1])[0]


        load_delta = ground_truth[i+1, 2:4] - ground_truth[i, 2:4]
        # load_delta = 0
        next_pos = state[:2] + pos_delta
        next_load = state[2:4] + load_delta
        # next_load = ground_truth[i+1, 2:4]

        

        # p_max_0= 217.74
        # p_min_0= 3.9091999999999993
        # p_max_1= -1.9529632352941166
        # p_min_1= -190.01

        # p_max_0= 203.56
        # p_min_0= 7.6
        # p_max_1= -5.12
        # p_min_1= -173.12

        # p_max_0= 197.1276000000001
        # p_min_0= 8.74
        # p_max_1= -6.719999999999999
        # p_min_1= -166.4

        p_max_0= 183.10679999999994
        p_min_0= 10.1
        p_max_1= -9.6
        p_min_1= -154.1234

        # p_max_0= 172.48
        # p_min_0= 11.1
        # p_max_1= -11.98
        # p_min_1= -144.86

        # p_max_0= 149.95
        # p_min_0= 13.19
        # p_max_1= -17.08
        # p_min_1= -125.85

        yd_0 = np.clip(next_load[0], p_min_0, p_max_0)
        yd_1 = np.clip(next_load[1], p_min_1, p_max_1)
        yd = np.stack([yd_0, yd_1], axis=0)
        # next_load = yd

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