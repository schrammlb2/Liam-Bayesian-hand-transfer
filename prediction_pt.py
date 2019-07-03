import scipy.io
# from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb
import torch
from common.data_normalization import *
from common.pt_build_model import *
import matplotlib.pyplot as plt

from sys import argv

task = 'real_A' #Which task we're training. This tells us what file to use
outfile = None
append = False
held_out = .1
test_traj = 1
# _ , arg1, arg2, arg3 = argv
nn_type = '1'
method = ''

if len(argv) > 1 and argv[1] != '_':
    test_traj = int(argv[1])
if len(argv) > 2 and argv[2] != '_':
    task = argv[2]
if len(argv) > 3 and argv[3] != '_':
    nn_type = argv[3]
if len(argv) > 4 and argv[4] != '_' and 'transfer' in task:
    method = 'retrain'
    method = argv[4]

assert task in ['real_A', 'real_B','transferA2B','transferB2A']



save_path = 'save_model/robotic_hand_real/pytorch/'

# model_loc_map = {'real_A': ('save_model/robotic_hand_real/A/pos', 'save_model/robotic_hand_real/A/load'),
#     'real_B': ('save_model/robotic_hand_real/B/pos', 'save_model/robotic_hand_real/B/load'),
#     'sim_s1' : ('save_model/robotic_hand_simulator/A/d4_s1_pos', 'save_model/robotic_hand_simulator/A/d4_s1_load'),
#     'sim_s10' : ('save_model/robotic_hand_simulator/A/d4_s10_pos', 'save_model/robotic_hand_simulator/A/d4_s10_load')
#     }
trajectory_path_map = {#'real_A': 'data/robotic_hand_real/A/t42_cyl45_right_test_paths.obj', 
    'real_A': 'data/robotic_hand_real/A/testpaths_cyl35_d_v0.pkl', 
    'real_B': 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl',
    'transferA2B': 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl',
    'transferB2A': 'data/robotic_hand_real/A/testpaths_cyl35_d_v0.pkl',
    # 'sim_s1': 'data/robotic_hand_simulator/A/test_trajectory/jt_rollout_1_v14_d8_m1.pkl', 
    # 'sim_s10' : 'data/robotic_hand_simulator/A/test_trajectory/jt_path'+str(1)+'_v14_m10.pkl'
    }
trajectory_path = trajectory_path_map[task]

with open(trajectory_path, 'rb') as pickle_file:
    trajectory = pickle.load(pickle_file, encoding='latin1')

def make_traj(trajectory, test_traj):
    real_positions = trajectory[0][test_traj]
    acts = trajectory[1][test_traj]

    if task in ['real_B', 'transferA2B'] and test_traj == 0:
        start = 199
        real_positions = trajectory[0][test_traj][start:]
        acts = trajectory[1][test_traj][start:]
    
    return np.append(real_positions, acts, axis=1)


if task in ['real_A', 'real_B', 'transferA2B', 'transferB2A']:
    ground_truth = make_traj(trajectory, test_traj)


state_dim = 4
action_dim = 6
alpha = .4

dtype = torch.float
cuda = False#torch.cuda.is_available()
print('cuda is_available: '+ str(cuda))

def run_traj(model, traj, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr):
    true_states = traj[:,:state_dim]
    state = traj[0][:state_dim]
    states = []#state.view(1, state_dim)

    for i, point in enumerate(traj):
        states.append(state)
        action = point[state_dim:state_dim+action_dim]
        # if cuda: action = action.cuda()    
        inpt = torch.cat((state, action), 0)
        inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

        state_delta = model(inpt)
        # if task in ['transferA2B', 'transferB2A']: 
        # if task in ['real_A']:state_delta *= torch.tensor([-1,-1,1,1], dtype=dtype)
        
        state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
        state= state_delta + state
        #May need random component here to prevent overfitting
        # states = torch.cat((states,state.view(1, state_dim)), 0)
                # return mse_fn(states[:,:2], true_states[:states.shape[0],:2])
    states = torch.stack(states, 0)
    return states



# model = pt_build_model('0', state_dim+action_dim, state_dim, .1)
model_file = save_path+task + '_' + nn_type + '.pkl'

if method != '': 
    model_file = save_path+task + '_' + method + '_' + nn_type + '.pkl'
    if method == 'constrained_restart':
        if len(argv) > 5:
            l2_coeff = argv[5]
            model_file = save_path+task + '_' + method + '_' + str(float(l2_coeff))  + '_' + nn_type+ '.pkl'
        else:
            model_file = save_path+task + '_'  + nn_type + '.pkl'


with open(model_file, 'rb') as pickle_file:
    print("Running " + model_file)
    model = torch.load(pickle_file, map_location='cpu')


with open(save_path+'/normalization_arr/normalization_arr', 'rb') as pickle_file:
    x_norm_arr, y_norm_arr = pickle.load(pickle_file)

x_mean_arr, x_std_arr = x_norm_arr[0], x_norm_arr[1]
y_mean_arr, y_std_arr = y_norm_arr[0], y_norm_arr[1]

# x_mean_arr = torch.tensor(x_mean_arr)
# x_std_arr = torch.tensor(x_std_arr)
# y_mean_arr = torch.tensor(y_mean_arr)
# y_std_arr = torch.tensor(y_std_arr)
x_mean_arr = torch.tensor(x_mean_arr, dtype=dtype)
x_std_arr = torch.tensor(x_std_arr, dtype=dtype)
y_mean_arr = torch.tensor(y_mean_arr, dtype=dtype)
y_std_arr = torch.tensor(y_std_arr, dtype=dtype)
# out = [z_score_normalize]
# if cuda:
#     x_mean_arr = x_mean_arr.cuda()
#     x_std_arr = x_std_arr.cuda()
#     y_mean_arr = y_mean_arr.cuda()
#     y_std_arr = y_std_arr.cuda()

max_mses = []
mses = []
for test_traj in range(4):
    ground_truth = make_traj(trajectory, test_traj)

    states = run_traj(model, torch.tensor(ground_truth, dtype=dtype), x_mean_arr, x_std_arr, y_mean_arr, y_std_arr).detach().numpy()

    max_mse = ((states[:,:2] - ground_truth[:,:2])**2).sum(axis=1).max()
    mse = ((states[:,:2] - ground_truth[:,:2])**2).sum(axis=1).mean()
    print('Maximum drift: ' + str(max_mse))
    print('Average drift: ' + str(mse))
    print('\n')
    max_mses.append(max_mse)
    mses.append(mse)

    plt.figure(1)
    plt.scatter(ground_truth[0, 0], ground_truth[0, 1], marker="*", label='start')
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], color='blue', label='Ground Truth', marker='.')
    plt.plot(states[:, 0], states[:, 1], color='red', label='NN Prediction')
    plt.axis('scaled')
    plt.title('Bayesian NN Prediction -- pos Space')
    plt.legend()

    fig_loc = '/home/liam/results/recurrent_network_results/'
    if 'transfer' in task: 
        # method = 
        fig_loc += 'transfer/'
        fig_loc += method + '/'
    if task == 'real_B':
        task_str = 'real_b'
    elif task == 'real_A':
        task_str = 'real_a'
    else: task_str = task

    fig_loc += task_str + '_pretrain_batch/'
    fig_loc += 'traj' + str(test_traj) + '.png'

    # plt.savefig(fig_loc)
    # plt.close()
    plt.show()

