import scipy.io
# from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb
import torch
from common.data_normalization import *
from common.pt_build_model import *
from common.utils_clean_traj import *
import matplotlib.pyplot as plt

from sys import argv

task = 'real_A' #Which task we're training. This tells us what file to use
outfile = None
append = False
held_out = .1
test_traj = 1
bayes = False
# _ , arg1, arg2, arg3 = argv
nn_type = '1'
# nn_type = 'LSTM'
method = ''


if len(argv) > 1 and argv[1] != '_':
    task = argv[1]
if len(argv) > 2 and argv[2] != '_':
    held_out = float(argv[2])
if len(argv) > 3 and argv[3] != '_' and 'transfer' in task:
    method = 'retrain'
    method = argv[3]
if len(argv) > 4 and argv[4] != '_':
    bayes = bool(argv[4])
if len(argv) > 5 and argv[5] != '_':
    nn_type = argv[5]

# assert task in ['real_A', 'real_B','transferA2B','transferB2A']


save_path = 'save_model/robotic_hand_real/pytorch/'
if 'acrobot' in task:
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
    # 'sim_A': 'data/robotic_hand_simulator/A/test_trajectory/jt_rollout_1_v14_d8_m1.pkl', 
    # 'sim_B': 'data/robotic_hand_simulator/A/test_trajectory/jt_rollout_1_v14_d8_m1.pkl',,
    'sim_A': 'data/robotic_hand_simulator/A/sim_data_discrete_v14_d4_m1_episodes.obj', 
    'sim_B' : 'data/robotic_hand_simulator/B/sim_data_discrete_v14_d4_m1_modified_episodes.obj',
    'sim_cont': 'data/robotic_hand_simulator/sim_data_cont_v0_d4_m1_episodes.obj',
    'sim_transferA2B': 'data/robotic_hand_simulator/B/sim_data_discrete_v14_d4_m1_modified_episodes.obj', 
    'sim_transferB2A' : 'data/robotic_hand_simulator/A/sim_data_discrete_v14_d4_m1_episodes.obj',
    # 'sim_s10' : 'data/robotic_hand_simulator/A/test_trajectory/jt_path'+str(1)+'_v14_m10.pkl'
    'acrobot_transferA2B': 'data/acrobot/heavy_acrobot/acrobot_data.pkl',
    'acrobot_transferB2A': 'data/acrobot/long_acrobot/acrobot_data.pkl',
    'acrobot_B': 'data/acrobot/heavy_acrobot/acrobot_data.pkl',
    'acrobot_A': 'data/acrobot/long_acrobot/acrobot_data.pkl'
    }
trajectory_path = trajectory_path_map[task]

with open(trajectory_path, 'rb') as pickle_file:
    trajectory = pickle.load(pickle_file, encoding='latin1')



state_dim = 4
action_dim = 6
alpha = .4

dtype = torch.float
cuda = torch.cuda.is_available()
# cuda = False
print('cuda is_available: '+ str(cuda))

mse_fn = torch.nn.MSELoss()

def make_traj(trajectory, test_traj):
    real_positions = trajectory[0][test_traj]
    acts = trajectory[1][test_traj]

    if task in ['real_B', 'transferA2B'] and test_traj == 0:
        start = 199
        real_positions = trajectory[0][test_traj][start:]
        acts = trajectory[1][test_traj][start:]
    

    return np.append(real_positions[:len(acts)], acts, axis=1)

# def make_traj_sim(trajectory):
#     real_positions = trajectory[0][...,:4]
#     acts = trajectory[1]
    
#     return np.append(real_positions[:len(acts)], acts, axis=1)
def make_traj_sim(trajectory, test_traj):
    # out = list(filter(lambda x: len(x) > length_threshold, trajectory))
    # pdb.set_trace()
    traj = trajectory
    if 'acrobot' in task:
        traj = trajectory[:-2]
    # if 'sim' in task:
    # out = clean_data(traj, cutoff = .05, cut = 'Absolute')
    out = clean_data(traj, cutoff = 99.6)
    out = out[-test_traj][...,:6]
    # out = clean_data(out)
    return out

def transfer(x, state_dim): 
    return torch.cat((x[...,:state_dim], x[...,state_dim:state_dim+2]*-1,  x[...,state_dim+2:]), -1) 

if task in ['real_A', 'real_B', 'transferA2B', 'transferB2A']:
    ground_truth = make_traj(trajectory, test_traj)


# task  = 'transferB2A_traj_transfer'

model_file = save_path+task + '_' + nn_type + '.pkl'



if 'transfer' in task: 
    print('method = ' + method)
    if method == '':
        model_file = save_path+task + '_heldout' + str(held_out) +  '_'+ nn_type + '.pkl'
    elif method == 'direct':
        model_file = save_path + 'real_' + task[-3] + '_heldout' + str(held_out) + '_' + nn_type + '.pkl'
    else:
        model_file = save_path+task + '_' + method + '_heldout' + str(held_out) + '_' + nn_type + '.pkl'
# if 'real' in task:
else: 
    model_file = save_path+task + '_heldout' + str(held_out) +  '_'+ nn_type + '.pkl'


# if bayes:
#     model_file = model_file[:-4] + '_bayesian.pkl'

# model_file = 'save_model/robotic_hand_real/pytorch/transferB2A_traj_transfer_heldout0.1_1.pkl'
# model_file = 'save_model/robotic_hand_real/pytorch/real_A_heldout0.1_1.pkl'

# model_file = save_path + 'real_' + task[-3] + '_heldout' + str(held_out) + '_' + nn_type + '.pkl'

with open(model_file, 'rb') as pickle_file:
    print("Running " + model_file)
    model = torch.load(pickle_file, map_location='cpu')#.eval()


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



method_rename_dict = {'traj_transfer_timeless' : 'cumulative residual', 
    'traj_transfer_timeless_recurrent' : 'recurrent residual', 
    'retrain': 'trajectory fine-tuning',
    'retrain_naive' : 'naive fine-tuning',
    'direct' : 'direct transfer'
    }
def rename(string):
    if string in method_rename_dict.keys():
        return method_rename_dict[string]
    else:
        return string

max_mses = []
mses = []
threshold = None

for test_traj in range(100):
    if 'sim' in task or 'acrobot' in task:
        # ground_truth = make_traj_sim(trajectory)
        ground_truth = make_traj_sim(trajectory, test_traj)
        # pdb.set_trace()
    else:
        ground_truth = make_traj(trajectory, test_traj)

    # ground_truth = ground_truth[:len(ground_truth)//4]
    # ground_truth = ground_truth[...,:6]
    # if method in ['direct', 'retrain']:
    #     model.task = 'transferA2B'
    model.task = task
    # pdb.set_trace()

    model.coeff = .3
    # model.coeff = -.3#.45
    # if method == 'traj_transfer_timeless':
    #     model.coeff = .45

    gt = torch.tensor(ground_truth, dtype=dtype)

    if cuda: 
        gt = gt.cuda()
        x_mean_arr = x_mean_arr.cuda()
        x_std_arr = x_std_arr.cuda()
        y_mean_arr = y_mean_arr.cuda()
        y_std_arr = y_std_arr.cuda()

        model = model.to('cuda')
        model.norm = tuple([n.cuda() for n in model.norm])

    # pdb.set_trace()
    if task[-1] == 'A':
        gt = transfer(gt, state_dim)

    model.eval()

    states = model.run_traj(gt, threshold=threshold)
    states = states.squeeze(0)
    if cuda:
        states = states.cpu()
    states = states.detach().numpy()
    duration = states.shape[-2]
    finished = duration == ground_truth.shape[0]


    plt.figure(1)
    plt.scatter(ground_truth[0, 0], ground_truth[0, 1], marker="*", label='start')
    # plt.plot(ground_truth[:duration, 0], ground_truth[:duration, 1], color='blue', label='Ground Truth', marker='.')
    # plt.plot(states[:, 0], states[:, 1], color='red', label='NN Prediction')
    plt.plot(ground_truth[:duration, 2], ground_truth[:duration, 3], color='blue', label='Ground Truth', marker='.')
    plt.plot(states[:, 2], states[:, 3], color='red', label='NN Prediction')

    plt.axis('scaled')
    plt.title('Duration: ' + str(duration) + '.\nFinished: ' + str(finished))
    plt.legend()

    fig_loc = '/home/liam/results/recurrent_network_results/'
    if 'transfer' in task: 
        # method = 
        fig_loc += 'transfer/'
        fig_loc += method + '_'#+ '/'
    if task == 'real_B':
        task_str = 'real_b'
    elif task == 'real_A':
        task_str = 'real_a'
    else: task_str = task

    # fig_loc += task_str + '_pretrain_batch/'
    fig_loc += task_str + '_pretrain_batch_'
    fig_loc += 'traj' + str(test_traj) + '.png'

    # fig_loc = '/home/liam/results/' + task + '_heldout.95_traj_' + str(test_traj) + '.png'
    # if bayes:
    #     fig_loc = '/home/liam/results/' + task + '_heldout' + str(held_out)+'_traj_' + str(test_traj) + '_bayesian.png'
    # plt.savefig(fig_loc)
    # plt.close()
    plt.show()