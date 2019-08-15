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




save_path = 'save_model/robotic_hand_real/pytorch/'
model_file = save_path+task + '_heldout' + str(held_out) + '_' + nn_type + '.pkl'

trajectory_path_map = {#'real_A': 'data/robotic_hand_real/A/t42_cyl45_right_test_paths.obj', 
    'real_A': 'data/robotic_hand_real/A/testpaths_cyl35_d_v0.pkl', 
    'real_B': 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl',
    'transferA2B': 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl',
    'transferB2A': 'data/robotic_hand_real/A/testpaths_cyl35_d_v0.pkl',
    'sim_A': 'data/robotic_hand_simulator/A/test_trajectory/jt_rollout_1_v14_d8_m1.pkl', 
    # 'sim_s10' : 'data/robotic_hand_simulator/A/test_trajectory/jt_path'+str(1)+'_v14_m10.pkl'
    }
trajectory_path = trajectory_path_map[task]

datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'

with open(datafile_name, 'rb') as pickle_file:
    trajectories = pickle.load(pickle_file, encoding='latin1')



state_dim = 4
action_dim = 6
alpha = .4

dtype = torch.float
cuda = torch.cuda.is_available()
print('cuda is_available: '+ str(cuda))

mse_fn = torch.nn.MSELoss()

def make_traj():
    state_file = 'problematic/naive_goal68.0_74.0_n38019_traj.txt'
    action_file = 'problematic/naive_goal68.0_74.0_n38019_plan.txt'

    states = np.genfromtxt(state_file, delimiter=',')
    actions = np.genfromtxt(action_file, delimiter=',')

    # pdb.set_trace()
    states = states[:-1]
    s0 = states[0]
    start_states = np.stack([s0]*states.shape[0], 0)
    traj = np.concatenate([states, actions, start_states], -1)

    return traj


def make_rollouts():
    rollout_file = 'problematic/naive_goal68.0_74.0_n38019_plan.pkl'

    with open(rollout_file, 'rb') as pickle_file:
        rollouts = pickle.load(pickle_file, encoding='latin1')

    return rollouts







def run_traj(model, traj, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold=None):
    true_states = traj[:,:state_dim]
    state = traj[0][:state_dim]
    states = []#state.view(1, state_dim)
    # model = model.eval()


    for i, point in enumerate(traj):
        states.append(state)
        action = point[state_dim:state_dim+action_dim]
        if task in ['transferA2B', 'transferB2A']: 
            action*= torch.tensor(np.array([-1,-1,1,1,1,1]),dtype=torch.float)
        # if cuda: action = action.cuda() 
        # pdb.set_trace()   
        inpt = torch.cat((state, action), 0)
        inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

    
        state_delta = model(inpt)
        
        state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
        state= state_delta + state

        if threshold:
            with torch.no_grad():
                mse = mse_fn(state[...,:2], true_states[...,i,:2])
            if mse > threshold:
                states = torch.stack(states, 0)
                return states, False, i

    if threshold:
        with torch.no_grad():
            mse = mse_fn(state[...,:2], true_states[...,i,:2])
        if mse > threshold:
            states = torch.stack(states, 0)
            return states, True, len(states)

    states = torch.stack(states, 0)

    return states

def run_traj(model, traj, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold=None):
    states = model.run_traj(torch.tensor(traj, dtype=dtype), threshold=None)
    # states = model.run_traj(torch.tensor(traj, dtype=dtype), threshold=threshold)
    states = states.squeeze(0)
    # states = states.detach().numpy()
    if threshold:
        minitraj = traj[...,:states.shape[-1]]
        duration = states.shape[-2]

        # if 'transfer' in task: 
            # pdb.set_trace()

        mse_fn = torch.nn.MSELoss()
        # diff = mse_fn(states[...,:2], true_states[...,:2])
        diff = torch.sum((minitraj[...,:2]-states[...,:2])**2, -1)**.5
        if (diff <= threshold).all():
            return states, 1 , duration
        unbounded = diff > threshold
        # pdb.set_trace()
        duration = torch.min(unbounded.nonzero()).item()
    
        finished = duration == traj.shape[0]
        return states, finished, duration
    else: 
        return states

# model_file = save_path+task + '_' + nn_type + '.pkl'

# if len(argv) > 2 and argv[2] != '_': 
#     if 'real' in task: model_file = save_path+task + '_heldout' + str(held_out) +  '_'+ nn_type + '.pkl'
#     if task == 'transferB2A': model_file = save_path+task + '_nonlinear_transform_heldout' + str(held_out) +  '_'+ nn_type + '.pkl'

# if method != '': 
#     model_file = save_path+task + '_' + method + '_heldout' + str(held_out) + '_' + nn_type + '.pkl'


# if bayes:
#     model_file = model_file[:-4] + '_bayesian.pkl'

# model_file = 'save_model/robotic_hand_simulator/pytorch/'+task+'_heldout' + str(held_out) + '_1.pkl'

# model_file = save_path + 'real_' + task[-3] + '_heldout' + str(held_out) + '_' + nn_type + '.pkl'

with open(model_file, 'rb') as pickle_file:
    print("Running " + model_file)
    model = torch.load(pickle_file, map_location='cpu')#.eval()


with open(save_path+'/normalization_arr/normalization_arr', 'rb') as pickle_file:
    x_norm_arr, y_norm_arr = pickle.load(pickle_file)

x_mean_arr, x_std_arr = x_norm_arr[0], x_norm_arr[1]
y_mean_arr, y_std_arr = y_norm_arr[0], y_norm_arr[1]


x_mean_arr = torch.tensor(x_mean_arr, dtype=dtype)
x_std_arr = torch.tensor(x_std_arr, dtype=dtype)
y_mean_arr = torch.tensor(y_mean_arr, dtype=dtype)
y_std_arr = torch.tensor(y_std_arr, dtype=dtype)

max_mses = []
mses = []
threshold = None
# if held_out > .9: 
#     threshold = 10

ground_truth = make_traj()
rollouts = make_rollouts()


gt = torch.tensor(ground_truth, dtype=dtype)
if cuda: 
    gt = gt.cuda()
    x_mean_arr = x_mean_arr.cuda()
    x_std_arr = x_std_arr.cuda()
    y_mean_arr = y_mean_arr.cuda()
    y_std_arr = y_std_arr.cuda()

    model = model.to('cuda')
    model.norm = tuple([n.cuda() for n in model.norm])

states = run_traj(model, gt, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr)
if cuda:
    states = states.cpu()

states = states.detach().numpy()

# max_mse = ((states[:,:2] - ground_truth[:,:2])**2).sum(axis=1).max()
# mse = ((states[:,:2] - ground_truth[:,:2])**2).sum(axis=1).mean()
# print('Maximum drift: ' + str(max_mse))
# print('Average drift: ' + str(mse))
# print('\n')
# max_mses.append(max_mse)
# mses.append(mse)

plt.figure(1)
# for traj in trajectories:
#     plt.plot(traj[:, 0], traj[:, 1], color='yellow')

plt.plot(ground_truth[:, 0], ground_truth[:, 1], color='blue', label='Plan', marker='.', zorder=1)
plt.plot(states[:, 0], states[:, 1], color='green', label='New Model', marker='.', zorder=2)

for r in rollouts:
    plt.plot(r[:, 0], r[:, 1], color='red', zorder=0)

for r in rollouts:
    plt.scatter(r[-1, 0], r[-1, 1], color='black', zorder=10)

plt.scatter(states[-1, 0], states[-1, 1], color='purple', zorder=15)  
plt.axis('scaled')
plt.title('NN Prediction -- pos Space')
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

fig_loc = '/home/liam/results/problematic_path/' + task + '_heldout' + str(held_out) + '_traj_' + str(test_traj) + '.png'
if bayes:
    fig_loc = '/home/liam/results/' + task + '_heldout' + str(held_out)+'_traj_' + str(test_traj) + '_bayesian.png'
# plt.savefig(fig_loc)
# plt.close()
plt.show()
