import scipy.io
# from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb
import torch
from common.data_normalization import *
from common.sim_build_model import pt_build_model
import matplotlib.pyplot as plt
import matplotlib as mpl

from sys import argv

task = 'sim_cont' #Which task we're training. This tells us what file to use
append_name = 'trajF_bs512_model512'
skip_step = 1
outfile = None
append = False
test_traj = -1
# _ , arg1, arg2, arg3 = argv
nn_type = '0'

if len(argv) > 1:
    test_traj = int(argv[1])


assert task in ['real_A', 'real_B','sim_A', 'sim_B', 'sim_cont', 'sim_transferA2B', 'sim_transferB2A']
assert skip_step in [1,10]
if (len(argv) > 3 and task != 'sim_A'):
    print('Err: Skip step only appicable to sim_A task. Do not use this argument for other tasks')
    exit(1)


if task == 'real_A' or task == 'real_B':
    save_path = 'save_model/robotic_hand_real/pytorch'
else:
    save_path = 'save_model/robotic_hand_simulator/pytorch'

trajectory_path_map = {#'real_A': 'data/robotic_hand_real/A/t42_cyl45_right_test_paths.obj', 
    'real_A': 'data/robotic_hand_real/A/testpaths_cyl35_d_v0.pkl', 
    'real_B': 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl',
    'sim_A': 'data/robotic_hand_simulator/A/sim_data_discrete_v14_d4_m1_episodes.obj', 
    'sim_B' : 'data/robotic_hand_simulator/B/sim_data_discrete_v14_d4_m1_modified_episodes.obj',
    'sim_cont': 'data/robotic_hand_simulator/sim_data_cont_v0_d4_m1_episodes.obj',
    'sim_transferA2B': 'data/robotic_hand_simulator/B/sim_data_discrete_v14_d4_m1_modified_episodes.obj', 
    'sim_transferB2A' : 'data/robotic_hand_simulator/A/sim_data_discrete_v14_d4_m1_episodes.obj'
    }
trajectory_path = trajectory_path_map[task]

with open(trajectory_path, 'rb') as pickle_file:
    trajectory = pickle.load(pickle_file, encoding='latin1')

"""remove one-point trajectories"""
temp = []
for i in range(len(trajectory)):
    if len(trajectory[i]) > 1:
        temp.append(trajectory[i])
trajectory = temp

"""remove calibration point"""
if task in ['sim_A', 'sim_B']:
    mod = []
    for i in range(len(trajectory)):
        mod = trajectory[i]
        mod = mod[1:]
        trajectory[i] = mod

if task in ['real_A', 'real_B']:
    real_positions = trajectory[0][test_traj]
    acts = trajectory[1][test_traj]

    if task == 'real_B' and test_traj == 0:
        start = 199
        real_positions = trajectory[0][test_traj][start:]
        acts = trajectory[1][test_traj][start:]


    
    ground_truth = np.append(real_positions, acts, axis=1)
    
else:
    ground_truth = np.asarray(trajectory[test_traj])

state_dim = 4
action_dim = 2 if ('sim' in task) else 6
alpha = .4

dtype = torch.float
cuda = False#torch.cuda.is_available()
print('cuda is_available: '+ str(cuda))

def run_traj(model, traj, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr):
    true_states = traj[:,:state_dim]
    state = traj[0][:state_dim]
    states = []#state.view(1, state_dim)
    # if cuda:
    #     state = state.cuda()
    #     true_states = true_states.cuda()

    mse_fn = torch.nn.MSELoss()#reduction='none')
    threshold=100

    for i, point in enumerate(traj):
        states.append(state)
        action = point[state_dim:state_dim+action_dim]
        # if cuda: action = action.cuda()    
        inpt = torch.cat((state, action), 0)
        inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

        state_delta = model(inpt)
        # if task == 'real_B': state_delta *= torch.tensor([-1,1,1,1], dtype=dtype)
        
        state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
        state= state_delta + state
        #May need random component here to prevent overfitting
        # states = torch.cat((states,state.view(1, state_dim)), 0)
                # return mse_fn(states[:,:2], true_states[:states.shape[0],:2])
    states = torch.stack(states, 0)
    return states



model = pt_build_model(nn_type, state_dim+action_dim, state_dim, .1)

with open(save_path + '/' + task + '_' + append_name, 'rb') as pickle_file:
    model = torch.load(pickle_file).cpu()
# if cuda: 
#     model = model.cuda()

with open(save_path+'/normalization_arr/normalization_arr_' + task + '_' + append_name, 'rb') as pickle_file:
    x_norm_arr, y_norm_arr = pickle.load(pickle_file)

x_mean_arr, x_std_arr = x_norm_arr[0], x_norm_arr[1]
y_mean_arr, y_std_arr = y_norm_arr[0], y_norm_arr[1]

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

if __name__ == "__main__":

    states = run_traj(model, torch.tensor(ground_truth, dtype=dtype), x_mean_arr, x_std_arr, y_mean_arr, y_std_arr).detach().numpy()

pos_mse = ((states[:, :2] - ground_truth[:, :2])**2).mean(axis=None)
load_mse = ((states[:, 2:4] - ground_truth[:, 2:4])**2).mean(axis=None)
state_mse = ((states - ground_truth[:, :4])**2).mean(axis=None)

mpl.rc('font', family='Cambria')
plt.subplot(121)#121v,211h
plt.scatter(ground_truth[0, 0], ground_truth[0, 1], marker="*", s=100, color='tab:green', label='Start')
plt.plot(ground_truth[:, 0], ground_truth[:, 1], color='tab:blue', label='Ground Truth')
plt.plot(states[:, 0], states[:, 1], color='tab:red', label='NN Prediction')
plt.grid(True, color='silver')
plt.axis('scaled')
plt.title('Position')
#plt.legend()
#plt.show()
plt.subplot(122)#122v,212h
plt.scatter(ground_truth[0, 2], ground_truth[0, 3], marker="*", s=100, color='tab:green', label='Start')
plt.plot(ground_truth[:, 2], ground_truth[:, 3], color='tab:blue', label='Ground Truth')
plt.plot(states[:, 2], states[:, 3], color='tab:red', label='NN Prediction')
plt.grid(True, color='silver')
plt.axis('scaled')
plt.title('Load')
plt.figtext(0.5, 0.01, "Pos MSE: " + str(pos_mse) + "\nLoad MSE: " + str(load_mse) + "\nState MSE: " + str(state_mse), ha='center')
#plt.legend()
plt.show()

