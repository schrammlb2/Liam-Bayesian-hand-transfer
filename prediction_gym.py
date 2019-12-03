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
suffix = ''
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




base = 'data/'

if 'acrobot' in task:
    task_loc = base + 'acrobot_task'
elif 'cartpole' in task:
    task_loc = base + 'cartpole_task'

with open(task_loc, 'rb') as pickle_file:
    task_dict = pickle.load(pickle_file)
    
state_dim = task_dict['state_dim']
action_dim = task_dict['action_dim']

# pdb.set_trace()
datafile_name = task_dict['test_' + task[-1]]
save_path = 'save_model/robotic_hand_real/pytorch/'
if 'transfer' in task: 
    save_file = save_path + task[:-len('transferA2B')] + task[-3] + '_heldout0.1_1.pkl'
    model_save_path = save_path+ task + '_' + method +  '_heldout' + str(held_out)+ '_' + nn_type + suffix+ '.pkl'
else:
    model_save_path = save_path+ task + '_heldout' + str(held_out)+ '_' + nn_type + suffix+ '.pkl'



with open(datafile_name, 'rb') as pickle_file:
    trajectory = pickle.load(pickle_file, encoding='latin1')



# state_dim = 4
# action_dim = 6
alpha = .4

dtype = torch.float
cuda = torch.cuda.is_available()
cuda = False
print('cuda is_available: '+ str(cuda))

mse_fn = torch.nn.MSELoss()

def make_traj(trajectory, test_traj):
    return trajectory[test_traj]


with open(model_save_path, 'rb') as pickle_file:
    print("Running " + model_save_path)
    model = torch.load(pickle_file, map_location='cpu')#.eval()


max_mses = []
mses = []
threshold = None

for test_traj in range(4):
    if task == 'sim_A':
        ground_truth = make_traj_sim(trajectory)
    else:
        ground_truth = make_traj(trajectory, test_traj)

    # ground_truth = ground_truth[:len(ground_truth)//4]
    # ground_truth = ground_truth[...,:6]
    # if method in ['direct', 'retrain']:
    #     model.task = 'transferA2B'
    model.task = task
    # pdb.set_trace()

    model.coeff = .45
    if method == 'traj_transfer_timeless':
        model.coeff = .45

    gt = torch.tensor(ground_truth, dtype=dtype)

    if cuda: 
        gt = gt.cuda()
        x_mean_arr = x_mean_arr.cuda()
        x_std_arr = x_std_arr.cuda()
        y_mean_arr = y_mean_arr.cuda()
        y_std_arr = y_std_arr.cuda()

        model = model.to('cuda')
        model.norm = tuple([n.cuda() for n in model.norm])


    states = model.run_traj(gt, threshold=threshold)
    states = states.squeeze(0)
    if cuda:
        states = states.cpu()
    states = states.detach().numpy()
    duration = states.shape[-2]
    finished = duration == ground_truth.shape[0]


    plt.figure(1)
    plt.scatter(ground_truth[0, 0], ground_truth[0, 1], marker="*", label='start')
    plt.plot(ground_truth[:duration, 0], ground_truth[:duration, 1], color='blue', label='Ground Truth', marker='.')
    plt.plot(states[:, 0], states[:, 1], color='red', label='NN Prediction')

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