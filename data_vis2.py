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

if len(argv) > 1:
    test_traj = int(argv[1])
if len(argv) > 2:
    task = argv[2]
if len(argv) > 3:
    nn_type = argv[3]

assert task in ['real_A', 'real_B','transferA2B','transferB2A']



save_path = 'save_model/robotic_hand_real/pytorch/'

# model_loc_map = {'real_A': ('save_model/robotic_hand_real/A/pos', 'save_model/robotic_hand_real/A/load'),
#     'real_B': ('save_model/robotic_hand_real/B/pos', 'save_model/robotic_hand_real/B/load'),
#     'sim_s1' : ('save_model/robotic_hand_simulator/A/d4_s1_pos', 'save_model/robotic_hand_simulator/A/d4_s1_load'),
#     'sim_s10' : ('save_model/robotic_hand_simulator/A/d4_s10_pos', 'save_model/robotic_hand_simulator/A/d4_s10_load')
#     }

# trajectory_path = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'



def clean_data(episodes):
    DATA = np.concatenate(episodes)
    yd_pos = DATA[:, -4:-2] - DATA[:, :2]
    y2 = np.sum(yd_pos**2, axis=1)
    max_dist = np.percentile(y2, 99.84)
    # max_dist = np.percentile(y2, 99.6)

    skip_list = [np.sum((ep[:, -4:-2] - ep[:, :2])**5, axis=1)>max_dist for ep in episodes]
    divided_episodes = []
    for i,ep in enumerate(episodes):
        if np.sum(skip_list[i]) == 0:
            divided_episodes += [ep]

        else: 
            ep_lists = np.split(ep, np.argwhere(skip_list[i]).reshape(-1))
            divided_episodes += ep_lists

    divided_episodes = [ep[3:-3] for ep in divided_episodes]

    length_threshold = 30
    return list(filter(lambda x: len(x) > length_threshold, divided_episodes))




state_dim = 4
action_dim = 6
alpha = .4

systems = ['A', 'B']
traj_paths = {'A': 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj', 'B': 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1_episodes.obj'}


datasets = []

for system in systems:

    with open(traj_paths[system], 'rb') as pickle_file:
        trajectories = pickle.load(pickle_file, encoding='latin1')

    with open(save_path+'/normalization_arr/normalization_arr' + system, 'rb') as pickle_file:
        x_norm_arr, y_norm_arr = pickle.load(pickle_file)

    x_mean_arr, x_std_arr = x_norm_arr[0], x_norm_arr[1]
    y_mean_arr, y_std_arr = y_norm_arr[0], y_norm_arr[1]

    DATA = np.concatenate(trajectories)

    x_data = DATA[:, :10]
    y_data = DATA[:, -4:] - DATA[:, :4]
    
    x_data = z_score_normalize(x_data, x_mean_arr, x_std_arr)
    y_data = z_score_normalize(y_data, y_mean_arr, y_std_arr)

    DATA = np.concatenate((x_data, y_data), -1)

    datasets.append(DATA)

min_len = min([len(d) for d in datasets])

valid = [np.isclose(datasets[0][:min_len,:6], np.roll(datasets[1][:min_len,:6], i, axis=0), atol=.075).all(axis=-1) for i in range(6000)]
valid = np.stack(valid, 0)
# pdb.set_trace()
if np.sum(valid) == 0:
    print('No points close enough')
    pdb.set_trace()

locs = np.argwhere(valid)

# trimmed_locs = locs[np.sum(valid[locs[:,0]], 1) > 2]
high_density_locations = np.argwhere(np.sum(valid, 1) > 2)


# pdb.set_trace()
title_dict = {0: 'pos1', 1: 'pos2', 2: 'load1', 3: 'load2'}
# for i in range(10):
#     interval = 10
#     A_int = A_locs[i*interval:interval*(i+1)]
#     B_int = B_locs[i*interval:interval*(i+1)]
#     plot_ranges = [datasets[0][A_int], datasets[1][B_int]]
#     # pdb.set_trace()

#     for j in range(4):
#         name = '/home/liam/results/recurrent_network_results/hand_comparison/' + str(i) + '_' + str(j)
#         plt.figure(1)
#         plt.scatter(plot_ranges[0][:,-j], plot_ranges[1][:,-j], marker='.')
#         plt.title('interval ' + str(i) + ', ' + title_dict[j])
#         plt.legend()
#         plt.savefig(name)
#         # plt.show()

#         plt.close()

# for location in trimmed_locs[:,0]:
def find_nearest(value, array):
    try: 
        idx = np.sum((array[...,:6] - value[...,:6])**2, 1).argmin()
        return array[idx]
    except: 
        pdb.set_trace()

plotted_locs = []

for location in high_density_locations[:,0]:
    if location in plotted_locs: 
        continue
    neighborhood = locs[np.argwhere(np.isclose(location, locs[:,0], atol = 30))].reshape(-1,2)
    # pdb.set_trace()
    plotted_locs += neighborhood[:,0].tolist()

    A_locs = neighborhood[...,1]
    B_locs = neighborhood[...,1] + neighborhood[...,0]
    
    A_points = datasets[0][A_locs, :]
    B_points = datasets[1][B_locs, :]

    # pdb.set_trace()

    for j in range(4):
        all_pairs = []
        for point in A_points:
            pair = [point, find_nearest(point, B_points)]
            all_pairs.append(np.stack(pair, 0))
        all_pairs = np.stack(all_pairs, 0)
        pdb.set_trace()

        name = '/home/liam/results/recurrent_network_results/hand_comparison/y_predict' + str(location) + '_' + str(j)
        plt.figure(1)
        # plt.scatter(all_pairs[:, 0, j], all_pairs[:, 1, j], marker='.')
        # plt.scatter(all_pairs[:, 0, -(4-j)], all_pairs[:, 1, -(4-j)], marker='.')

        # plt.scatter(A_points[..., j], B_points[..., j], marker='.')
        plt.scatter(A_points[..., -(4-j)], B_points[..., -(4-j)], marker='.')
        plt.title('location' + str(location) + ', ' + title_dict[j])
        plt.legend()
        plt.savefig(name)
        # plt.show()

        plt.close()
