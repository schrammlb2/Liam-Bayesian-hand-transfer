import scipy.io
# from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb
import torch
import torch.utils.data
from common.data_normalization import *
from common.pt_build_model import *
import random
import matplotlib.pyplot as plt

from sys import argv

data_type = 'pos' #type of data used for this task
task = 'transferA2B' #Which task we're training. This tells us what file to use
outfile = None
append = False
held_out = .1
# _ , arg1, arg2, arg3 = argv
epochs = 12
nn_type = '1'

# pdb.set_trace()
if len(argv) > 1 and argv[1] != '_':
    task = argv[1]


state_dim = 4
action_dim = 6
alpha = .4
lr = .0003
# lr
# lr = .01

dtype = torch.float
cuda = torch.cuda.is_available()
print('cuda is_available: '+ str(cuda))

if task in ['real_A', 'real_B']:
    if task == 'real_A':
        datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'
    elif task == 'real_B':
        datafile_name = 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1_episodes.obj'

    save_path = 'save_model/robotic_hand_real/pytorch'
    with open(datafile_name, 'rb') as pickle_file:
        out = pickle.load(pickle_file, encoding='latin1')



elif task in ['transferA2B', 'transferB2A']:

    method = 'linear_transform'
    if len(argv) > 5:
        method = argv[5]

    if task == 'transferA2B':
        datafile_name = 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1_episodes.obj'
        save_path = 'save_model/robotic_hand_real/pytorch'
        save_file = save_path+'/real_A'+ '_' + nn_type + '.pkl'
    if task == 'transferB2A':
        datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'
        save_path = 'save_model/robotic_hand_real/pytorch'
        save_file = save_path+'/real_B'+ '_' + nn_type + '.pkl'



    with open(datafile_name, 'rb') as pickle_file:
        out = pickle.load(pickle_file, encoding='latin1')


def clean_data(out):
    DATA = np.concatenate(out)
    yd_pos = DATA[:, -4:-2] - DATA[:, :2]
    y2 = np.sum(yd_pos**2, axis=1)
    max_dist = np.percentile(y2, 99.84)

    skip_list = [np.sum((ep[:, -4:-2] - ep[:, :2])**5, axis=1)>max_dist for ep in out]
    divided_out = []
    for i,ep in enumerate(out):
        if np.sum(skip_list[i]) == 0:
            divided_out += [ep]

        else: 
            ep_lists = np.split(ep, np.argwhere(skip_list[i]).reshape(-1))
            divided_out += ep_lists

    divided_out = [ep[3:-3] for ep in divided_out]

    length_threshold = 30
    return list(filter(lambda x: len(x) > length_threshold, divided_out))

out = clean_data(out)

np.random.shuffle(out)
cutoff = 6
test_data = out[:cutoff]
train_data = out[cutoff:]


training_file_name = 'data/robotic_hand_real/B/B_training_episodes.obj'
with open(training_file_name, 'rb') as pickle_file:
    pickle.dump(train_data, pickle_file)

test_file_name = 'data/robotic_hand_real/B/B_test_episodes.obj'
with open(training_file_name, 'rb') as pickle_file:
    pickle.dump(train_data, pickle_file)
# pdb.set_trace()


