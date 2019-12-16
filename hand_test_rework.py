import scipy.io
# from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb
import torch
import torch.utils.data
from common.data_normalization import *
from common.TrajNet import *
from common.utils_clean_traj import *
import random
import matplotlib.pyplot as plt
import gpytorch 

from sys import argv

# task = 'real_A' #Which task we're training. This tells us what file to use
outfile = None
append = False
held_out = .1
# _ , arg1, arg2, arg3 = argv
# epochs = 250
nn_type = '1'
SAVE = False
method = 'traj_transfer'
suffix = ''
# pdb.set_trace()
task = 'hand_A'
if len(argv) > 1 and argv[1] != '_' :
    task = argv[1]
if len(argv) > 2 and argv[2] != '_':
    held_out = float(argv[2])
if len(argv) > 3 and argv[3] != '_':
    method = argv[3]
if len(argv) > 4 and argv[4] != '_':
    suffix = argv[4]


# action_dim = 6
alpha = .4
lr = .0002
new_lr = lr/2
# lr
# lr = .01
dropout_rate = .1
l2_coeff = .01

dtype = torch.float
# cuda = torch.cuda.is_available()
print('cuda is_available: '+ str(cuda))
# cuda = False
reg_loss = None


base_epochs = 20
base_lr = .00005


base = 'data/'


task_loc = base + 'hand_task'

with open(task_loc, 'rb') as pickle_file:
    task_dict = pickle.load(pickle_file)
    
state_dim = task_dict['state_dim']
action_dim = task_dict['action_dim']


# datafile_name = task_dict['train_' + task[-1]]
datafile_name = 'data/hand/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'
task='hand_A'


with open(datafile_name, 'rb') as pickle_file:
    out = pickle.load(pickle_file, encoding='latin1')

# cutoff = int(len(out)*.9)
cutoff = len(out) - 10
train_out = out[:cutoff]
test_out = out[cutoff:]

train_file = task_dict['train_' + task[-1]]
test_file = task_dict['test_' + task[-1]]

with open(train_file, 'wb') as pickle_file:
    pickle.dump(train_out, pickle_file)

with open(test_file, 'wb') as pickle_file:
    pickle.dump(test_out, pickle_file)

#--------------------------------------------------------------------------------------------
datafile_name = 'data/hand/t42_cyl35_red_data_discrete_v0.1_d4_m1_episodes.obj'
task='hand_B'


with open(datafile_name, 'rb') as pickle_file:
    out = pickle.load(pickle_file, encoding='latin1')

# cutoff = int(len(out)*.9)
cutoff = len(out) - 10
train_out = out[:cutoff]
test_out = out[cutoff:]

train_file = task_dict['train_' + task[-1]]
test_file = task_dict['test_' + task[-1]]

with open(train_file, 'wb') as pickle_file:
    pickle.dump(train_out, pickle_file)

with open(test_file, 'wb') as pickle_file:
    pickle.dump(test_out, pickle_file)
