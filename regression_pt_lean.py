import scipy.io
# from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb
import torch
from common.data_normalization import *
from common.pt_build_model import pt_build_model
from common.RecurrentNet import RecurrentNet
import random
import matplotlib.pyplot as plt

from sys import argv

data_type = 'pos' #type of data used for this task
task = 'real_A' #Which task we're training. This tells us what file to use
skip_step = 1
outfile = None
append = False
held_out = .1
# _ , arg1, arg2, arg3 = argv
nn_type = '1'

# pdb.set_trace()
if len(argv) > 1:
    data_type = argv[1]
if len(argv) > 2:
    task = argv[2]
if len(argv) > 3:
    held_out = float(argv[3])

assert data_type in ['pos', 'load']
assert task in ['real_A', 'real_B','sim_A', 'sim_B']
assert skip_step in [1,10]
if (len(argv) > 3 and task != 'sim_A'):
    print('Err: Skip step only appicable to sim_A task. Do not use this argument for other tasks')
    exit(1)



state_dim = 4
action_dim = 6
alpha = .4

datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'



save_path = 'save_model/robotic_hand_real/pytorch'
with open(datafile_name, 'rb') as pickle_file:
    out = pickle.load(pickle_file, encoding='latin1')




dtype = torch.float
cuda = torch.cuda.is_available()
print('cuda is_available: '+ str(cuda))



model = RecurrentNet(nn_type, state_dim+action_dim, state_dim, .1, save_path=save_path, cuda=cuda, file_name = task + '_' + nn_type + '.pkl')




DATA = np.concatenate(out)

out = [torch.tensor(ep, dtype=dtype) for ep in out]
if __name__ == "__main__":

    model.train(out, DATA, held_out)


    # if outfile: 
    #     if append:
    #         f = open(outfile, 'a+')
    #     else:
    #         f = open(outfile, 'w+')
    #     out_string= ('cold start\t' + data_type +
    #                 '\t' + task + '\t' + str(held_out) +
    #                 '\t:' + str(final_loss) + '\n')
    #     f.write(out_string)
    #     f.close()
    