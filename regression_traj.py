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

from sys import argv

task = 'real_A' #Which task we're training. This tells us what file to use
outfile = None
append = False
held_out = .1
# _ , arg1, arg2, arg3 = argv
epochs = 250
nn_type = '1'
SAVE = False
method = None

# pdb.set_trace()
if len(argv) > 1 and argv[1] != '_' :
    task = argv[1]
if len(argv) > 2 and argv[2] != '_':
    held_out = float(argv[2])
if len(argv) > 3 and argv[3] != '_':
    nn_type = argv[3]

assert task in ['real_A', 'real_B', 'transferA2B', 'transferB2A', 'sim_A', 'sim_B']



state_dim = 4
action_dim = 2 #if (task == 'sim_A' or task == 'sim_B' or nn_type == 'LSTM') else 6
# action_dim = 6
alpha = .4
lr = .0002
new_lr = lr/2
# lr
# lr = .01
dropout_rate = .1
l2_coeff = .01

dtype = torch.float
cuda = torch.cuda.is_available()
print('cuda is_available: '+ str(cuda))
cuda = False
reg_loss = None



method = 'traj_transfer'
if len(argv) > 4:
    method = argv[4]


save_path = 'save_model/robotic_hand_real/pytorch'
if task[-1] == 'B':
    datafile_name = 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1_episodes.obj'
    save_file = save_path+'/real_A'+ '_' + nn_type + '.pkl'
if task[-1] == 'A':
    datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'
    save_file = save_path+'/real_B'+ '_' + nn_type + '.pkl'



with open(datafile_name, 'rb') as pickle_file:
    out = pickle.load(pickle_file, encoding='latin1')

with open(save_file, 'rb') as pickle_file:
    if cuda: model = torch.load(pickle_file).cuda()
    else: model = torch.load(pickle_file, map_location='cpu')


# model_save_path = save_path+'/'+ task + '_' + method + '_heldout' + str(held_out)+ '_' + nn_type + '.pkl'






if task in ['real_B', 'sim_A']:
    out = clean_data(out)
    print("data cleaning worked")

new_state_dim = 4

task_ofs = new_state_dim + action_dim

length_threshold = 30
out = list(filter(lambda x: len(x) > length_threshold, out))


out = [torch.tensor(ep, dtype=dtype) for ep in out]

full_dataset = out

val_size = int(len(out)*held_out)

val_data = out[-val_size:]
val_data = val_data[:min(10, len(val_data))]
out = out[:-val_size]


print("\nTraining with " + str(len(out)) + ' trajectories')


model_save_path = save_path+'/'+ task + '_heldout' + str(held_out)+ '_' + nn_type + '.pkl'
print('\n\n Beginning task: ')
print('\t' + model_save_path)
# pdb.set_trace()


if __name__ == "__main__":
    np.random.shuffle(out)

    trainer = TrajModelTrainer(task, out, model_save_path=model_save_path, save_path=save_path, state_dim=state_dim, action_dim=action_dim,
     task_ofs = task_ofs, reg_loss=reg_loss, nn_type=nn_type) 

    norm = trainer.norm
    # traj_model = NBackNet(task, norm, state_dim=state_dim, action_dim=action_dim)
    traj_model = TrajNet(task, norm, state_dim=state_dim, action_dim=action_dim)
    traj_model = LSTMTrajNet(task, norm, state_dim=state_dim, action_dim=action_dim)
    # traj_model = LSTMStateTrajNet(task, norm, state_dim=state_dim, action_dim=action_dim)

    print(traj_model.prediction)

    lr = .0000025
    opt = torch.optim.Adam(traj_model.parameters(), lr=lr, weight_decay=.001)
    trainer.pretrain(traj_model, opt, epochs=70, batch_size=64)

    opt_traj = torch.optim.Adam(traj_model.parameters(), lr=.000005, weight_decay=.001)
    # opt_old2new = torch.optim.Adam(old2new.parameters(), lr=.000005, weight_decay=.001)
    # opt_new2old = torch.optim.Adam(new2old.parameters(), lr=.000005, weight_decay=.001)
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=30, batch_size=512, sub_chance=1.0)
    trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=30, batch_size=512, sub_chance=.3)
    trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=30, batch_size=512, sub_chance=.2)
    trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=30, batch_size=512, sub_chance=.1)
    trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=60, batch_size=256)
    trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=50, batch_size=8)
    trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=50, batch_size=4)
    trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=50, batch_size=2)

    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=50, batch_size=256, sub_chance=1.0)
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=40, batch_size=256, sub_chance=.50)
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=20, batch_size=256, sub_chance=.30)
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=20, batch_size=256, sub_chance=.20)
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=20, batch_size=256, sub_chance=.15)
    
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=30, batch_size=8, sub_chance=.15)
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=20, batch_size=8, sub_chance=.1)
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=10, batch_size=8, sub_chance=.075)
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=10, batch_size=8, sub_chance=.05)

    # 
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=60, batch_size=4)