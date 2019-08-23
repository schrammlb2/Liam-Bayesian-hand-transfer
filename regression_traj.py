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
# SAVE = False

SAVE = True
method = 'traj_transfer'
suffix = ''

# pdb.set_trace()
if len(argv) > 1 and argv[1] != '_' :
    task = argv[1]
if len(argv) > 2 and argv[2] != '_':
    held_out = float(argv[2])
if len(argv) > 3 and argv[3] != '_':
    nn_type = argv[3]
if len(argv) > 4 and argv[4] != '_':
    method = argv[4]
if len(argv) > 5 and argv[5] != '_':
    suffix = argv[5]

assert task in ['real_A', 'real_B', 'transferA2B', 'transferB2A', 'sim_A', 'sim_B']



state_dim = 4
action_dim = 2 if (task == 'sim_A' or task == 'sim_B' or nn_type == 'LSTM') else 6
action_dim = 2 if (task == 'sim_A' or task == 'sim_B') else 6
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




save_path = 'save_model/robotic_hand_real/pytorch'
if task[-1] == 'B':
    datafile_name = 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1_episodes.obj'
    # save_file = save_path+'/real_A'+ '_' + nn_type + '.pkl'
    save_file = save_path+'/real_A'+ '_heldout0.2_' + nn_type + '.pkl'
if task[-1] == 'A':
    datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'
    # save_file = save_path+'/real_B'+ '_' + nn_type + '.pkl'
    save_file = save_path+'/real_B'+ '_heldout0.1_' + nn_type + '.pkl'


model_save_path = save_path+'/'+ task + '_heldout' + str(held_out)+ '_' + nn_type + '.pkl'


with open(datafile_name, 'rb') as pickle_file:
    out = pickle.load(pickle_file, encoding='latin1')

if 'transfer' in task:
    with open(save_file, 'rb') as pickle_file:
        if cuda: model = torch.load(pickle_file)
        else: model = torch.load(pickle_file, map_location='cpu')


    model_save_path = save_path+'/'+ task + '_' + method + '_heldout' + str(held_out)+ '_' + nn_type + '.pkl'



# pdb.set_trace()
model_save_path = model_save_path[:-4] + suffix + '.pkl'

if task in ['real_B', 'sim_A']:
    out = clean_data(out)
    print("data cleaning worked")

new_state_dim = 4

task_ofs = new_state_dim + action_dim

length_threshold = 30
out = list(filter(lambda x: len(x) > length_threshold, out))


out = [torch.tensor(ep, dtype=dtype) for ep in out]
# pdb.set_trace()
if suffix != '':
    np.random.shuffle(out)

def transfer(x, state_dim): 
    return torch.cat((x[...,:state_dim], x[...,state_dim:state_dim+2]*-1,  x[...,state_dim+2:]), -1) 
if task[-1] == 'A':
    out = [transfer(ep, state_dim) for ep in out]

val_size = int(len(out)*held_out)

val_data = out[-val_size:]
val_data = val_data[:min(10, len(val_data))]

print("\nTraining with " + str(int(len(out)*(1-held_out))) + ' trajectories')


print('\n\n Beginning task: ')
print('\t' + model_save_path)
# pdb.set_trace()

# if cuda: 
#     out = [ep.cuda() for ep in out]

if __name__ == "__main__":
    np.random.shuffle(out)

    trainer = TrajModelTrainer(task, out, model_save_path=model_save_path, save_path=save_path, state_dim=state_dim, action_dim=action_dim,
     task_ofs = task_ofs, reg_loss=reg_loss, nn_type=nn_type, held_out=held_out) 

    norm = trainer.norm
    

    if 'transfer' in task:
        if method == 'traj_transfer':
            # model.model = TrivialNet()
            # traj_model = LatentNet(task, norm, model=model, state_dim=state_dim, action_dim=action_dim)
            traj_model = LatentDeltaNet(task, norm, model=model, state_dim=state_dim, action_dim=action_dim)
            # opt_traj = torch.optim.Adam(traj_model.parameters(), lr=.000005, weight_decay=.001)
            opt = torch.optim.Adam(traj_model.parameters(), lr=lr, weight_decay=.001)
            trainer.pretrain(traj_model, opt, epochs=5, batch_size=64)
            # trainer.pretrain(traj_model, opt, epochs=60, batch_size=64)
            opt_traj = torch.optim.Adam(traj_model.parameters(), lr=.00005, weight_decay=.001)
            # opt_traj = torch.optim.Adam(traj_model.parameters(), lr=.00000000000000000000005, weight_decay=.001)
            if held_out < .9:
                trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=30, batch_size=256)
                trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=70, batch_size=8)

            else:
                # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=30, batch_size=256)
                trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=60, batch_size=32)
                trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=80, batch_size=8)
                trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=50, batch_size=4)
                trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=50, batch_size=2)
                trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=30, batch_size=1)
                # trainer.visualize(traj_model, val_data[0])
        elif method == 'traj_transfer_timeless':
            traj_model = TimeIndepLatentDeltaNet(task, norm, model=model, state_dim=state_dim, action_dim=action_dim)
            opt = torch.optim.Adam(traj_model.parameters(), lr=lr/10, weight_decay=.001)
            # trainer.pretrain(traj_model, opt, epochs=60, batch_size=64)
            trainer.pretrain(traj_model, opt, epochs=5, batch_size=64)
            # trainer.batch_train(model, opt, val_data=val_data, epochs=30, batch_size=8)

        elif method == 'traj_transfer_timeless_recurrent':
            traj_model = model
            for param in traj_model.model.parameters():
                param.requires_grad = False

            traj_model.task = task
            traj_model.res = pt_build_model('1', state_dim + action_dim, state_dim, dropout_p=.1)
            traj_model.coeff = 1


            # lr = .000025
            opt = torch.optim.Adam(traj_model.parameters(), lr=lr/100, weight_decay=.001)
            # trainer.pretrain(traj_model, opt, epochs=60, batch_size=64)
            trainer.pretrain(traj_model, opt, epochs=100, batch_size=64)


        elif method == 'retrain':
            if cuda: 
                model = model.to('cuda')
                model.norm = tuple(n.cuda() for n in model.norm)
            model.task = task
            # pdb.set_trace()

            lr = .000025
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
            # opt = torch.optim.Adam(model.parameters(), lr=.0000000000000000001, weight_decay=.001)
            # trainer.pretrain(model, opt, epochs=3, batch_size=64)
            # trainer.pretrain(model, opt, epochs=30, batch_size=64)
            # trainer.pretrain(model, opt, epochs=30, batch_size=64)
            # opt = torch.optim.Adam(model.parameters(), lr=.000005, weight_decay=.001)
            # trainer.batch_train(model, opt, val_data=val_data, epochs=80, batch_size=8)
            trainer.batch_train(model, opt, val_data=val_data, epochs=30, batch_size=8)
            # trainer.batch_train(model, opt, val_data=val_data, epochs=1, batch_size=8)
        elif method == 'retrain_naive':
            if cuda: 
                model = model.to('cuda')
                model.norm = tuple(n.cuda() for n in model.norm)
            model.task = task
            lr = .0000025
            lr = .000025
            # lr = 0
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
            # pdb.set_trace()
            trainer.pretrain(model, opt, epochs=30, batch_size=64)
        else:
            print('invalid method')
            print(method)
            assert False


    else:
        traj_model = TrajNet(task, norm, state_dim=state_dim, action_dim=action_dim)
        if nn_type == 'LSTM':
            traj_model = LSTMStateTrajNet(task, norm, state_dim=state_dim, action_dim=action_dim)
            print(traj_model.prediction)

        lr = .0000025
        opt_traj = torch.optim.Adam(traj_model.parameters(), lr=lr, weight_decay=.001)
        if cuda: 
            traj_model = traj_model.to('cuda')
            traj_model.norm = tuple(n.cuda() for n in traj_model.norm)
        # trainer.pretrain(traj_model, opt_traj, epochs=30, batch_size=64)
        if nn_type == 'LSTM': 
            trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=50, batch_size=256, sub_chance=.1)
            trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=50, batch_size=256, sub_chance=.02)
            trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=40, batch_size=256, sub_chance=.01)
            # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=30, batch_size=256)
            trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=200, batch_size=8)
            trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=100, batch_size=4)
            trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=100, batch_size=2)
            trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=100, batch_size=1)
        else:
            trainer.pretrain(traj_model, opt_traj, epochs=100, batch_size=64)
            trainer.pretrain(traj_model, opt_traj, epochs=100, batch_size=64)
            trainer.pretrain(traj_model, opt_traj, epochs=100, batch_size=64)
            model= traj_model


            # opt = torch.optim.Adam(model.parameters(), lr=.000025, weight_decay=.001)
            # trainer.batch_train(model, opt, val_data =val_data, epochs=10, batch_size=256)
            # trainer.batch_train(model, opt, val_data =val_data, epochs=40, batch_size=64)
            # if task == 'real_B': trainer.batch_train(model, opt, out, val_data =val_data, epochs=30, batch_size=64)
            # # trainer.batch_train(model, opt, out, val_data =val_data, epochs=20, batch_size=32)
            # trainer.batch_train(model, opt, val_data =val_data, epochs=30, batch_size=16)
            # if task == 'real_B': trainer.batch_train(model, opt, out, val_data =val_data, epochs=20, batch_size=16)
            # trainer.batch_train(model, opt, val_data =val_data, epochs=30, batch_size=8)
            # trainer.batch_train(model, opt, val_data =val_data, epochs=20, batch_size=4)
            # trainer.batch_train(model, opt, val_data =val_data, epochs=20, batch_size=2)
        # trainer.pretrain(traj_model, opt_traj, epochs=150, batch_size=64)
        # opt_traj = torch.optim.Adam(traj_model.parameters(), lr=.000005, weight_decay=.001)

        # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=30, batch_size=8)

    # opt_old2new = torch.optim.Adam(old2new.parameters(), lr=.000005, weight_decay=.001)
    # opt_new2old = torch.optim.Adam(new2old.parameters(), lr=.000005, weight_decay=.001)

    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=50, batch_size=256, sub_chance=1.0)
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=40, batch_size=256, sub_chance=.50)
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=20, batch_size=256, sub_chance=.30)
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=20, batch_size=256, sub_chance=.20)
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=20, batch_size=256, sub_chance=.15)
    


    # 
    # trainer.batch_train(traj_model, opt_traj, val_data=val_data, epochs=60, batch_size=4)