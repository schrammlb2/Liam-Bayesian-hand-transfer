import scipy.io
# from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb
import torch
import torch.utils.data
from common.data_normalization import *
from common.sim_build_model import *
from common.utils_clean import *
import random
import matplotlib.pyplot as plt

from sys import argv

task = 'sim_cont' #Which task we're training. This tells us what file to use
append_name = 'test'
outfile = None
append = False
held_out = .9 if ('sim_transfer' in task) else .1
# _ , arg1, arg2, arg3 = argv
epochs = 250
nn_type = '0'
SAVE = True
method = ''

# pdb.set_trace()
if len(argv) > 1 and argv[1] != '_' :
    task = argv[1]
if len(argv) > 2 and argv[2] != '_':
    held_out = float(argv[2])
if len(argv) > 3 and argv[3] != '_':
    nn_type = argv[3]

assert task in ['real_A', 'real_B', 'transferA2B', 'transferB2A', 'sim_A', 'sim_B', 'sim_cont', 'sim_transferA2B', 'sim_transferB2A']



state_dim = 4
action_dim = 2 if ('sim' in task) else 6
alpha = .4
lr = .0002
new_lr = lr/2
dropout_rate = .1
l2_coeff = .01

dtype = torch.float
cuda = torch.cuda.is_available()
print('cuda is_available: '+ str(cuda))
cuda = False
reg_loss = None

if task in ['real_A', 'real_B', 'sim_A', 'sim_B', 'sim_cont']:
    if task == 'real_A':
        datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'
    elif task == 'real_B':
        datafile_name = 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1_episodes.obj'
    elif task == 'sim_A':
        datafile_name = 'data/robotic_hand_simulator/A/sim_data_discrete_v14_d4_m1_episodes.obj'
    elif task == 'sim_B':
        datafile_name = 'data/robotic_hand_simulator/B/sim_data_discrete_v14_d4_m1_modified_episodes.obj'
    elif task == 'sim_cont':
        datafile_name = 'data/robotic_hand_simulator/sim_data_cont_v0_d4_m1_episodes.obj'
        
    save_path = 'save_model/robotic_hand_simulator/pytorch' if ('sim' in task) else 'save_model/robotic_hand_real/pytorch'
    with open(datafile_name, 'rb') as pickle_file:
        out = pickle.load(pickle_file, encoding='latin1')

    model = pt_build_model(nn_type, state_dim+action_dim, state_dim, dropout_rate)

    if cuda: 
        model = model.cuda()

    model_save_path = save_path + '/' + task + '_' + append_name




elif task in ['transferA2B', 'transferB2A', 'sim_transferA2B', 'sim_transferB2A']:

    #method = 'retrain'
    if len(argv) > 4:
        method = argv[4]

    if task == 'transferA2B':
        datafile_name = 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1_episodes.obj'
        save_path = 'save_model/robotic_hand_real/pytorch'
        save_file = save_path+'/real_A'+ '_' + nn_type + '.pkl'
    if task == 'transferB2A':
        datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'
        save_path = 'save_model/robotic_hand_real/pytorch'
        save_file = save_path+'/real_B'+ '_' + nn_type + '.pkl'
    if task == 'sim_transferA2B':
        datafile_name = '../data/robotic_hand_simulator/B/sim_data_discrete_v14_d4_m1_modified_episodes.obj'
        save_path = '../save_model/robotic_hand_simulator/pytorch'
        save_file = save_path + '/sim_A_no_traj_bs16'
    if task == 'sim_transferB2A':
        datafile_name = '../data/robotic_hand_simulator/A/sim_data_discrete_v14_d4_m1_episodes.obj'
        save_path = '../save_model/robotic_hand_simulator/pytorch'
        save_file = save_path + '/sim_B_no_traj_bs16'


    with open(datafile_name, 'rb') as pickle_file:
        out = pickle.load(pickle_file, encoding='latin1')

    with open(save_file, 'rb') as pickle_file:
        if cuda: model = torch.load(pickle_file).cuda()
        else: model = torch.load(pickle_file, map_location='cpu')


    model_save_path = save_path + '/' + task + '_' + append_name
    

    if method in ['constrained_retrain', 'constrained_restart']:
        base_parameters = [p.detach() for p in copy.deepcopy(model).parameters()]
        def reg_loss(new_model):
            loss = 0
            for i,parameter in enumerate(new_model.parameters()):
                loss += torch.norm(base_parameters[i]-parameter)*l2_coeff
            return loss

        if len(argv) > 6:
            l2_coeff = float(argv[6])
            model_save_path = save_path+'/'+ task + '_' + method + '_heldout' + str(held_out)+ '_' + str(l2_coeff) + '_' + nn_type + '.pkl'

        if method == 'constrained_restart':
            model = pt_build_model(nn_type, state_dim+action_dim, state_dim, dropout_rate)

    elif method == 'linear_transform':
        model = LinearTransformedModel(model, state_dim + action_dim, state_dim)

    elif method == 'nonlinear_transform':
        model = NonlinearTransformedModel(model, state_dim + action_dim, state_dim)

    elif method != 'retrain': 
        print("Invalid method type")
        assert False




if task in ['real_B']:
    out = clean_data(out)
    print("data cleaning worked")

new_state_dim = 4

task_ofs = new_state_dim + action_dim

"""remove one-point trajectories"""
temp = []
for i in range(len(out)):
    if len(out[i]) > 1:
        temp.append(out[i])
out = temp

"""remove calibration point"""
if task in ['sim_A', 'sim_B']:
    mod = []
    for i in range(len(out)):
        mod = out[i]
        mod = mod[1:]
        out[i] = mod

out = [torch.tensor(ep, dtype=dtype) for ep in out]

full_dataset = out
concatenated = np.concatenate(out)
val_size = int(len(out)*held_out)

val_data = out[-val_size:]
val_data = val_data[:min(10, len(val_data))]
out = out[:-val_size]


print("\nTraining with " + str(len(out)) + ' trajectories')
print("Training with " + str(len(concatenated)) + " points")

print('\n\n Beginning task: ')
print('\t' + model_save_path)
# pdb.set_trace()

if __name__ == "__main__":
    np.random.shuffle(out)

    trainer = Trainer(task, out, model_save_path=model_save_path, save_path=save_path, state_dim=state_dim, action_dim=action_dim, task_ofs = task_ofs, reg_loss=reg_loss, append_name = append_name) 
    
    if held_out > .95: 
        lr = .000065
        lr = .0001
        lr = .0003
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
        # trainer.pretrain(model, opt, epochs=1)#, train_load=False)
        
        trainer.pretrain(model, opt, epochs=50)
        # if method == 'nonlinear_transform':
        #     model.set_base_model_train(True)
        opt = torch.optim.Adam(model.parameters(), lr=.000005, weight_decay=.001)
        trainer.batch_train(model, opt, val_data=val_data, epochs=10, batch_size=16)
        trainer.batch_train(model, opt, val_data=val_data, epochs=10, batch_size=4)
        trainer.batch_train(model, opt, val_data=val_data, epochs=10, batch_size=1)

    elif held_out > .9: 
        lr = .0001
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
        trainer.pretrain(model, x_data, y_data, opt, epochs=30)
        # if method == 'nonlinear_transform':
        #     model.set_base_model_train(True)
        opt = torch.optim.Adam(model.parameters(), lr=.000025, weight_decay=.001)
        trainer.batch_train(model, opt, out, val_data =val_data, epochs=25, batch_size=64)
    elif task == 'sim_cont':
        load_pre = True
        traj_train = True
        if load_pre:
            model = pt_build_model(nn_type, state_dim+action_dim, state_dim, .1)
            pre_task = 'sim_cont'
            pre_append_name = 'trajF_bs512_model512'
            with open(save_path + '/' + pre_task + '_' + pre_append_name, 'rb') as pickle_file:
                model = torch.load(pickle_file).cpu()
        else:
            opt = torch.optim.Adam(model.parameters(), lr=.00001, weight_decay=.001)
            trainer.pretrain(model, opt, epochs=100, batch_size=512)
        if traj_train:
            opt = torch.optim.Adam(model.parameters(), lr=.00001, weight_decay=.001)
            trainer.batch_train(model, opt, val_data=val_data, epochs=100, batch_size=16)
    elif ('sim' in task):
        opt = torch.optim.Adam(model.parameters(), lr=.00001, weight_decay=.001)
        trainer.pretrain(model, opt, epochs=100, batch_size=256)
        #opt = torch.optim.Adam(model.parameters(), lr=.00001, weight_decay=.001)
        #trainer.batch_train(model, opt, val_data =val_data, epochs=50, batch_size=16)
    else:
        lr = .000025
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
        # trainer.pretrain(model, x_data, y_data, opt, epochs=10, train_load=True, batch_size=256)
        if task == 'real_A': trainer.pretrain(model, x_data, y_data, opt, epochs=70, train_load=True, batch_size=256)
        if task == 'real_B': trainer.pretrain(model, x_data, y_data, opt, epochs=150, train_load=True, batch_size=256)

        # if method == 'nonlinear_transform':
        #     model.set_base_model_train(True)
        opt = torch.optim.Adam(model.parameters(), lr=.000005, weight_decay=.001)
        trainer.batch_train(model, opt, out, val_data =val_data, epochs=10, batch_size=256)
        trainer.batch_train(model, opt, out, val_data =val_data, epochs=40, batch_size=64)
        if task == 'real_B': trainer.batch_train(model, opt, out, val_data =val_data, epochs=30, batch_size=64)
        # trainer.batch_train(model, opt, out, val_data =val_data, epochs=20, batch_size=32)
        trainer.batch_train(model, opt, out, val_data =val_data, epochs=30, batch_size=16)
        if task == 'real_B': trainer.batch_train(model, opt, out, val_data =val_data, epochs=20, batch_size=16)
        trainer.batch_train(model, opt, out, val_data =val_data, epochs=30, batch_size=8)
        trainer.batch_train(model, opt, out, val_data =val_data, epochs=20, batch_size=4)
        trainer.batch_train(model, opt, out, val_data =val_data, epochs=20, batch_size=2)
# diagnostics_file.close()
    
