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

if 'acrobot' in task:
    task_loc = base + 'acrobot_task'
elif 'cartpole' in task:
    task_loc = base + 'cartpole_task'
elif 'hand' in task:
    task_loc = base + 'hand_task'

with open(task_loc, 'rb') as pickle_file:
    task_dict = pickle.load(pickle_file)
    
state_dim = task_dict['state_dim']
action_dim = task_dict['action_dim']



datafile_name = task_dict['train_' + task[-1]]
save_path = 'save_model/robotic_hand_real/pytorch/'
if 'transfer' in task: 
    save_file = save_path + task[:-len('transferA2B')] + task[-3] + '_heldout0.1_1.pkl'
    model_save_path = save_path+ task + '_' + method +  '_heldout' + str(held_out)+ '_' + nn_type + suffix+ '.pkl'
    with open(save_file, 'rb') as pickle_file:
        if cuda: model = torch.load(pickle_file)
        else: model = torch.load(pickle_file, map_location='cpu')
else:
    model_save_path = save_path+ task + '_heldout' + str(held_out)+ '_' + nn_type + suffix+ '.pkl'


with open(datafile_name, 'rb') as pickle_file:
    out = pickle.load(pickle_file, encoding='latin1')





task_ofs = state_dim + action_dim
out = [torch.tensor(ep, dtype=dtype) for ep in out]


val_size = int(len(out)*held_out)

val_data = out[-val_size:]
val_data = val_data[:min(10, len(val_data))]

print("\nTraining with " + str(int(len(out)*(1-held_out))) + ' trajectories')


print('\n\n Beginning task: ')
print('\t' + model_save_path)


if __name__ == "__main__":
    np.random.shuffle(out)

    # pdb.set_trace()

    trainer = TrajModelTrainer(task, out, model_save_path=model_save_path, save_path=save_path, state_dim=state_dim, action_dim=action_dim,
     task_ofs = task_ofs, reg_loss=reg_loss, nn_type=nn_type, held_out=held_out) 

    norm = trainer.norm
    

    if 'transfer' not in task:
        traj_model = TrajNet(task, norm, state_dim=state_dim, action_dim=action_dim)
        if nn_type == 'LSTM':
            traj_model = LSTMStateTrajNet(task, norm, state_dim=state_dim, action_dim=action_dim)
            print(traj_model.prediction)

        lr = base_lr
        opt_traj = torch.optim.Adam(traj_model.parameters(), lr=lr, weight_decay=.001)
        if cuda: 
            traj_model = traj_model.to('cuda')
            traj_model.norm = tuple(n.cuda() for n in traj_model.norm)

        if held_out < .2:
            epochs = base_epochs*2
        else:
            epochs = base_epochs

        trainer.pretrain(traj_model, opt_traj, epochs=epochs, batch_size=64)
        # model= traj_model


    elif 'transfer' in task:
        if method == 'traj_transfer_timeless':
            traj_model = TimeIndepLatentDeltaNet(task, norm, model=model, state_dim=state_dim, action_dim=action_dim)
            opt = torch.optim.Adam(traj_model.parameters(), lr=lr, weight_decay=.001)
            trainer.pretrain(traj_model, opt, epochs=base_epochs, batch_size=64)

        elif method in ['traj_transfer_timeless_recurrent', 'tt_scaled']:
            traj_model = model
            for param in traj_model.model.parameters():
                param.requires_grad = False

            traj_model.task = task
            traj_model.res = pt_build_model('1', state_dim + action_dim, state_dim, dropout_p=.1)
            traj_model.coeff = 1
            if method == 'tt_scaled':
                model.__class__ = ScaleNet
            opt = torch.optim.Adam(traj_model.parameters(), lr=lr, weight_decay=.001)
            trainer.pretrain(traj_model, opt, epochs=base_epochs, batch_size=64)

        elif method == 'gp':
            traj_model = model
            for param in traj_model.model.parameters():
                param.requires_grad = False

            traj_model.task = task
            #Initialize gp model
            train_x = trainer.x_data
            train_y = trainer.y_data

            res_train_y = train_y - traj_model(train_x)

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            res_model =  ExactGPModel(train_x, res_train_y, likelihood=likelihood)

            res_model.train()
            opt = torch.optim.Adam([
                {'params': res_model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=0.1)
            
            # trainer.pretrain(traj_model, opt, epochs=base_epochs//4 + 1, batch_size=1024)

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, res_model)

            training_iter = 50
            for i in range(training_iter):
                # Zero gradients from previous iteration
                opt.zero_grad()
                output = res_model(train_x)
                # Calc loss and backprop gradient
                # pdb.set_trace() 
                loss2 = -mll(output, res_train_y.transpose(1,0))
                loss = torch.mean(loss2)
                # l.backward()
                loss.backward()
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    res_model.covar_module.base_kernel.lengthscale.item(),
                    res_model.likelihood.noise.item()
                ))
                opt.step()

            traj_model.res = res_model
            with open(trainer.model_save_path, 'wb') as pickle_file:
                torch.save(traj_model, pickle_file)


        elif method == 'retrain':
            if cuda: 
                model = model.to('cuda')
                model.norm = tuple(n.cuda() for n in model.norm)
            model.task = task

            lr = base_lr
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
            trainer.batch_train(model, opt, val_data=val_data, epochs=base_epochs, batch_size=8)
        elif method == 'retrain_naive':
            if cuda: 
                model = model.to('cuda')
                model.norm = tuple(n.cuda() for n in model.norm)
            model.task = task
            # lr = .0000025
            lr = base_lr
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
            trainer.pretrain(model, opt, epochs=base_epochs, batch_size=64)
        else:
            print('invalid method')
            print(method)
            assert False
