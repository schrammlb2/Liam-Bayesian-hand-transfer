import scipy.io
# from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb
import torch
from common.data_normalization import *
from common.pt_build_model import *
from common.utils_clean_traj import *
import matplotlib.pyplot as plt

from sys import argv


outfile = None
append = False
held_out = .1
test_traj = 1
# _ , arg1, arg2, arg3 = argv
nn_type = '1'
method = ''


if len(argv) > 1 and argv[1] != '_':
    nn_type = argv[1]
if len(argv) > 2 and argv[2] != '_':
    held_out = argv[3]
if len(argv) > 3 and argv[3] != '_':
    method = 'retrain'
    method = argv[3]

task = 'acrobot_A'



save_path = 'save_model/robotic_hand_real/pytorch/'

base = 'data/'


if 'acrobot' in task:
    task_loc = base + 'acrobot_task'
elif 'cartpole' in task:
    task_loc = base + 'cartpole_task'

with open(task_loc, 'rb') as pickle_file:
    task_dict = pickle.load(pickle_file)

base_version = task[-1]
target_version = 'B' if task[-1] == 'A' else 'A'

traj_filename = task_dict['test_' + target_version]
print(traj_filename)
with open(traj_filename, 'rb') as pickle_file:
    trajectory = pickle.load(pickle_file, encoding='latin1')

def make_traj(trajectory, test_traj):
    return trajectory[test_traj]



state_dim = task_dict['state_dim']
action_dim = task_dict['action_dim']

alpha = .4

dtype = torch.float
cuda = False#torch.cuda.is_available()
print('cuda is_available: '+ str(cuda))
mse_fn = torch.nn.MSELoss()

# held_out_list = [.99,.98,.97,.96,.95,.94,.93,.92,.91,.9,.8,.7,.6,.5,.4,.3,.2,.1]
# held_out_list = [.99,.98,.97,.96,.95]#,.94,.93,.92,.91,.9]#,.8,.7,.6,.5,.4,.3,.2,.1]
# held_out_list = [.998,.997,.996,.995,.994,.992,.992,.991,.99,.98,.97,.96,.95]

held_out_list = [.997,.996,.995,.994,.992,.991,.99]#,.98,.97,.96,.95,.94,.93,.92,.91,.9]
# held_out_list = [.99,.98,.97]
# held_out_list = [.998,.997,.996,.995,.994,.993,.992,.991]

def build_gp(model, held_out):

    with open(task_dict['train_' + target_version], 'rb') as pickle_file:
        out = pickle.load(pickle_file, encoding='latin1')
    
    task_ofs = state_dim + action_dim
    out = [torch.tensor(ep, dtype=dtype) for ep in out]

    trainer = TrajModelTrainer(task, out, model_save_path='', save_path='', state_dim=state_dim, action_dim=action_dim,
     task_ofs = task_ofs, reg_loss=None, nn_type='1', held_out=held_out) 

    norm = trainer.norm


    traj_model = model
    for param in traj_model.model.parameters():
        param.requires_grad = False

    traj_model.task = task
    #Initialize gp model
    train_x = trainer.x_data
    train_y = trainer.y_data

    res_train_y = train_y - traj_model(train_x)

    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    res_model =  ExactGPModel(train_x, res_train_y, likelihood=likelihood)

    res_model.train()

    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(.001),
        'covar_module.base_kernel.lengthscale': torch.tensor(2.8),
    }
    res_model.initialize(**hypers)

    opt = torch.optim.Adam([
        {'params': res_model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, res_model)

    training_iter = 1
    for i in range(training_iter):
        # Zero gradients from previous iteration
        opt.zero_grad()
        output = res_model(train_x)
        # Calc loss and backprop gradient
        loss2 = -mll(output, res_train_y.transpose(1,0))
        loss = torch.mean(loss2)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            res_model.covar_module.base_kernel.lengthscale.item(),
            res_model.likelihood.noise.item()
        ))
        opt.step()

    traj_model.res = res_model

    return traj_model

def run_traj(task, model, traj, threshold=None):
    true_states = traj[:,:state_dim]
    state = traj[0][:state_dim]
    states = []#state.view(1, state_dim)

    states = model.run_traj(traj)
    states = states.squeeze(0)

    for i, point in enumerate(traj):
        state = states[i]
        with torch.no_grad():
            mse = mse_fn(state[...,:2], point[...,:2])
        if mse > threshold:
            return states[:i], False, i


    return states, True, len(states)



def duration_lc(task, threshold, method=None):
    suffixes = 4

    if method == 'direct':
        iter_list = [.1]
    else:
        iter_list = held_out_list

    mean_durs = []
    err_durs = []
    for held_out in iter_list:
        durs = []

        for suffix in range(suffixes):
            if method == 'direct':
                model_save_path = save_path+ task[:-len('transferA2B')] + base_version + '_heldout' + str(held_out)+ '_' + nn_type + '.pkl'
            elif method == 'gp':
                model_save_path = save_path+ task[:-len('transferA2B')] + base_version + '_heldout' + str(.1)+ '_' + nn_type + '.pkl'
            elif method== None:
                model_save_path = save_path+ task[:-1] + target_version + '_heldout' + str(held_out)+ '_' + nn_type + str(suffix) + '.pkl'                
            else:
                model_save_path = save_path+ task + '_' + method +  '_heldout' + str(held_out)+ '_' + nn_type + str(suffix)+ '.pkl'

            if method == 'gp':            
                print("Running " + model_save_path)
                with open(model_save_path, 'rb') as pickle_file:
                    temp_model = torch.load(pickle_file, map_location='cpu')

                model = build_gp(temp_model, held_out)
                
            else:
                print("Running " + model_save_path)
                with open(model_save_path, 'rb') as pickle_file:
                    model = torch.load(pickle_file, map_location='cpu')

            for test_traj in range(4):
                ground_truth = make_traj(trajectory, test_traj)
                ground_truth = ground_truth[:len(ground_truth)]

                states, finished, duration = run_traj(task, model, torch.tensor(ground_truth, dtype=dtype), threshold=threshold)
                # if method == 'direct':
                states = states.detach().numpy()
                # if method != None:
                #     plt.close()
                #     plt.figure(1)
                #     plt.plot(ground_truth[:duration, 0], ground_truth[:duration, 1], color='blue', label='Ground Truth', marker='.')
                #     plt.plot(states[:duration, 0], states[:duration, 1], color='red', label='NN Prediction')
                #     plt.axis('scaled')
                #     plt.show()

                durs.append(duration)

        mean_durs.append(np.mean(durs))

        err_dur = np.std(durs)/suffixes**.5
        err_durs.append(err_dur)

    if method == 'direct':
        return_durs = []
        return_errs = []
        for held_out in held_out_list:
            return_durs = return_durs + mean_durs
            return_errs = return_errs + err_durs
    else:
        return_durs = mean_durs
        return_errs = err_durs

    return np.stack(return_durs, 0), np.stack(return_errs, 0)


single_shot = False

# lc_nl_trans = get_lc('transferB2A')
methods = ['gp', 'direct', 'traj_transfer_timeless_recurrent', 'traj_transfer_timeless', 'retrain_naive']
# methods = ['traj_transfer_timeless']

threshold = .001


lc_mean, lc_err = duration_lc(task, threshold)
held_out_arr = 1 - np.array(held_out_list)

plt.plot(held_out_arr, lc_mean, color='blue', label='New model')
plt.fill_between(held_out_arr, lc_mean+lc_err, lc_mean-lc_err, color='blue', label='New model', alpha=.7)

color_list = ['red', 'green', 'purple', 'black', 'orange', 'yellow', 'megenta']
plt.figure(1)
for method, color in zip(methods, color_list):
    lc_nl_trans, lc_err= duration_lc(task[:-2] + '_transferA2B', threshold, method=method)
    plt.plot(held_out_arr, lc_nl_trans, color=color, label=method)
    plt.fill_between(held_out_arr, lc_nl_trans+lc_err, lc_nl_trans-lc_err, color=color, label=method, alpha=.7)
plt.title('Duration')
plt.legend()
fig_loc = '/home/liam/results/recurrent_network_results/learning_curve_duration' + str(held_out_list[-1])+ '_traj_' + str(test_traj) + '.png'
plt.show()
