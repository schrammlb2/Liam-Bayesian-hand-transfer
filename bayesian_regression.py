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
from common.utils import *
import random
import matplotlib.pyplot as plt

from sys import argv

data_type = 'pos' #type of data used for this task
task = 'real_A' #Which task we're training. This tells us what file to use
outfile = None
append = False
held_out = .1
# _ , arg1, arg2, arg3 = argv
epochs = 250
nn_type = '1'
# nn_type = 'LSTM'
SAVE = True
method = None

# pdb.set_trace()
if len(argv) > 1 and argv[1] != '_':
    data_type = argv[1]
if len(argv) > 2 and argv[2] != '_' :
    task = argv[2]
if len(argv) > 3 and argv[3] != '_':
    held_out = float(argv[3])
if len(argv) > 4 and argv[4] != '_':
    nn_type = argv[4]

assert data_type in ['pos', 'load']



state_dim = 4
action_dim = 2 if (task == 'sim_A' or task == 'sim_B') else 6
alpha = .4
lr = .0002
new_lr = lr/2

input_dim = state_dim + action_dim
output_dim = state_dim*2
# lr
# lr = .01
dropout_rate = .1

dtype = torch.float
cuda = torch.cuda.is_available()
print('cuda is_available: '+ str(cuda))
cuda = False

if task in ['real_A', 'real_B', 'sim_A', 'sim_B']:
    if task == 'real_A':
        datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'
        save_path = 'save_model/robotic_hand_real/pytorch'
    elif task == 'real_B':
        datafile_name = 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1_episodes.obj'
        save_path = 'save_model/robotic_hand_real/pytorch'
    elif task == 'sim_A':
        datafile_name = 'data/robotic_hand_simulator/A/sim_data_discrete_v14_d4_m1_episodes.obj'
        save_path = 'save_model/robotic_hand_simulator/pytorch'
    elif task == 'sim_B':
        datafile_name = 'data/robotic_hand_simulator/B/sim_data_discrete_v14_d4_m1_modified_episodes.obj'
        save_path = 'save_model/robotic_hand_simulator/pytorch'



    # save_path = 'save_model/robotic_hand_real/pytorch'
    with open(datafile_name, 'rb') as pickle_file:
        out = pickle.load(pickle_file, encoding='latin1')

    model = pt_build_model(nn_type, input_dim, output_dim, dropout_rate)
    if cuda: 
        model = model.cuda()

    model_save_path = save_path+'/'+ task + '_heldout' + str(held_out)+ '_' + nn_type



    l2_coeff = .000



elif task in ['transferA2B', 'transferB2A']:

    method = 'retrain'
    if len(argv) > 5:
        method = argv[5]



    if task == 'transferA2B':
        datafile_name = 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1_episodes.obj'
        save_path = 'save_model/robotic_hand_real/pytorch'
        save_file = save_path+'/real_A'+ '_' + nn_type
    if task == 'transferB2A':
        datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'
        save_path = 'save_model/robotic_hand_real/pytorch'
        save_file = save_path+'/real_B'+ '_' + nn_type



    model_save_path = save_path+'/'+ task + '_' + method + '_heldout' + str(held_out)+ '_' + nn_type


    if method in ['constrained_retrain', 'constrained_restart']:
        l2_coeff = .001
        if len(argv) > 6:
            l2_coeff = float(argv[6])
            model_save_path = save_path+'/'+ task + '_' + method + '_heldout' + str(held_out)+ '_' + str(l2_coeff) + '_' + nn_type


    with open(datafile_name, 'rb') as pickle_file:
        out = pickle.load(pickle_file, encoding='latin1')

    with open(save_file, 'rb') as pickle_file:
        if cuda: model = torch.load(pickle_file).cuda()
        else: model = torch.load(pickle_file, map_location='cpu')

    if method == 'constrained_retrain':
        mse_fn = torch.nn.MSELoss()
        base_parameters = [p.detach() for p in copy.deepcopy(model).parameters()]
        def offset_l2(new_model):
            loss = 0
            for i,parameter in enumerate(new_model.parameters()):
                loss += torch.norm(base_parameters[i]-parameter)

            return loss

    elif method == 'constrained_restart':
        mse_fn = torch.nn.MSELoss()
        base_parameters = [p.detach() for p in copy.deepcopy(model).parameters()]
        def offset_l2(new_model):
            loss = 0
            for i,parameter in enumerate(new_model.parameters()):
                loss += torch.norm(base_parameters[i]-parameter)

            return loss
        model = pt_build_model(nn_type, input_dim, output_dim, dropout_rate)

    elif method == 'linear_transform':
        model = LinearTransformedModel(model, input_dim, output_dim)

    elif method == 'nonlinear_transform':
        model = NonlinearTransformedModel(model, input_dim, output_dim)
        # model = pt_build_model(nn_type, state_dim+action_dim, state_dim, dropout_rate)        

    elif method == 'single_transform':
        model = torch.nn.Sequential(*[model, torch.nn.Linear(output_dim, output_dim)])
        for param in model[0].parameters():
            param.requires_grad = False

    else: 
        print("Invalid method type")
        assert False


if task == 'real_B':
    out = clean_data(out)
    print("data cleaning worked")
    # print(len(out))


out = [torch.tensor(ep, dtype=dtype) for ep in out]


model_save_path += '_bayesian.pkl'


new_state_dim = 4

data_type_offset = {'ave_load':4, 'load':2, 'pos':0}
dt_ofs = data_type_offset[data_type]
task_ofs = new_state_dim + action_dim

length_threshold = 30
out = list(filter(lambda x: len(x) > length_threshold, out))


val_size = int(len(out)*held_out)
# val_size = len(out) - int(held_out)
val_data = out[val_size:]
val_data = val_data[:min(10, len(val_data))]
out = out[:len(out)-val_size]

print("\nTraining with " + str(len(out)) + ' trajectories')

DATA = np.concatenate(out)

new_state_dim = 4

data_type_offset = {'ave_load':4, 'load':2, 'pos':0}
dt_ofs = data_type_offset[data_type]
task_ofs = new_state_dim + action_dim

x_data = DATA[:, :task_ofs]
y_data = DATA[:, -4:] - DATA[:, :4]


x_mean_arr = np.mean(x_data, axis=0)
x_std_arr = np.std(x_data, axis=0)
y_mean_arr = np.mean(y_data, axis=0)
y_std_arr = np.std(y_data, axis=0)
x_data = z_score_normalize(x_data, x_mean_arr, x_std_arr)
y_data = z_score_normalize(y_data, y_mean_arr, y_std_arr)
if SAVE:
    with open(save_path+'/normalization_arr/normalization_arr', 'wb') as pickle_file:
        pickle.dump(((x_mean_arr, x_std_arr),(y_mean_arr, y_std_arr)), pickle_file)


x_data = torch.tensor(x_data, dtype=dtype)
y_data = torch.tensor(y_data, dtype=dtype)

x_mean_arr = torch.tensor(x_mean_arr, dtype=dtype)
x_std_arr = torch.tensor(x_std_arr, dtype=dtype)
y_mean_arr = torch.tensor(y_mean_arr, dtype=dtype)
y_std_arr = torch.tensor(y_std_arr, dtype=dtype)

if cuda:
    x_mean_arr = x_mean_arr.cuda()
    x_std_arr = x_std_arr.cuda()
    y_mean_arr = y_mean_arr.cuda()
    y_std_arr = y_std_arr.cuda()



print('\n\n Beginning task: ' + model_save_path)
# pdb.set_trace()


if __name__ == "__main__":
    thresh = 10
    # model = BNNWrapper(model, input_dim, state_dim)
    with open("save_model/robotic_hand_real/pytorch/real_A_heldout0.1_1.pkl", 'rb') as pickle_file:
        # print("Running " + model_file)
        # pdb.set_trace()
        mean_model = torch.load(pickle_file, map_location='cpu')
        model = DividedBNN(mean_model, state_dim+action_dim, state_dim)

    if cuda: 
        model = model.cuda()
    norm = x_mean_arr, x_std_arr, y_mean_arr, y_std_arr
    trainer = BayesianTrainer(task, norm, model_save_path=model_save_path) 
    # trainer = DividedBayesTrainer(task, norm, model_save_path=model_save_path) 

    np.random.shuffle(out)

    lr = .000025
    # opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
    opt = torch.optim.Adam(model.std_dev_model.parameters(), lr=lr, weight_decay=.001)
    # opt = torch.optim.Adam(model.std_dev_model.parameters(), lr=lr, weight_decay=.001)

    trainer.pretrain(model, x_data, y_data, opt, epochs=1, batch_size=256)
    # if method == 'nonlinear_transform':
    #     model.set_base_model_train(True)
    # opt = torch.optim.Adam(model.parameters(), lr=.0000025, weight_decay=.001)
    opt = torch.optim.Adam(model.std_dev_model.parameters(), lr=.0000025, weight_decay=.001)
    trainer.batch_train(model, opt, out, val_data =val_data, epochs=10, batch_size=128)
    trainer.batch_train(model, opt, out, val_data =val_data, epochs=30, batch_size=64)      
    trainer.batch_train(model, opt, out, val_data =val_data, epochs=20, batch_size=16)
    trainer.batch_train(model, opt, out, val_data =val_data, epochs=30, batch_size=8)
    trainer.batch_train(model, opt, out, val_data =val_data, epochs=30, batch_size=4)
    trainer.batch_train(model, opt, out, val_data =val_data, epochs=30, batch_size=2)
