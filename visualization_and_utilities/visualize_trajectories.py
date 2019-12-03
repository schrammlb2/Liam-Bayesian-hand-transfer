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

if task in ['real_A', 'real_B', 'sim_A', 'sim_B']:
    if task == 'real_A':
        datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'
    elif task == 'real_B':
        datafile_name = 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1_episodes.obj'
    elif task == 'sim_A':
        datafile_name = 'data/robotic_hand_simulator/A/sim_data_discrete_v14_d4_m1_episodes.obj'
    elif task == 'sim_B':
        datafile_name = 'data/robotic_hand_simulator/B/sim_data_discrete_v14_d4_m1_modified_episodes.obj'

    save_path = 'save_model/robotic_hand_simulator/pytorch' if (task == 'sim_A' or task == 'sim_B') else 'save_model/robotic_hand_real/pytorch'
    with open(datafile_name, 'rb') as pickle_file:
        out = pickle.load(pickle_file, encoding='latin1')

    model = pt_build_model(nn_type, state_dim+action_dim, state_dim, .1)
    if cuda: 
        model = model.cuda()

    model_save_path = save_path+'/'+ task + '_heldout' + str(held_out)+ '_' + nn_type + '.pkl'



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

    with open(save_file, 'rb') as pickle_file:
        if cuda: model = torch.load(pickle_file).cuda()
        else: model = torch.load(pickle_file, map_location='cpu')

    if method == 'constrained_retrain':
        mse_fn = torch.nn.MSELoss()
        base_parameters = [p.detach() for p in copy.deepcopy(model).parameters()]
        def offset_l2(new_model):
            loss = 0
            for i,parameter in enumerate(new_model.parameters()):
                # pdb.set_trace()
                loss += torch.norm(base_parameters[i]-parameter)

            return loss

    elif method == 'constrained_restart':
        mse_fn = torch.nn.MSELoss()
        base_parameters = [p.detach() for p in copy.deepcopy(model).parameters()]
        def offset_l2(new_model):
            loss = 0
            for i,parameter in enumerate(new_model.parameters()):
                # pdb.set_trace()
                loss += torch.norm(base_parameters[i]-parameter)

            return loss
        model = pt_build_model(nn_type, state_dim+action_dim, state_dim, .1)


    elif method == 'linear_transform':
        model = LinearTransformedModel(model, state_dim + action_dim, state_dim)

    else: assert False


def pretrain(model, x_data, y_data, opt, train_load = True):
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 64)
    # loss_fn = torch.nn.NLLLoss()
    loss_fn = torch.nn.MSELoss()
    for i in range(30):
        print("Pretraining epoch: " + str(i))
        for batch_ndx, sample in enumerate(loader):
            # pdb.set_trace()
            out = model(sample[0])
            if train_load:
                loss = loss_fn(out, sample[1]) 
            else:
                loss = loss_fn(out[:,:2], sample[1][:,:2])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            opt.step()
            opt.zero_grad()


opt = torch.optim.Adam(model.parameters(), lr=lr)



def run_traj(model, traj, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = 50, return_states = False, 
    loss_type = 'softmax', alpha = .5):
    true_states = traj[:,:state_dim]
    state = traj[0][:state_dim]
    states = []#state.view(1, state_dim)
    sim_deltas = []
    if cuda:
        state = state.cuda()
        true_states = true_states.cuda()

    mse_fn = torch.nn.MSELoss()

    def softmax(states):
        mse_fn = torch.nn.MSELoss(reduction='none')
        mse = mse_fn(states[:,:2], true_states[:states.shape[0],:2])
        mse = torch.sum(mse, 1)  #Sum two position losses at each time step to get the Euclidean distance
        return torch.logsumexp(mse, 0)

    def stepwise(sim_deltas):
        # sim_deltas = states[1:, :2] - states[:-1, :2] #starting position version
        # real_deltas = traj[1:, :2] - traj[:-1, :2] #starting position version
        real_deltas = traj[:, :2] - traj[:, -4:-2] # y_data version
        mse_fn = torch.nn.MSELoss()
        mse = mse_fn(sim_deltas[:,:2], real_deltas[:len(sim_deltas)])
        return mse

    def mix(sim_deltas, states, alpha = .65):
        return stepwise(sim_deltas)*alpha + softmax(states)*(1-alpha)

    def get_loss(loss_type, states = None, sim_deltas = None):
        if loss_type == 'soft maximum':
            loss = softmax(states)
        if loss_type == 'mix':
            loss = mix(sim_deltas, states)
            return loss
        else:
            mse_fn = torch.nn.MSELoss()
            loss = mse_fn(states[:,:2], true_states[:,:2])

        return loss

    for i, point in enumerate(traj):
        states.append(state)
        action = point[state_dim:state_dim+action_dim]
        if cuda: action = action.cuda()    
        inpt = torch.cat((state, action), 0)
        inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

        # pdb.set_trace()
        state_delta = model(inpt)
        state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
        if task in ['transferA2B', 'transferB2A']: state_delta *= torch.tensor([-1,-1,1,1], dtype=dtype)
        sim_deltas.append(state_delta)

        state= state_delta + state

        #May need random component here to prevent overfitting
        # states = torch.cat((states,state.view(1, state_dim)), 0)
        if threshold and i%10:
            with torch.no_grad():
                mse = mse_fn(state[:2], true_states[i,:2])
            if mse > threshold:
                states = torch.stack(states, 0)
                sim_deltas = torch.stack(sim_deltas, 0)
                # loss = mix(sim_deltas, states, alpha)
                loss = softmax(states)
                # loss = get_loss(loss_type, states=states, sim_deltas=sim_deltas)
                return loss, 0, i
                # return mse_fn(state[:2], true_states[i,:2]), 0, i

    sim_deltas = torch.stack(sim_deltas, 0)
    states = torch.stack(states, 0)
    if return_states:
        return states

    return get_loss(loss_type, states=states, sim_deltas=sim_deltas), 1, len(traj)



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

    divided_out = [ep[5:-5] for ep in divided_out]

    length_threshold = 40
    return list(filter(lambda x: len(x) > length_threshold, divided_out))

out = clean_data(out)
# pdb.set_trace()



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
with open(save_path+'/normalization_arr/normalization_arr', 'wb') as pickle_file:
    pickle.dump(((x_mean_arr, x_std_arr),(y_mean_arr, y_std_arr)), pickle_file)


x_data = torch.tensor(x_data, dtype=dtype)
y_data = torch.tensor(y_data, dtype=dtype)

x_mean_arr = torch.tensor(x_mean_arr, dtype=dtype)
x_std_arr = torch.tensor(x_std_arr, dtype=dtype)
y_mean_arr = torch.tensor(y_mean_arr, dtype=dtype)
y_std_arr = torch.tensor(y_std_arr, dtype=dtype)
# out = [z_score_normalize]

if cuda:
    x_mean_arr = x_mean_arr.cuda()
    x_std_arr = x_std_arr.cuda()
    y_mean_arr = y_mean_arr.cuda()
    y_std_arr = y_std_arr.cuda()
    # 
out = [torch.tensor(ep, dtype=dtype) for ep in out]



if __name__ == "__main__":
    thresh = 10
    print('beginning run')

    for episode in out:
        states = run_traj(model, episode, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = None, return_states=True).cpu().detach().numpy()
            
        eps = episode.cpu().detach().numpy()
        plt.figure(1)
        plt.scatter(eps[0, 0], eps[0, 1], marker="*", label='start')
        plt.plot(eps[:, 0], eps[:, 1], color='blue', label='Ground Truth', marker='.')
        # plt.scatter(eps[0, 2], eps[0, 3], marker="*", label='start')
        # plt.plot(eps[:, 2], eps[:, 3], color='blue', label='Ground Truth', marker='.')
        # plt.plot(states[:, 0], states[:, 1], color='red', label='NN Prediction')
        plt.axis('scaled')
        plt.title('Bayesian NN Prediction -- pos Space')
        plt.legend()
        plt.show()


        