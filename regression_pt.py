import scipy.io
# from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb
import torch
from common.data_normalization import *
from common.pt_build_model import pt_build_model
import random
import matplotlib.pyplot as plt

from sys import argv

data_type = 'pos' #type of data used for this task
task = 'real_A' #Which task we're training. This tells us what file to use
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
assert task in ['real_A', 'real_B', 'transfer']



state_dim = 4
action_dim = 6
alpha = .4
lr = .0003

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

    model = pt_build_model(nn_type, state_dim+action_dim, state_dim, .1)
    if cuda: 
        model = model.cuda()


elif task == 'transfer':

    method = 'retrain'
    if len(argv) > 4:
        method = argv[4]


    datafile_name = 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1_episodes.obj'
    save_path = 'save_model/robotic_hand_real/pytorch'
    with open(datafile_name, 'rb') as pickle_file:
        out = pickle.load(pickle_file, encoding='latin1')

    with open(save_path+'/real_A'+ '_' + nn_type + '.pkl', 'rb') as pickle_file:
        if cuda: model = torch.load(pickle_file).cuda()
        else: model = torch.load(pickle_file, map_location='cpu')

    if method == 'constrained_retrain':
        mse_fn = torch.nn.MSELoss()
        base_parameters = model.parameters().clone().detach()
        def offset_l2(new_model):
            loss = 0
            for i,parameter in enumerate(new_model.parameters()):
                loss += torch.norm(base_parameters[i], parameter)

            return loss


    if method == 'linear_transform':
        class LinearTransformedModel(torch.nn.Module):
            def __init__(self, old_model, input_dim, output_dim):
                self.input_dim = input_dim
                self.output_dim = output_dim

                self.model = old_model
                for param in self.model.parameters():
                    param.requires_grad = False

                # self.lt = torch.nn.Linear(input_dim, input_dim)
                # self.lt_inv = torch.nn.Linear(output_dim, output_dim)

                self.mat = torch.autograd.Variable(np.identity(input_dim), requires_grad=True)
                self.mat_inv = self.mat[:output_dim, :output_dim].transpose()
                    # Initialize transform as identity matrix
            def forward(self, inpt):
                # return self.lt(self.model(self.lt_inv(inpt)))

                trans_in = torch.mm(mat, inpt)
                out = self.model(trans_in)
                detrans_out = torch.mm(mat_inv, out)

                return detrans_out

            def get_constistency_loss(self, inpt):
                # trans_state = self.lt(inpt)[:self.output_dim]
                # detrans_state = self.lt_inv(trans_state)
                # mse_fn = torch.nn.MSELoss()
                # return mse_fn(detrans_state, inpt[:self.output_dim])


                trans_state = torch.mm(mat, inpt)[:self.output_dim]
                detrans_state = torch.mm(mat_inv, trans_state)
                mse_fn = torch.nn.MSELoss()
                return mse_fn(detrans_state, inpt[:self.output_dim])


        model = LinearTransformedModel(model, state_dim + action_dim, state_dim)




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

        state_delta = model(inpt)
        state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
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

    np.random.shuffle(out)
    val_data = out[int(len(out)*(1-held_out)):]
    out = out[:int(len(out)*(1-held_out))]
    for epoch in range(500):
        print('Epoch: ' + str(epoch))
        np.random.shuffle(out)
        total_loss = 0
        # pdb.set_trace()
        total_completed = 0
        total_distance = 0
        switch = True
        if epoch < 5: 
            loss_type = 'stepwise'
            thresh = None
        # elif epoch < 6: 
        #     loss_type = 'mix'
        #     thresh = 100
        else: 
            if switch == True:
                switch = False
                opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss_type = 'softmax'
            thresh = 150



        for i, episode in enumerate(out):
            if i % 30 == 0:
                print(i)
            loss, completed, dist = run_traj(model, episode, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = thresh, loss_type=loss_type)

            if task == 'transfer':
                if method == 'constrained_retrain':
                    factor = .0001
                    loss += offset_l2(model)*factor
                if method == 'linear_transform':
                    factor = .0001
                    loss += model.get_constistency_loss(episode)*factor

            opt.zero_grad()
            loss.backward()
            opt.step()

        for i, episode in enumerate(val_data):
            loss, completed, dist = run_traj(model, episode, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = 50)
            total_loss += loss.data
            total_completed += completed
            total_distance += dist

        # episode = random.choice(val_data)
        # states = run_traj(model, episode, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = None, return_states=True).cpu().detach().numpy()
            
        # eps = episode.cpu().detach().numpy()
        # plt.figure(1)
        # plt.scatter(eps[0, 0], eps[0, 1], marker="*", label='start')
        # plt.plot(eps[:, 0], eps[:, 1], color='blue', label='Ground Truth', marker='.')
        # plt.plot(states[:, 0], states[:, 1], color='red', label='NN Prediction')
        # plt.axis('scaled')
        # plt.title('Bayesian NN Prediction -- pos Space')
        # plt.legend()
        # plt.show()


        thresh = 150
        print('Loss: ' + str(total_loss/len(val_data)))
        print('completed: ' + str(total_completed/len(val_data)))
        print('Average time before divergence: ' + str(total_distance/len(val_data)))
        with open(save_path+'/'+ task + '_' + nn_type + '.pkl', 'wb') as pickle_file:
            torch.save(model, pickle_file)

    if outfile: 
        if append:
            f = open(outfile, 'a+')
        else:
            f = open(outfile, 'w+')
        out_string= ('cold start\t' + data_type +
                    '\t' + task + '\t' + str(held_out) +
                    '\t:' + str(final_loss) + '\n')
        f.write(out_string)
        f.close()
    