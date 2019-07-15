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
# nn_type = '1'
nn_type = 'LSTM'
SAVE = False
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
assert task in ['real_A', 'real_B', 'transferA2B', 'transferB2A']



state_dim = 4
action_dim = 6
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

if task in ['real_A', 'real_B']:
    if task == 'real_A':
        datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'
        save_path = 'save_model/robotic_hand_real/pytorch'
    elif task == 'real_B':
        datafile_name = 'data/robotic_hand_real/B/t42_cyl35_red_data_discrete_v0_d4_m1_episodes.obj'
        save_path = 'save_model/robotic_hand_real/pytorch'
    elif task == 'sim_A':
        datafile_name = '../data/robotic_hand_simulator/A/sim_data_discrete_v14_d4_m1_episodes.obj'
        save_path = '../save_model/robotic_hand_simulator/pytorch'
    elif task == 'sim_B':
        datafile_name = '../data/robotic_hand_simulator/B/sim_data_discrete_v14_d4_m1_modified_episodes.obj'
        save_path = '../save_model/robotic_hand_simulator/pytorch'



    save_path = 'save_model/robotic_hand_real/pytorch'
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

class BNNWrapper(torch.nn.Module):
    def __init__(self, model, input_dim, output_dim):
        super(BNNWrapper, self).__init__()
        self.model = torch.nn.LSTM(input_dim, output_dim, num_layers=3, dropout=dropout_p)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def get_distro(self,x):
        # pdb.set_trace()
        output = self.model(x)
        means = output[...,: self.output_dim]
        stds = F.elu(output[..., self.output_dim:]) + 1
            #Make sure Standard Deviation > 0
        # pdb.set_trace()
        distro = torch.distributions.normal.Normal(means, stds)
        return distro

    def forward(self,x, true_state=None):
        distro = self.get_distro(x)
        # sample = distro.sample()
        sample = distro.mean
        # interp = .9
        # sample = distro.mean*interp+ distro.sample()*(1-interp)
        if true_state is not None: 
            log_p = distro.log_prob(true_state)
            nan_locs = (log_p != log_p) #Get locations where log_p is undefined
            if nan_locs.any():
                pdb.set_trace()
            log_p[nan_locs] = 0 #Set the loss in those locations to 0
            return sample, log_p

        return sample




class BayesianTrainer:
    def __init__(self, task, norm, method=None, save=True, model_save_path=None):
        self.state_dim = 4
        self.action_dim = 6
        self.task = task
        self.norm = norm
        self.method = method
        self.save=save
        self.model_save_path = model_save_path

    def pretrain(self, model, x_data, y_data, opt, train_load = True, epochs = 30):
        dataset = torch.utils.data.TensorDataset(x_data, y_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size = 64)
        # loss_fn = torch.nn.NLLLoss()
        loss_fn = torch.nn.MSELoss()
        for i in range(epochs):
            print("Pretraining epoch: " + str(i))
            for batch_ndx, sample in enumerate(loader):
                opt.zero_grad()
                out, log_p = model(sample[0], sample[1])

                loss = -torch.sum(log_p[...,:self.state_dim])

                # pdb.set_trace()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                opt.step()

        if self.save:
            with open(self.model_save_path, 'wb') as pickle_file:
                torch.save(model, pickle_file)

    def run_traj_batch(self, model, batch, threshold = 50, return_states = False, 
        loss_type = 'softmax', alpha = .5):
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        true_states = batch[:,:,:self.state_dim]
        state = batch[:,0,:self.state_dim]
        states = []#state.view(1, self.state_dim)
        sim_deltas = []
        if cuda:
            state = state.cuda()
            true_states = true_states.cuda()

        mse_fn = torch.nn.MSELoss()

        loss = 0

        for i in range(batch.shape[1]):
            states.append(state)
            action = batch[:,i,self.state_dim:self.state_dim+self.action_dim]
            if cuda: action = action.cuda() 
            inpt = torch.cat((state, action), 1)

            inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

            # pdb.set_trace()
            residual = true_states[:,i] - state
            norm_res = z_score_norm_single(residual, y_mean_arr, y_std_arr)
            # pdb.set_trace()

            state_delta, hidden, log_p = model(inpt, residual)
            # l = -log_p[...,:self.state_dim]/((i+1)*batch.shape[1])
            l = -log_p[...,:2]/((i+1)*batch.shape[1])
            # pdb.set_trace()
            loss += l

            state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
            if self.task in ['transferA2B', 'transferB2A']: state_delta *= torch.tensor([-1,-1,1,1], dtype=dtype)
            sim_deltas.append(state_delta)

            state= state_delta + state           

        sim_deltas = torch.stack(sim_deltas, 1)
        states = torch.stack(states, 1)

        return torch.sum(loss), 1, batch.shape[1]


    def run_traj(self, model, traj, threshold = 50, return_states = False, 
        loss_type = 'softmax', alpha = .5):
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        # if type(traj) != type(torch.tensor(1)):
        #     pdb.set_trace() 

        true_states = traj[:,:self.state_dim]
        state = traj[0][:self.state_dim]
        states = []#state.view(1, self.state_dim)
        sim_deltas = []
        if cuda:
            state = state.cuda()
            true_states = true_states.cuda()

        mse_fn = torch.nn.MSELoss()

        hidden

        def softmax(states):
            mse_fn = torch.nn.MSELoss(reduction='none')
            mse = mse_fn(states[:,:2], true_states[:states.shape[0],:2])
            mse = torch.sum(mse, 1)  #Sum two position losses at each time step to get the Euclidean distance
            return torch.logsumexp(mse, 0)

        for i, point in enumerate(traj):
            states.append(state)
            action = point[self.state_dim:self.state_dim+self.action_dim]
            if cuda: action = action.cuda()   

            inpt = torch.cat((state, action), 0)
            inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

            state_delta = model(inpt)
            state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
            if self.task in ['transferA2B', 'transferB2A']: 
                state_delta *= torch.tensor([-1,-1,1,1], dtype=dtype)
            sim_deltas.append(state_delta)

            state= state_delta + state

            #May need random component here to prevent overfitting
            # states = torch.cat((states,state.view(1, self.state_dim)), 0)
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

        return softmax(states), 1, len(traj)

    def batch_train(self, model, opt, out, val_data = None, epochs = 500, batch_size = 8):
        j=0
        print('\nBatched trajectory training')
        for epoch in range(epochs):
            grad_norms = []

            print('Epoch: ' + str(epoch))
            np.random.shuffle(out)
            total_loss = 0
            # pdb.set_trace()
            total_completed = 0
            total_distance = 0
            switch = True
            # loss_type = 'stepwise'
            # loss_type = 'asdfsdf'
            loss_type = 'mix'
            thresh = 150


            batch_lists = [out[i: min(len(out), i+ batch_size)] for i in range(0, len(out), batch_size)] 
            episode_lengths = [[len(ep) for ep in batch] for batch in batch_lists]
            min_lengths = [min(episode_length) for episode_length in episode_lengths]
            rand_maxes = [[len(episode) - min_length for episode in batch_list] for batch_list, min_length in zip(batch_lists,min_lengths)]
            rand_starts = [[random.randint(0, rmax) for rmax in rmaxes] for rmaxes in rand_maxes]
            batch_slices = [[episode[start:start+length] for episode, start in zip(batch, starts)] for batch, starts, length in zip(batch_lists, rand_starts, min_lengths)]

            batches = [torch.stack(batch, 0) for batch in batch_slices] 

            accum = 8//batch_size

            for i, batch in enumerate(batches):
                if accum == 0 or j % accum ==0: opt.zero_grad()

                j += 1
                # if i % 30 == 0:
                    # print(i*batch_size)
                loss, completed, dist = self.run_traj_batch(model, batch, threshold = thresh, loss_type=loss_type)
                total_loss += loss.data
                total_completed += completed
                total_distance += dist
                # pdb.set_trace()

                loss.backward()
                if accum == 0 or j % accum ==0: 
                    if self.task == 'transferA2B' and method in ['constrained_retrain', 'constrained_restart']:
                        loss = offset_l2(model)*l2_coeff*accum
                        loss.backward()

                    grad_norms.append(grad_norm(model))

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    opt.step()

            if val_data:
                total_loss = 0
                total_completed = 0
                total_distance = 0
                for i, episode in enumerate(val_data[:len(val_data)//2]):
                    _ , completed, dist = self.run_traj(model, episode, threshold = 50)
                    total_completed += completed
                    total_distance += dist

                for i, episode in enumerate(val_data[len(val_data)//2:]):
                    val_loss, completed, dist = self.run_traj(model, episode, threshold = None)
                    total_loss += val_loss.data
                print('Loss: ' + str(total_loss/len(val_data)))
                print('completed: ' + str(total_completed/len(val_data)))
                print('Average time before divergence: ' + str(total_distance/len(val_data)))

            else:
                print('Loss: ' + str(total_loss/len(batches)))
                print('completed: ' + str(total_completed/len(batches)))
                print('Average time before divergence: ' + str(total_distance/len(batches)))

            if self.save:
                with open(self.model_save_path, 'wb') as pickle_file:
                    torch.save(model, pickle_file)





if __name__ == "__main__":
    thresh = 10
    model = BNNWrapper(model, input_dim, state_dim)
    norm = x_mean_arr, x_std_arr, y_mean_arr, y_std_arr
    trainer = BayesianTrainer(task, norm, model_save_path=model_save_path) 

    np.random.shuffle(out)

    lr = .00005
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
    trainer.pretrain(model, x_data, y_data, opt, epochs=5)
    # if method == 'nonlinear_transform':
    #     model.set_base_model_train(True)
    opt = torch.optim.Adam(model.parameters(), lr=.0000025, weight_decay=.001)
    trainer.batch_train(model, opt, out, val_data =val_data, epochs=100, batch_size=128)
