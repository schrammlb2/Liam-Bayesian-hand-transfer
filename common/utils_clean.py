import numpy as np 
import pdb
import torch
import torch.utils.data
from common.data_normalization import *
import random
import pickle

import matplotlib.pyplot as plt


cuda = torch.cuda.is_available()
cuda = False
dtype = torch.float


def weight_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item()**2
    return total_norm**(.5)
    return 

def grad_norm(model):
    # return 0
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            try: 
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item()**2
            except: 
                pass
    return total_norm**(.5)


def clean_data(episodes):
    DATA = np.concatenate(episodes)
    yd_pos = DATA[:, -4:-2] - DATA[:, :2]
    y2 = np.sum(yd_pos**2, axis=1)
    max_dist = np.percentile(y2, 99.84)
    # max_dist = np.percentile(y2, 99.6)

    skip_list = [np.sum((ep[:, -4:-2] - ep[:, :2])**5, axis=1)>max_dist for ep in episodes]
    divided_episodes = []
    for i,ep in enumerate(episodes):
        if np.sum(skip_list[i]) == 0:
            divided_episodes += [ep]

        else: 
            ep_lists = np.split(ep, np.argwhere(skip_list[i]).reshape(-1))
            divided_episodes += ep_lists

    divided_episodes = [ep[3:-3] for ep in divided_episodes]

    length_threshold = 30
    return list(filter(lambda x: len(x) > length_threshold, divided_episodes))


def softmax(states, true_states):
    mse_fn = torch.nn.MSELoss(reduction='none')
    mse = mse_fn(states[...,:,:2], true_states[...,:states.shape[-2],:2])
    mse = torch.mean(mse, -1) #Sum two position losses at each time step to get the Euclidean distance 
    loss = torch.logsumexp(mse, -1) #Softmax divergence over the path
    loss = torch.mean(loss) #Sum over batch
    return loss

def pointwise(states, true_states):
    mse_fn = torch.nn.MSELoss(reduction='none')
    scaling = 1/((torch.arange(states.shape[1], dtype=torch.float)+1))
    if cuda: scaling = scaling.cuda()
    loss_temp = mse_fn(states[...,:,:2], true_states[...,:states.shape[-2],:2])
    loss = torch.einsum('...kj,k->', [loss_temp, scaling])/loss_temp.numel()
    return loss


class Trainer():
    def __init__(self, task, episodes,  method=None, save=True, model_save_path=None, save_path = None, state_dim = 4, action_dim = 6, task_ofs = 10, reg_loss = None):
        self.task_ofs = task_ofs
        self.state_dim = state_dim
        self.new_state_dim = state_dim
        self.action_dim = action_dim
        self.task = task
        self.method = method
        self.save=save
        self.save_path = save_path
        self.model_save_path = model_save_path
        self.dtype = torch.float
        self.cuda = cuda

        self.norm, data = self.get_norms(episodes)
        self.x_data, self.y_data = data
        self.episodes = episodes
        # if reg_loss == None:
        #     self.reg_loss = lambda m: 0
        #     self.reg_loss = False
        # else:
        self.reg_loss = reg_loss        




    def pretrain(self, model, opt, train_load = True, epochs = 30, batch_size = 64):
        dataset = torch.utils.data.TensorDataset(self.x_data, self.y_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
        loss_fn = torch.nn.MSELoss()
        for i in range(epochs):
            print("Pretraining epoch: " + str(i))
            total_loss = 0
            for batch_ndx, sample in enumerate(loader):
                opt.zero_grad()

                if cuda:
                    sample[0]  = sample[0].cuda()
                    sample[1]  = sample[1].cuda()

                output = model(sample[0])

                if self.task in ['transferA2B', 'transferB2A']: 
                    output *= torch.tensor([-1,-1,1,1], dtype=dtype)

                loss = loss_fn(output, sample[1]) 
                if self.reg_loss != None: 
                    loss += self.reg_loss(model)
                total_loss += loss.data
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                opt.step()
            print("total_loss: " + str(total_loss))
        # if self.save:
        with open(self.model_save_path, 'wb') as pickle_file:
            torch.save(model, pickle_file)



    def get_norms(self, episodes):
        DATA = np.concatenate(episodes)
        FULL_DATA = np.concatenate(episodes)

        x_data = DATA[:, :self.task_ofs]
        y_data = DATA[:, -self.new_state_dim:] - DATA[:, :self.new_state_dim]


        full_x_data = FULL_DATA[:, :self.task_ofs]
        full_y_data = FULL_DATA[:, -self.new_state_dim:] - FULL_DATA[:, :self.new_state_dim]

        x_mean_arr = np.mean(x_data, axis=0)
        x_std_arr = np.std(x_data, axis=0)
        y_mean_arr = np.mean(y_data, axis=0)
        y_std_arr = np.std(y_data, axis=0)

        x_data = z_score_normalize(x_data, x_mean_arr, x_std_arr)
        y_data = z_score_normalize(y_data, y_mean_arr, y_std_arr)
        if self.save_path:
            with open(self.save_path+'/normalization_arr/normalization_arr', 'wb') as pickle_file:
                pickle.dump(((x_mean_arr, x_std_arr),(y_mean_arr, y_std_arr)), pickle_file)

        x_data = torch.tensor(x_data, dtype=self.dtype)
        y_data = torch.tensor(y_data, dtype=self.dtype)

        x_mean_arr = torch.tensor(x_mean_arr, dtype=self.dtype)
        x_std_arr = torch.tensor(x_std_arr, dtype=self.dtype)
        y_mean_arr = torch.tensor(y_mean_arr, dtype=self.dtype)
        y_std_arr = torch.tensor(y_std_arr, dtype=self.dtype)

        if self.cuda:
            x_mean_arr = x_mean_arr.cuda()
            x_std_arr = x_std_arr.cuda()
            y_mean_arr = y_mean_arr.cuda()
            y_std_arr = y_std_arr.cuda()

        return (x_mean_arr, x_std_arr, y_mean_arr, y_std_arr), (x_data, y_data)


    def visualize(self, model,  episode):
        states = self.run_traj_batch(model, episode, threshold = None, return_states=True)

        episode = episode.detach().numpy()
        states = states.detach().numpy()

        plt.figure(1)
        plt.scatter(episode[...,0, 0], episode[..., 0, 1], marker="*", label='start')
        plt.plot(episode[...,:, 0], episode[..., :, 1], color='blue', label='Ground Truth', marker='.')
        plt.plot(states[..., :, 0], states[ ...,:, 1], color='red', label='NN Prediction')
        plt.axis('scaled')
        plt.title('Bayesian NN Prediction -- pos Space')
        plt.legend()
        plt.show()


    def run_traj_batch(self, model, batch, threshold = 50, return_states = False, 
        loss_type = 'softmax', alpha = .5):
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        true_states = batch[...,:,:self.state_dim]
        state = batch[...,0,:self.state_dim]
        states = []#state.view(1, self.state_dim)
        sim_deltas = []
        if cuda:
            state = state.cuda()
            true_states = true_states.cuda()

        mse_fn = torch.nn.MSELoss()

        def get_loss(loss_type, states, true_states):
            if loss_type in ['soft maximum', 'softmax']:
                loss = softmax(states, true_states)

            elif loss_type == 'mix':
                loss= pointwise(states, true_states)
                alpha = .9
                loss*= alpha
                loss += softmax(states, true_states)*(1-alpha)
                return loss

            elif loss_type == 'pointwise':
                loss= pointwise(states, true_states)

            return loss

        for i in range(batch.shape[1]):
            states.append(state)
            action = batch[...,i,self.state_dim:self.state_dim+self.action_dim]
            if cuda: action = action.cuda() 
            inpt = torch.cat((state, action), -1)
            inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

            state_delta = model(inpt)

            state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
            if self.task in ['transferA2B', 'transferB2A']: 
                state_delta *= torch.tensor([-1,-1,1,1], dtype=torch.float)
            sim_deltas.append(state_delta)

            state= state_delta + state

            #May need random component here to prevent overfitting
            if threshold and i%10:
                with torch.no_grad():
                    mse = mse_fn(state[...,:2], true_states[...,i,:2])
                if mse > threshold:
                    states = torch.stack(states, -2)
                    # sim_deltas = torch.stack(sim_deltas, 0)
                    loss = softmax(states, true_states)
                    return loss, 0, i

        # sim_deltas = torch.stack(sim_deltas, 1)
        states = torch.stack(states, -2)
        if return_states:
            return states

        return get_loss(loss_type, states, true_states), 1, batch.shape[-2]


    #------------------------------------------------------------------------------------------------------------------------------------

    def batch_train(self, model, opt, val_data = None, epochs = 500, batch_size = 8, loss_type = 'pointwise'):
        j=0
        episodes= self.episodes
        print('\nBatched trajectory training with batch size ' + str(batch_size))
        for epoch in range(epochs):
            grad_norms = []

            print('Epoch: ' + str(epoch))
            np.random.shuffle(episodes)

            total_loss = 0
            # pdb.set_trace()
            total_completed = 0
            total_distance = 0
            switch = True
            thresh = 150


            batch_lists = [episodes[i: min(len(episodes), i+ batch_size)] for i in range(0, len(episodes), batch_size)] 
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
                loss, completed, dist = self.run_traj_batch(model, batch, threshold = thresh, loss_type=loss_type)
                total_loss += loss.data
                total_completed += completed
                total_distance += dist               

                loss.backward()
                if accum == 0 or j % accum ==0:
                    if self.reg_loss: loss += self.reg_loss(model)

                    grad_norms.append(grad_norm(model))

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    opt.step()

            if val_data:
                total_loss = 0
                total_completed = 0
                total_distance = 0
                for i, episode in enumerate(val_data[:len(val_data)//2]):
                    _ , completed, dist = self.run_traj_batch(model, episode, threshold = 50)
                    total_completed += completed
                    total_distance += dist

                for i, episode in enumerate(val_data[len(val_data)//2:]):
                    val_loss, completed, dist = self.run_traj_batch(model, episode, threshold = None)
                    total_loss += val_loss.data
                print('Loss: ' + str(total_loss/(len(val_data)/2)))
                print('completed: ' + str(total_completed/(len(val_data)/2)))
                print('Average time before divergence: ' + str(total_distance/(len(val_data)/2)))

                episode = random.choice(val_data)
                # self.visualize(model, episode)


            else:
                print('Loss: ' + str(total_loss/len(batches)))
                print('completed: ' + str(total_completed/len(batches)))
                print('Average time before divergence: ' + str(total_distance/len(batches)))

            # if self.save:
            with open(self.model_save_path, 'wb') as pickle_file:
                torch.save(model, pickle_file)
