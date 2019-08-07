import torch
import torch.nn.functional as F
from common.data_normalization import *
from common.pt_build_model import *
import pdb
import numpy as np
import random

dtype = torch.float
cuda = False

class ResBlock(torch.nn.Module):
    def __init__(self, model):
        self.model = model
    def forward(self, x):
        return x + self.model(x)

class TrajNet(torch.nn.Module):
    def __init__(self, task, norms, model = None ,state_dim = 4, action_dim = 6, task_ofs = 10, reg_loss = None):
        super(TrajNet, self).__init__()
        self.task_ofs = task_ofs
        self.state_dim = state_dim
        self.task = task
        # self.action_dim = action_dim
        self.action_dim = 2
        self.dtype = torch.float
        self.cuda = cuda

        self.norm = norms
        self.reg_loss = reg_loss 
        if model:
            self.model = model
        else:
            self.model = pt_build_model('1', state_dim + action_dim, state_dim, dropout_p=.1)

    def run_traj(self, batch, threshold = 50, sub_chance = 0.0):
        dim = 3
        if len(batch.shape) == 2:
            batch = batch.unsqueeze(0)
            dim = 2
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        true_states = batch[...,:,:self.state_dim]
        state = batch[...,0,:self.state_dim]
        states = []#state.view(1, self.state_dim)
        sim_deltas = []
        if cuda:
            state = state.cuda()
            true_states = true_states.cuda()

        for i in range(batch.shape[1]):
            states.append(state)
            action = batch[...,i,self.state_dim:self.state_dim+self.action_dim]
            if cuda: action = action.cuda() 
            inpt = torch.cat((state, action), -1)
            inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

            state_delta = self.model(inpt)

            state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
            if self.task in ['transferA2B', 'transferB2A']: 
                state_delta *= torch.tensor([-1,-1,1,1], dtype=torch.float)
            sim_deltas.append(state_delta)

            state= state_delta + state

            #May need random component here to prevent overfitting
            if threshold and i%10:
                with torch.no_grad():
                    mse_fn = torch.nn.MSELoss()
                    mse = mse_fn(state[...,:2], true_states[...,i,:2])
                if mse > threshold:
                    states = torch.stack(states, -2)
                    return states

        states = torch.stack(states, -2)
        if dim == 2:
            states=states.squeeze(0)
        return states

    def forward(self, x):
        return self.model(x)


class LSTMStateTrajNet(torch.nn.Module):
    def __init__(self, task, norms, model = None ,state_dim = 4, action_dim = 2, task_ofs = 10, reg_loss = None):
        super(LSTMStateTrajNet, self).__init__()
        self.task_ofs = task_ofs
        self.state_dim = state_dim
        self.task = task
        # self.action_dim = action_dim
        self.action_dim = 2
        self.dtype = torch.float
        self.cuda = cuda

        self.norm = norms
        self.reg_loss = reg_loss 
        self.LSTM = False
        h = 100
        self.h = h
        # self.l1 = torch.nn.GRU(action_dim+state_dim, state_dim, dropout = .1, batch_first = True)
        self.l1 = torch.nn.GRU(action_dim+state_dim, h, dropout = .25, batch_first = True, num_layers=2)
        self.l2 = torch.nn.GRU(h, state_dim, dropout = .25, batch_first = True)
        self.prediction = 'state'

    def run_traj(self, batch, threshold = 50, sub_chance = 0.0):
        return self.forward(batch[...,:self.state_dim+self.action_dim])

    def forward(self, inpt, given_hidden = None):
        hidden = given_hidden

        if len(inpt.shape) == 1:
            inpt = inpt.unsqueeze(0)
        if len(inpt.shape) == 2:
            inpt = inpt.unsqueeze(1)

        if hidden == None or hidden == 'fill':
            m1 = torch.zeros(2, inpt.shape[0] ,self.h)
            sd1 = torch.ones(2, inpt.shape[0] ,self.h)

            m2 = torch.zeros(1, inpt.shape[0] ,self.state_dim)
            sd2 = torch.ones(1, inpt.shape[0] ,self.state_dim)

            h1_distro = torch.distributions.normal.Normal(m1, sd1)
            h2_distro = torch.distributions.normal.Normal(m2, sd2)

            if self.LSTM:
                hc1 = (h1_distro.sample(), h1_distro.sample())
                hc2 = (h2_distro.sample(), h2_distro.sample())
                hidden = (hc1, hc2)
            else:
                # self.prediction = 'delta'
                if self.prediction == 'state':
                    hidden = (h1_distro.sample(), inpt.transpose(0,1)[:1,:,:self.state_dim])
                elif self.prediction == 'delta':
                    hidden = (h1_distro.sample(), h2_distro.sample())
                else:
                    pdb.set_trace()

        f1, h1 = self.l1(inpt, hidden[0])    
        # dyn = self.dyn(inpt)
        mid = F.dropout(f1, .2)
        # mid += dyn 
        f2, h2 = self.l2(mid, hidden[1])
        out = f2.squeeze(1)

        if given_hidden == None:
            return out
        return out, (h1,h2)

class LSTMTrajNet(LSTMStateTrajNet):
    def __init__(self, task, norms, model = None ,state_dim = 4, action_dim = 2, task_ofs = 10, reg_loss = None):
        super(LSTMTrajNet, self).__init__(task, norms)
        self.task_ofs = task_ofs
        self.state_dim = state_dim
        self.task = task
        # self.action_dim = action_dim
        self.action_dim = 2
        self.dtype = torch.float
        self.cuda = cuda

        self.norm = norms
        self.reg_loss = reg_loss 
        self.LSTM = False
        h = 100
        self.h = h
        # self.l1 = torch.nn.GRU(action_dim+state_dim, state_dim, dropout = .1, batch_first = True)
        self.l1 = torch.nn.GRU(action_dim+state_dim, h, dropout = .25, batch_first = True, num_layers=2)
        self.l2 = torch.nn.GRU(h, state_dim, dropout = .25, batch_first = True)
        self.prediction = 'delta'
        self.model = pt_build_model('1', state_dim + action_dim, state_dim, dropout_p=.1)


    def run_traj(self, batch, threshold = 50, sub_chance = 0.0):
        dim = 3
        if len(batch.shape) == 2:
            batch = batch.unsqueeze(0)
            dim = 2
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        true_states = batch[...,:,:self.state_dim]
        state = batch[...,0,:self.state_dim]
        states = []#state.view(1, self.state_dim)
        sim_deltas = []
        if cuda:
            state = state.cuda()
            true_states = true_states.cuda()

        hidden = 'fill'
        for i in range(batch.shape[1]):
            states.append(state)

            if random.random() < sub_chance: 
                state=true_states[:,i]

            action = batch[...,i,self.state_dim:self.state_dim+self.action_dim]
            if cuda: action = action.cuda() 
            inpt = torch.cat((state, action), -1)
            inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

            state_delta, hidden = self.forward(inpt, hidden)

            state_delta += self.model(inpt)

            state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
            if self.task in ['transferA2B', 'transferB2A']: 
                state_delta *= torch.tensor([-1,-1,1,1], dtype=torch.float)
            sim_deltas.append(state_delta)

            state= state_delta + state

            #May need random component here to prevent overfitting
            if threshold and i%10:
                with torch.no_grad():
                    mse_fn = torch.nn.MSELoss()
                    mse = mse_fn(state[...,:2], true_states[...,i,:2])
                if mse > threshold:
                    states = torch.stack(states, -2)
                    return states

        states = torch.stack(states, -2)
        if dim == 2:
            states=states.squeeze(0)
        return states

    def forward(self, inpt, hidden=None):
        if hidden:
            return super(LSTMTrajNet, self).forward(inpt, hidden)
        else: 
            return self.model(inpt)



class NBackNet(torch.nn.Module):
    def __init__(self, task, norms, model = None ,state_dim = 4, action_dim = 2, task_ofs = 10, reg_loss = None, n=5):
        super(NBackNet, self).__init__()
        self.task_ofs = task_ofs
        self.state_dim = state_dim
        self.task = task
        # self.action_dim = action_dim
        self.action_dim = 2
        self.dtype = torch.float
        self.cuda = cuda

        self.norm = norms
        self.reg_loss = reg_loss   
        self.n = n
        if model:
            self.model = model
        else:
            self.model = pt_build_model('1', (state_dim + action_dim)*5, state_dim, dropout_p=.1)

    def run_traj(self, batch, threshold = 50, sub_chance = 0.0, n=5):
        if len(batch.shape) == 2:
            batch = batch.unsqueeze(0)
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        true_states = batch[...,:,:self.state_dim]
        state = batch[...,0,:self.state_dim]
        states = []#state.view(1, self.state_dim)
        sim_deltas = []
        if cuda:
            state = state.cuda()
            true_states = true_states.cuda()

        state_action = batch[...,0,:self.state_dim+self.action_dim]
        nback = torch.stack([state_action]*self.n, -2)

        for i in range(batch.shape[1]):
            states.append(state)

            if random.random() < sub_chance: 
                state=true_states[:,i]

            action = batch[...,i,self.state_dim:self.state_dim+self.action_dim]
            # action = batch[...,i,self.state_dim:self.state_dim+self.action_dim]
            if cuda: action = action.cuda() 
            if self.task in ['transferA2B', 'transferB2A']: 
                action[...,:2] *= -1
            inpt = torch.cat((state, action), -1)
            inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)
            inpt = inpt.unsqueeze(1)

            inpt = torch.cat([nback[:,1:], inpt], 1)
            model_inpt = inpt.view(inpt.shape[0], -1)

            state_delta = self.model(model_inpt)

            state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
            sim_deltas.append(state_delta)

            state= state_delta + state

            #May need random component here to prevent overfitting
            if threshold and i%10:
                with torch.no_grad():
                    mse_fn = torch.nn.MSELoss()
                    mse = mse_fn(state[...,:2], true_states[...,i,:2])
                if mse > threshold:
                    states = torch.stack(states, -2)
                    return states

        states = torch.stack(states, -2)
        return states

    def forward(self, x):
        nback = torch.cat([x]*self.n, -1)
        return self.model(nback)


class LatentNet(torch.nn.Module):
    def __init__(self, task, norms, state_dim = 4, action_dim = 6, task_ofs = 10, reg_loss = None):
        super(LatentNet, self).__init__()
        self.task_ofs = task_ofs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dtype = torch.float
        self.cuda = cuda

        self.norm = norms
        self.reg_loss = reg_loss   
        self.internal_state_dim = state_dim
        self.model = TrajNet(task, norms)
        self.encoder = ResBlock(pt_build_model('1', state_dim, state_dim, dropout_p=.1))
        self.decoder = ResBlock(pt_build_model('1', state_dim, state_dim, dropout_p=.1))


    def run_traj(self, batch, threshold = 50, sub_chance=0.0):
        true_states = batch[...,:,:self.state_dim]
        actions = batch[...,:,self.state_dim:self.state_dim+self.action_dim]

        encoded_true_states = self.encoder(true_states)

        pass_batch = torch.cat([encoded_true_states, actions], -1)
        projected_states = self.model(encoded_true_states, threshold=threshold, sub_chance=sub_chance)

        states = torch.stack(states, -2)
        return states

    def forward(self, x):
        return self.model(x)

