import numpy as np 
import pdb
import torch
import torch.utils.data
from common.data_normalization import *
import random



cuda = torch.cuda.is_available()


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

    divided_out = [ep[3:-3] for ep in divided_out]

    length_threshold = 30
    return list(filter(lambda x: len(x) > length_threshold, divided_out))


class Trainer():
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
                # pdb.set_trace()
                out = model(sample[0])
                # if task in ['transferA2B', 'transferB2A']: 
                #     out *= torch.tensor([-1,-1,1,1], dtype=dtype)

                if train_load:
                    loss = loss_fn(out, sample[1]) 
                else:
                    loss = loss_fn(out[:,:2], sample[1][:,:2])

                # loss += l2_coeff*offset_l2(model)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                opt.step()
            # diagnostics(model, train_type='pretrain', epoch=i)

        if self.save:
            with open(self.model_save_path, 'wb') as pickle_file:
                torch.save(model, pickle_file)


    def get_norms(self, x_data, y_data):
        x_mean_arr = np.mean(x_data, axis=0)
        x_std_arr = np.std(x_data, axis=0)
        y_mean_arr = np.mean(y_data, axis=0)
        y_std_arr = np.std(y_data, axis=0)
        x_data = z_score_normalize(x_data, x_mean_arr, x_std_arr)
        y_data = z_score_normalize(y_data, y_mean_arr, y_std_arr)
        if self.save_path:
            with open(self.save_path+'/normalization_arr/normalization_arr', 'wb') as pickle_file:
                pickle.dump(((x_mean_arr, x_std_arr),(y_mean_arr, y_std_arr)), pickle_file)


        x_mean_arr = torch.tensor(x_mean_arr, dtype=self.dtype)
        x_std_arr = torch.tensor(x_std_arr, dtype=self.dtype)
        y_mean_arr = torch.tensor(y_mean_arr, dtype=self.dtype)
        y_std_arr = torch.tensor(y_std_arr, dtype=self.dtype)

        if self.cuda:
            x_mean_arr = x_mean_arr.cuda()
            x_std_arr = x_std_arr.cuda()
            y_mean_arr = y_mean_arr.cuda()
            y_std_arr = y_std_arr.cuda()

        return (x_mean_arr, x_std_arr, y_mean_arr, y_std_arr)


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
            action = point[self.state_dim:self.state_dim+self.action_dim]
            if cuda: action = action.cuda()   

            # if type(state) != type(torch.tensor(1)) or type(action) != type(torch.tensor(1)):
            #     pdb.set_trace() 
            inpt = torch.cat((state, action), 0)
            inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

            # pdb.set_trace()
            state_delta = model(inpt)
            state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
            # if self.task in ['transferA2B', 'transferB2A']: 
            #     state_delta *= torch.tensor([-1,-1,1,1], dtype=dtype)
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
        if return_states:
            return states


        return get_loss(loss_type, states=states, sim_deltas=sim_deltas), 1, len(traj)


    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

        def softmax(states):
            # pdb.set_trace()
            mse_fn = torch.nn.MSELoss(reduction='none')
            # mse_fn = torch.nn.MSELoss()
            msepos = mse_fn(states[:,:,:2], true_states[:,:states.shape[1],:2])
            mseload = mse_fn(states[:,:,2:4], true_states[:,:states.shape[1],2:4])
            alpha = .95
            mse = alpha*msepos + (1-alpha)*mseload
            mse = torch.mean(mse, 2) #Sum two position losses at each time step to get the Euclidean distance 
            # mse = torch.sum(mse, (1,2))     #Or sum to find the maximum divergence in the batch and emphasize that
            loss = torch.logsumexp(mse, 1) #Softmax divergence over the path
            loss = torch.mean(loss) #Sum over batch
            # loss = torch.logsumexp(loss, 0)
            return loss

        def stepwise(sim_deltas):
            # sim_deltas = states[1:, :2] - states[:-1, :2] #starting position version
            # real_deltas = traj[1:, :2] - traj[:-1, :2] #starting position version
            # real_deltas = traj[:, :2] - traj[:, -4:-2] # y_data version
            real_deltas = batch[:,:,:2] - batch[:,:,-4:-2]
            # real_deltas = batch[:,:,:4] - batch[:,:,-4:]
            mse_fn = torch.nn.MSELoss()
            # pdb.set_trace()
            mse = mse_fn(sim_deltas[:,:,:2], real_deltas[:,:sim_deltas.shape[1],:])
            return mse

        def mix(sim_deltas, states, alpha = .9):
            return stepwise(sim_deltas)*alpha + softmax(states)*(1-alpha)

        def get_loss(loss_type, states = None, sim_deltas = None):
            if loss_type == 'soft maximum':
                loss = softmax(states)
            elif loss_type == 'mix':
                loss = mix(sim_deltas, states)
                return loss
            elif loss_type == 'stepwise':
                loss = stepwise(sim_deltas)
                return loss
            else:
                mse_fn = torch.nn.MSELoss(reduction='none')
                scaling = 1/((torch.arange(states.shape[1], dtype=torch.float)+1)*states.shape[1])
                loss_temp = mse_fn(states[:,:,:2], true_states[:,:states.shape[1],:2])
                # loss = sum([loss_temp[:,i,:]*scale for i, scale in enumerate(scaling)])
                # pdb.set_trace()
                loss = torch.einsum('ikj,k->', [loss_temp, scaling])
                # loss = mse_fn(states[:,:2], true_states[:,:2])

            return loss

        for i in range(batch.shape[1]):
            states.append(state)
            action = batch[:,i,self.state_dim:self.state_dim+self.action_dim]
            if cuda: action = action.cuda() 
            inpt = torch.cat((state, action), 1)
            inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

            state_delta = model(inpt)
            state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
            # if self.task in ['transferA2B', 'transferB2A']: state_delta *= torch.tensor([-1,-1,1,1], dtype=dtype)
            sim_deltas.append(state_delta)

            state= state_delta + state

            #May need random component here to prevent overfitting
            # states = torch.cat((states,state.view(1, self.state_dim)), 0)
            if threshold and i%10:
                with torch.no_grad():
                    mse = mse_fn(state[:,:2], true_states[:,i,:2])
                if mse > threshold:
                    states = torch.stack(states, 1)
                    sim_deltas = torch.stack(sim_deltas, 0)
                    # loss = mix(sim_deltas, states, alpha)
                    loss = softmax(states)
                    # loss = get_loss(loss_type, states=states, sim_deltas=sim_deltas)
                    return loss, 0, i
                    # return mse_fn(state[:2], true_states[i,:2]), 0, i

        sim_deltas = torch.stack(sim_deltas, 1)
        states = torch.stack(states, 1)
        if return_states:
            return states

        return get_loss(loss_type, states=states, sim_deltas=sim_deltas), 1, batch.shape[1]


    #------------------------------------------------------------------------------------------------------------------------------------

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
            loss_type = 'asdfsdf'
            # loss_type = 'mix'
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
                    if self.task == 'transferA2B':
                        if method in ['constrained_retrain', 'constrained_restart']:
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
            # diagnostics(model, train_type='Batch Trajectory', loss=(total_loss/len(batches)), epoch=epoch, grad_norms=grad_norms)

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def traj_train(self,model, opt, out, val_data, epochs = 12):
        thresh = 10
        print('\nTrajectory training')
        j = 0

        for epoch in range(epochs):
            grad_norms = []
            # print('Epoch: ' + str(epoch))
            np.random.shuffle(out)
            total_loss = 0
            # pdb.set_trace()
            total_completed = 0
            total_distance = 0
            switch = True
            loss_type = 'softmax'
            thresh = 150

            accum = 8



            for i, episode in enumerate(out):
                if j % accum ==0: opt.zero_grad()

                j += 1
                # if i % 30 == 0:
                #     print(i)

                loss, completed, dist = self.run_traj(model, episode, threshold = thresh, loss_type=loss_type)
                loss.backward()

                if j % accum ==0: 
                    if self.task == 'transferA2B':
                        if method in ['constrained_retrain', 'constrained_restart']:
                            loss = offset_l2(model)*l2_coeff*accum
                            loss.backward()

                    grad_norms.append(grad_norm(model))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    opt.step()

            for i, episode in enumerate(val_data[:len(val_data)//2]):
                _ , completed, dist = self.run_traj(model, episode, threshold = 50)
                total_completed += completed
                total_distance += dist

            for i, episode in enumerate(val_data[len(val_data)//2:]):
                val_loss, completed, dist = self.run_traj(model, episode, threshold = None)
                total_loss += val_loss.data


            thresh = 150
            print('Loss: ' + str(total_loss/len(val_data)))
            print('completed: ' + str(total_completed/len(val_data)))
            print('Average time before divergence: ' + str(total_distance/len(val_data)))

            if self.save:
                with open(self.model_save_path, 'wb') as pickle_file:
                    torch.save(model, pickle_file)

            # diagnostics(model, train_type='Full Trajectory', epoch=epoch, loss=(total_loss/(len(val_data)//2)), divergence=(total_distance/len(val_data)), grad_norms=grad_norms)



