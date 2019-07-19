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
# from common.utils import *
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
assert task in ['real_A', 'real_B', 'transferA2B', 'transferB2A', 'sim_A', 'sim_B']



base_state_dim = 4
state_dim = 8
action_dim = 2 if (task == 'sim_A' or task == 'sim_B') else 6
# action_dim = 6
alpha = .4
lr = .0002
new_lr = lr/2
# lr
# lr = .01
dropout_rate = .1


ave_coeff = .9 

dtype = torch.float
cuda = torch.cuda.is_available()
print('cuda is_available: '+ str(cuda))
cuda = False

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

    model = pt_build_model(nn_type, state_dim+action_dim, base_state_dim, dropout_rate)
    if cuda: 
        model = model.cuda()

    model_save_path = save_path+'/'+ task + '_heldout' + str(held_out)+ '_' + nn_type + '.pkl'



    l2_coeff = .000

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

class Trainer():
    def __init__(self, task, norm, method=None, save=True, model_save_path=None, state_dim = 4, action_dim = 6):
        self.state_dim = base_state_dim
        self.action_dim = action_dim
        self.task = task
        self.norm = norm
        self.method = method
        self.save=save
        self.model_save_path = model_save_path

    def pretrain(self, model, x_data, y_data, opt, train_load = True, epochs = 30, batch_size = 64):
        dataset = torch.utils.data.TensorDataset(x_data, y_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
        # loss_fn = torch.nn.NLLLoss()
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

                # pdb.set_trace()
                loss = loss_fn(output, sample[1]) 
                
                total_loss += loss.data



                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                opt.step()
            # diagnostics(model, train_type='pretrain', epoch=i)
            print("total_loss: " + str(total_loss))
            if i> 0:
                pdb.set_trace()
        if self.save:
            with open(self.model_save_path, 'wb') as pickle_file:
                torch.save(model, pickle_file)




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


        vel = torch.zeros((4))        
        state_delta = torch.zeros((4))


        for i, point in enumerate(traj):
            states.append(state)
            action = point[self.state_dim:self.state_dim+self.action_dim]
            if cuda: action = action.cuda()   

            next_vel = vel*ave_coeff + state_delta*(1-ave_coeff)
            # velocities.append(next_vel)
            vel = next_vel      

            inpt = torch.cat((state, vel, action), 0)
            inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

            state_delta_norm = model(inpt)
            state_delta = z_score_denorm_single(state_delta_norm, y_mean_arr, y_std_arr)
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
        if return_states:
            return states


        return softmax(states), 1, len(traj)


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
            mse_fn = torch.nn.MSELoss(reduction='none')
            msepos = mse_fn(states[:,:,:2], true_states[:,:states.shape[1],:2])
            mse = torch.mean(msepos, 2) #Sum two position losses at each time step to get the Euclidean distance 
            loss = torch.logsumexp(mse, 1) #Softmax divergence over the path
            loss = torch.mean(loss) #Sum over batch
            return loss

        def get_loss(loss_type, states = None, sim_deltas = None):
            mse_fn = torch.nn.MSELoss(reduction='none')
            scaling = 1/((torch.arange(states.shape[1], dtype=torch.float)+1)*states.shape[1])
            if cuda: scaling = scaling.cuda()
            loss_temp = mse_fn(states[:,:,:2], true_states[:,:states.shape[1],:2])
            loss = torch.einsum('ikj,k->', [loss_temp, scaling])
            return loss


        vel = batch[:,0,self.state_dim: self.state_dim*2]       
        state_delta = batch[:,0,self.state_dim: self.state_dim*2]  


        for i in range(batch.shape[1]):
            states.append(state)
            action = batch[:,i,self.state_dim:self.state_dim+self.action_dim]
            if cuda: action = action.cuda() 

            next_vel = vel*ave_coeff + state_delta*(1-ave_coeff)
            # velocities.append(next_vel)
            vel = next_vel      

            # pdb.set_trace()
            inpt = torch.cat((state, vel, action), 1)
            inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

            state_delta = model(inpt)

            state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
            if self.task in ['transferA2B', 'transferB2A']: state_delta *= torch.tensor([-1,-1,1,1], dtype=torch.float)
            sim_deltas.append(state_delta)
      

            state= state_delta + state

            #May need random component here to prevent overfitting
            # states = torch.cat((states,state.view(1, self.state_dim)), 0)
            if threshold:
                with torch.no_grad():
                    # pdb.set_trace()
                    mse = mse_fn(states[i][:,:2], true_states[:,i,:2])
                if mse > threshold:
                    states = torch.stack(states, 1)
                    sim_deltas = torch.stack(sim_deltas, 0)
                    # loss = mix(sim_deltas, states, alpha)
                    loss = softmax(states)
                    # loss = get_loss(loss_type, states=states, sim_deltas=sim_deltas)
                    # pdb.set_trace()
                    return loss, 0, i
                    # return mse_fn(state[:2], true_states[i,:2]), 0, i

        sim_deltas = torch.stack(sim_deltas, 1)
        states = torch.stack(states, 1)
        if return_states:
            return states

        return get_loss(loss_type, states=states, sim_deltas=sim_deltas), 1, batch.shape[1]


    #------------------------------------------------------------------------------------------------------------------------------------

    def batch_train(self, model, opt, out, val_data = None, epochs = 500, batch_size = 8, loss_type = 'pointwise'):
        j=0
        print('\nBatched trajectory training with batch size ' + str(batch_size))
        for epoch in range(epochs):
            grad_norms = []

            print('Epoch: ' + str(epoch))
            np.random.shuffle(out)
            total_loss = 0
            # pdb.set_trace()
            total_completed = 0
            total_distance = 0
            switch = True
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
                print('Loss: ' + str(total_loss/(len(val_data)*2)))
                print('completed: ' + str(total_completed/(len(val_data)*2)))
                print('Average time before divergence: ' + str(total_distance/(len(val_data)*2)))

            else:
                print('Loss: ' + str(total_loss/len(batches)))
                print('completed: ' + str(total_completed/len(batches)))
                print('Average time before divergence: ' + str(total_distance/len(batches)))

            if self.save:
                with open(self.model_save_path, 'wb') as pickle_file:
                    torch.save(model, pickle_file)




if task == 'real_B':
    out = clean_data(out)
    print("data cleaning worked")
    # print(len(out))

new_state_dim = 8

# data_type_offset = {'ave_load':4, 'load':2, 'pos':0}
# dt_ofs = data_type_offset[data_type]
task_ofs = new_state_dim + action_dim

length_threshold = 30
out = list(filter(lambda x: len(x) > length_threshold, out))




out_y = [ep[:, (-state_dim//2):] - ep[:, :(state_dim//2)] for ep in out]
all_velocities = []
for episode in out_y:
    vel = np.zeros((4))
    velocities = [vel]
    for step in episode:
        next_vel = vel*ave_coeff + step*(1-ave_coeff)
        velocities.append(next_vel)
        vel = next_vel
    velocities = np.stack(velocities, 0)
    all_velocities.append(velocities)


new_out = [np.concatenate((out_ep[:,:-base_state_dim], vel_ep[:-1], out_ep[:,-base_state_dim:]), 1) for (out_ep, vel_ep) in zip(out, all_velocities)]
out = new_out
# state_dim *= 2
# pdb.set_trace()


out = [torch.tensor(ep, dtype=dtype) for ep in out]
# pdb.set_trace()


val_size = int(len(out)*held_out)
# val_size = len(out) - int(held_out)
val_data = out[val_size:]
val_data = val_data[:min(10, len(val_data))]
out = out[:len(out)-val_size]


print("\nTraining with " + str(len(out)) + ' trajectories')


DATA = np.concatenate(out)



x_data = DATA[:, :task_ofs]
y_data = DATA[:, -base_state_dim:] - DATA[:, :base_state_dim]


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
# out = [z_score_normalize]

if cuda:
    x_mean_arr = x_mean_arr.cuda()
    x_std_arr = x_std_arr.cuda()
    y_mean_arr = y_mean_arr.cuda()
    y_std_arr = y_std_arr.cuda()
    # 




print('\n\n Beginning task: ')
# print('\t' + outfile_name)
# pdb.set_trace()

if __name__ == "__main__":
    thresh = 10
    norm = x_mean_arr, x_std_arr, y_mean_arr, y_std_arr
    # if method:
    #     traj_obj = util(task, norm, method)
    trainer = Trainer(task, norm, model_save_path=model_save_path, state_dim=state_dim, action_dim=action_dim) 
    # print('beginning run')

    np.random.shuffle(out)
    # val_data = out[int(len(out)*(1-held_out)):]
    if held_out > .95: 
        lr = .000065
        lr = .0001
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
        trainer.pretrain(model, x_data, y_data, opt, epochs=5)
        # if method == 'nonlinear_transform':
        #     model.set_base_model_train(True)
        opt = torch.optim.Adam(model.parameters(), lr=.000005, weight_decay=.001)
        trainer.batch_train(model, opt, out, val_data=val_data, epochs=25, batch_size=64)
    elif held_out > .9: 
        lr = .0001
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
        trainer.pretrain(model, x_data, y_data, opt, epochs=30)
        # if method == 'nonlinear_transform':
        #     model.set_base_model_train(True)
        opt = torch.optim.Adam(model.parameters(), lr=.000025, weight_decay=.001)
        trainer.batch_train(model, opt, out, val_data =val_data, epochs=25, batch_size=64)
    else:
        lr = .000025
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
        trainer.pretrain(model, x_data, y_data, opt, epochs=5, train_load=True, batch_size=256)
        # if method == 'nonlinear_transform':
        #     model.set_base_model_train(True)
        opt = torch.optim.Adam(model.parameters(), lr=.000005, weight_decay=.001)
        trainer.batch_train(model, opt, out, val_data =val_data, epochs=20, batch_size=64)
        trainer.batch_train(model, opt, out, val_data =val_data, epochs=20, batch_size=32, loss_type='softmax')
        trainer.batch_train(model, opt, out, val_data =val_data, epochs=30, batch_size=4, loss_type='softmax')
        trainer.batch_train(model, opt, out, val_data =val_data, epochs=30, batch_size=2, loss_type='softmax')
        # trainer.batch_train(model, opt, out, val_data =val_data, epochs=10, batch_size=2)
        # trainer.batch_train(model, opt, out, val_data =val_data, epochs=10, batch_size=1)
# diagnostics_file.close()
    
