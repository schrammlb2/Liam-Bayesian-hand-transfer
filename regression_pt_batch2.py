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
task = 'real_A' #Which task we're training. This tells us what file to use
outfile = None
append = False
held_out = .1
# _ , arg1, arg2, arg3 = argv
epochs = 250
nn_type = '1'

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
# lr
# lr = .01
dropout_rate = .1

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

    model = pt_build_model(nn_type, state_dim+action_dim, state_dim, dropout_rate)
    if cuda: 
        model = model.cuda()

    model_save_path = save_path+'/'+ task + '_' + nn_type + '.pkl'



    l2_coeff = .000



elif task in ['transferA2B', 'transferB2A']:

    method = 'retrain'
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



    model_save_path = save_path+'/'+ task + '_' + method + '_' + nn_type + '.pkl'


    if method in ['constrained_retrain', 'constrained_restart']:
        l2_coeff = .001
        if len(argv) > 6:
            l2_coeff = float(argv[6])
            model_save_path = save_path+'/'+ task + '_' + method + '_' + str(l2_coeff) + '_' + nn_type + '.pkl'







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
        model = pt_build_model(nn_type, state_dim+action_dim, state_dim, dropout_rate)

    elif method == 'linear_transform':
        model = LinearTransformedModel(model, state_dim + action_dim, state_dim)

    elif method == 'nonlinear_transform':
        model = NonlinearTransformedModel(model, state_dim + action_dim, state_dim)

    elif method == 'single_transform':
        model = torch.nn.Sequential(*[model, torch.nn.Linear(state_dim , state_dim)])
        for param in model[0].parameters():
            param.requires_grad = False

    else: 
        print("Invalid method type")
        assert False




















opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.0001)

outfile_name = task + '_diagnostics'
if 'transfer' in task:  
    outfile_name = task + '_' + method + '_'
    if method == 'constrained_restart':
        outfile_name += 'l2.'+str(l2_coeff)

    outfile_name += '_held_out.'+str(held_out)
    outfile_name+='_diagnostics'

diagnostics_file = open(outfile_name, 'w+')
diagnostics_file.write('\n')
diagnostics_file.close()


def weight_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item()**2
    return total_norm**(.5)
    return 

def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
    return total_norm**(.5)

def diagnostics(model, train_type = None, epoch=None, loss=None, divergence=None, grad_norms= None):
    wn = weight_norm(model)
    if grad_norms:
        gn = sum(grad_norms)/len(grad_norms)
    else:
        gn = grad_norm(model)
    diag_str = ''
    if train_type:
        diag_str += train_type + ' epoch ' + str(epoch) + '\n'
    diag_str += 'Weight norm: ' + str(wn) + '\n'
    diag_str += 'Grad norm: ' + str(gn) + '\n'
    if loss: 
        diag_str += 'Loss: ' + str(loss) + '\n'
    if divergence: 
        diag_str += 'Divergence: ' + str(divergence) + '\n'
    diag_str += '--------------------------------------------\n'
    diagnostics_file = open(outfile_name, 'a+')
    diagnostics_file.write(diag_str)
    diagnostics_file.close()

#------------------------------------------------------------------------------------------------------------------------------------

def pretrain(model, x_data, y_data, opt, train_load = True, epochs = 30):
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
            if task in ['transferA2B', 'transferB2A']: 
                out *= torch.tensor([-1,-1,1,1], dtype=dtype)

            if train_load:
                loss = loss_fn(out, sample[1]) 
            else:
                loss = loss_fn(out[:,:2], sample[1][:,:2])

            loss += l2_coeff*offset_l2(model)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            opt.step()
        diagnostics(model, train_type='pretrain', epoch=i)

    with open(model_save_path, 'wb') as pickle_file:
        torch.save(model, pickle_file)
#------------------------------------------------------------------------------------------------------------------------------------

def run_traj(model, traj, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = 50, return_states = False, 
    loss_type = 'softmax', alpha = .5):
    # if type(traj) != type(torch.tensor(1)):
    #     pdb.set_trace() 

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

        # if type(state) != type(torch.tensor(1)) or type(action) != type(torch.tensor(1)):
        #     pdb.set_trace() 
        inpt = torch.cat((state, action), 0)
        inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

        # pdb.set_trace()
        state_delta = model(inpt)
        state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
        if task in ['transferA2B', 'transferB2A']: 
            state_delta *= torch.tensor([-1,-1,1,1], dtype=dtype)
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


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_traj_batch(model, batch, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = 50, return_states = False, 
    loss_type = 'softmax', alpha = .5):
    true_states = batch[:,:,:state_dim]
    state = batch[:,0,:state_dim]
    states = []#state.view(1, state_dim)
    sim_deltas = []
    if cuda:
        state = state.cuda()
        true_states = true_states.cuda()

    mse_fn = torch.nn.MSELoss()

    def softmax(states):
        # pdb.set_trace()
        mse_fn = torch.nn.MSELoss(reduction='none')
        # mse_fn = torch.nn.MSELoss()
        mse = mse_fn(states[:,:,:2], true_states[:,:states.shape[1],:2])
        mse = torch.mean(mse, 2) #Sum two position losses at each time step to get the Euclidean distance 
        # mse = torch.sum(mse, (1,2))     #Or sum to find the maximum divergence in the batch and emphasize that
        loss = torch.logsumexp(mse, 1) #Softmax divergence over the path
        loss = torch.mean(loss) #Sum over batch
        # loss = torch.logsumexp(loss, 0)
        return loss

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

    for i in range(batch.shape[1]):
        # point = batch[:,i]
        states.append(state)
        action = batch[:,i,state_dim:state_dim+action_dim]
        if cuda: action = action.cuda() 
        # pdb.set_trace()   
        inpt = torch.cat((state, action), 1)
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

def batch_train(model, opt, out, epochs = 500, batch_size = 8):
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
        loss_type = 'softmax'
        thresh = 150


        batch_lists = [out[i: min(len(out), i+ batch_size)] for i in range(0, len(out), batch_size)] 
        episode_lengths = [[len(ep) for ep in batch] for batch in batch_lists]
        min_lengths = [min(episode_length) for episode_length in episode_lengths]
        # min_lengths = [min([len(ep) for ep in batch]) for batch in batch_lists]
        rand_maxes = [[len(episode) - min_length for episode in batch_list] for batch_list, min_length in zip(batch_lists,min_lengths)]
        rand_starts = [[random.randint(0, rmax) for rmax in rmaxes] for rmaxes in rand_maxes]
        batch_slices = [[episode[start:start+length] for episode, start in zip(batch, starts)] for batch, starts, length in zip(batch_lists, rand_starts, min_lengths)]

        batches = [torch.stack(batch, 0) for batch in batch_slices] 

        accum = 8//batch_size

        for i, batch in enumerate(batches):
            if j % accum ==0: opt.zero_grad()

            j += 1
            # if i % 30 == 0:
                # print(i*batch_size)
            loss, completed, dist = run_traj_batch(model, batch, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = thresh, loss_type=loss_type)
            total_loss += loss.data
            total_completed += completed
            total_distance += dist
            # pdb.set_trace()

            loss.backward()
            if j % accum ==0: 
                if task == 'transferA2B':
                    if method in ['constrained_retrain', 'constrained_restart']:
                        loss = offset_l2(model)*l2_coeff*accum
                        loss.backward()

                grad_norms.append(grad_norm(model))

                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                opt.step()


        print('Loss: ' + str(total_loss/len(batches)))
        print('completed: ' + str(total_completed/len(batches)))
        print('Average time before divergence: ' + str(total_distance/len(batches)))
        with open(model_save_path, 'wb') as pickle_file:
            torch.save(model, pickle_file)
        diagnostics(model, train_type='Batch Trajectory', loss=(total_loss/len(batches)), epoch=epoch, grad_norms=grad_norms)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def traj_train(model, opt, out, val_data, epochs = 12):
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

            loss, completed, dist = run_traj(model, episode, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = thresh, loss_type=loss_type)
            loss.backward()

            if j % accum ==0: 
                if task == 'transferA2B':
                    if method in ['constrained_retrain', 'constrained_restart']:
                        loss = offset_l2(model)*l2_coeff*accum
                        loss.backward()

                grad_norms.append(grad_norm(model))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                opt.step()

        for i, episode in enumerate(val_data[:len(val_data)//2]):
            _ , completed, dist = run_traj(model, episode, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = 50)
            total_completed += completed
            total_distance += dist

        for i, episode in enumerate(val_data[len(val_data)//2:]):
            val_loss, completed, dist = run_traj(model, episode, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = None)
            total_loss += val_loss.data


        thresh = 150
        print('Loss: ' + str(total_loss/len(val_data)))
        print('completed: ' + str(total_completed/len(val_data)))
        print('Average time before divergence: ' + str(total_distance/len(val_data)))
        with open(model_save_path, 'wb') as pickle_file:
            torch.save(model, pickle_file)

        diagnostics(model, train_type='Full Trajectory', epoch=epoch, loss=(total_loss/(len(val_data)//2)), divergence=(total_distance/len(val_data)), grad_norms=grad_norms)



out = [torch.tensor(ep, dtype=dtype) for ep in out]

val_size = int(len(out)*held_out)
val_data = out[val_size:]
out = out[:len(out)-val_size]

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


print('\n\n Beginning task: ')
print('\t' + outfile_name)
# pdb.set_trace()
if __name__ == "__main__":
    thresh = 10
    # print('beginning run')

    np.random.shuffle(out)
    # val_data = out[int(len(out)*(1-held_out)):]
    TEST_MODE = False
    # TEST_MODE = True
    if 'transfer' in task and 'transform' in method:
        lr = .0003
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
        pretrain(model, x_data, y_data, opt, epochs=int(5))#/(1.1-held_out)))
        opt = torch.optim.Adam(model.parameters(), lr=.00003, weight_decay=.001)
        # opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.0003)
        batch_train(model, opt, out, epochs=1, batch_size=8)
        batch_train(model, opt, out, epochs=1, batch_size=4)
        batch_train(model, opt, out, epochs=1, batch_size=2)
        # batch_train(model, opt, out, epochs=10, batch_size=1)
        
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.0003)
        traj_train(model, opt, out, val_data)

    elif TEST_MODE:
        lr = .0003
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
        pretrain(model, x_data, y_data, opt, epochs=int(1))
        batch_train(model, opt, out, epochs=1, batch_size=8)
        traj_train(model, opt, out, val_data)


    else: 
        lr = .0003
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
        pretrain(model, x_data, y_data, opt, epochs=int(15/(1.1-held_out)))
        opt = torch.optim.Adam(model.parameters(), lr=.00003, weight_decay=.001)
        # # opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.0003)
        # batch_train(model, opt, out, epochs=1, batch_size=8)
        # batch_train(model, opt, out, epochs=1, batch_size=4)
        # batch_train(model, opt, out, epochs=2, batch_size=2)
        # batch_train(model, opt, out, epochs=10, batch_size=1)
        
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.0003)
        traj_train(model, opt, out, val_data, epochs=int(12/(1.1-held_out)))

# diagnostics_file.close()
    