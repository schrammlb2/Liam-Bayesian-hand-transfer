import scipy.io
# from common.BNN import BNN
import sys
import pickle
import numpy as np 
import pdb
import torch
from common.data_normalization import z_score_normalize, z_score_denormalize
from common.pt_build_model import pt_build_model

from sys import argv

data_type = 'pos' #type of data used for this task
task = 'real_A' #Which task we're training. This tells us what file to use
skip_step = 1
outfile = None
append = False
held_out = .1
# _ , arg1, arg2, arg3 = argv

# pdb.set_trace()
if len(argv) > 1:
    data_type = argv[1]
if len(argv) > 2:
    task = argv[2]
if len(argv) > 3:
    skip_step = int(argv[3])
if len(argv) > 4:
    outfile = argv[4]
if len(argv) > 5:
    append = (argv[5] in ['True' ,'T', 'true', 't'])
if len(argv) > 6:
    held_out = float(argv[6])

assert data_type in ['pos', 'load']
assert task in ['real_A', 'real_B','sim_A', 'sim_B']
assert skip_step in [1,10]
if (len(argv) > 3 and task != 'sim_A'):
    print('Err: Skip step only appicable to sim_A task. Do not use this argument for other tasks')
    exit(1)

datafile_name = 'data/robotic_hand_real/A/t42_cyl35_data_discrete_v0_d4_m1_episodes.obj'
save_path = 'save_model/robotic_hand_real/episodic/' + data_type
with open(datafile_name, 'rb') as pickle_file:
    out = pickle.load(pickle_file, encoding='latin1')

state_dim = 4
action_dim = 6
alpha = .4

dtype = torch.float
cuda = torch.cuda.is_available()
print('cuda is_available: '+ str(cuda))

def run_traj(model, traj, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr):
    states = []
    true_states = traj[:,:state_dim]
    state = traj[0][:state_dim]

    if cuda:
        state = state.cuda()
        true_states = true_states.cuda()
    mse_fn = torch.nn.MSELoss(reduction='none')


    for point in traj:
        states.append(state)
        action = point[state_dim:state_dim+action_dim]
        if cuda: action = action.cuda()
        inpt = torch.cat((state, action), 0)

        # pdb.set_trace()
        norm_inpt = z_score_normalize(inpt, x_mean_arr, x_std_arr)

        state_delta = model(inpt)
        state+= z_score_denormalize(state_delta, y_mean_arr, y_std_arr)
        #May need random component here to prevent overfitting

    states = torch.stack(states, 0)
    mse = mse_fn(states[:,:2], true_states[:,:2])

    loss_type = 'soft maximum'
    if loss_type == 'soft maximum':
        mse = torch.sum(mse, 1)  #Sum two position losses at each time step to get the Euclidean distance
        pos_loss = torch.logsumexp(mse, 0)
    else:
        pos_loss = torch.sum(mse)

    return pos_loss


model = pt_build_model('0', state_dim+action_dim, state_dim, .1)
if cuda: 
    model = model.cuda()
opt = torch.optim.Adam(model.parameters())



new_eps = []
for episode in out:
    x_ave = episode[0, 2:4]*0
    y_ave = episode[0, state_dim+action_dim+2:]*0
    x_ave_list_episode = []
    y_ave_list_episode = []
    for i in range(len(episode)):
        x_ave *= alpha
        y_ave *= alpha

        x_ave += episode[i, 2:4]*(1-alpha) 
        y_ave += episode[i, state_dim+action_dim+2:]*(1-alpha)
        x_ave_list_episode.append(x_ave)
        y_ave_list_episode.append(y_ave)
    x_stack = np.stack(x_ave_list_episode, axis=0)  
    y_stack = np.stack(y_ave_list_episode, axis=0)

    new_ep = np.concatenate([episode[:,:2], x_stack, episode[:,4:], y_stack], axis=1)
    new_eps.append(new_ep)
DATA = np.concatenate(new_eps)

new_state_dim = 4

data_type_offset = {'ave_load':4, 'load':2, 'pos':0}
dt_ofs = data_type_offset[data_type]
task_ofs = new_state_dim + action_dim

x_data = DATA[:, :task_ofs]
y_data = DATA[:, task_ofs+dt_ofs:task_ofs+dt_ofs+2] - DATA[:, dt_ofs:dt_ofs+2]


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

out = [torch.tensor(ep, dtype=dtype) for ep in out]
if __name__ == "__main__":

    print('beginning run')
    for epoch in range(500):
        np.random.shuffle(out)
        total_loss = 0
        # pdb.set_trace()

        for i, episode in enumerate(out):
            if i % 30 == 0:
                print(i)
            loss = run_traj(model, episode, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr)
            total_loss += loss.data

            opt.zero_grad()
            loss.backward()
            opt.step()
        print(total_loss/len(out))

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
    