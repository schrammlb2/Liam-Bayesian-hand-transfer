""" 
Author: Avishai Sintov
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from common.data_normalization import *
import pickle
import random

def transfer(x, state_dim): 
    x = torch.tensor(x, dtype=torch.float)
    return torch.cat((x[...,:state_dim], x[...,state_dim:state_dim+2]*-1,  x[...,state_dim+2:]), -1)

class predict_nn:
    def __init__(self):

        base_path = ''
        save_path = base_path + 'save_model/robotic_hand_real/pytorch/'
        # model_name = 'real_A_1.pkl' # Name of the model we want to depickle
        model_name = 'real_A_heldout0.1_1.pkl'
        model_name = 'transferB2A_traj_transfer_timeless_heldout0.9_1.pkl'
        self.model_path = save_path + model_name

        print('[predict_nn] Loading training data...')
        with open(save_path+'/normalization_arr/normalization_arr_py2', 'rb') as pickle_file:
            x_norm_arr, y_norm_arr = pickle.load(pickle_file)

        self.x_mean_arr, self.x_std_arr = x_norm_arr[0], x_norm_arr[1]
        self.y_mean_arr, self.y_std_arr = y_norm_arr[0], y_norm_arr[1]

        with open(self.model_path, 'rb') as pickle_file:
            self.model = torch.load(pickle_file, map_location='cpu')


    def normalize(self, data):
        return (data - self.x_mean_arr[:data.shape[-1]]) / self.x_std_arr[:data.shape[-1]]

    def denormalize(self, data, ):
        return data * self.y_std_arr[:data.shape[-1]] + self.y_mean_arr[:data.shape[-1]]


    def predict_trajectory(self, state, actions):

        # norm_state = self.normalize(state)
        state = torch.tensor(state, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)

        state_stack = torch.stack([state]*actions.shape[-2],-2) 
        inpt = torch.cat([state_stack, actions], -1)
        # inpt = state       
        # inpt = transfer(inpt, 4)

        states = self.model.run_traj(inpt, threshold=None)

        states = states.view_as(state_stack)

        return states





# if __name__ == "__main__":
#     task = 'real_A' 
#     held_out = .1
#     test_traj = 0
#     # _ , arg1, arg2, arg3 = argv
#     nn_type = '1'
#     method = ''
#     save_path = 'save_model/robotic_hand_real/pytorch/'

#     trajectory_path_map = {
#         'real_A': 'data/robotic_hand_real/A/testpaths_cyl35_d_v0.pkl', 
#         'real_B': 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl',
#         'transferA2B': 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl',
#         'transferB2A': 'data/robotic_hand_real/A/testpaths_cyl35_d_v0.pkl',
#         }
#     # trajectory_path = trajectory_path_map[task]
#     trajectory_path = 'data/robotic_hand_real/A/testpaths_py2.pkl'

#     with open(trajectory_path, 'rb') as pickle_file:
#         trajectory = pickle.load(pickle_file)#, encoding='latin1')

#         # u = pickle._Unpickler(pickle_file)
#         # u.encoding = 'latin1'
#         # trajectory = u.load()

#     def make_traj(trajectory, test_traj):
#         real_positions = trajectory[0][test_traj]
#         acts = trajectory[1][test_traj]    
#         return np.append(real_positions, acts, axis=1)

#     NN = predict_nn()
#     state_dim = 4
#     action_dim = 6



#     states = []
#     batch_size = 4
#     out=[make_traj(trajectory, i) for i in range(batch_size)]

#     lengths = [len(traj) for traj in out]
#     min_length = min(lengths)
#     batches = [traj[0:min_length] for traj in out]
#     traj = np.stack(batches,0)
#     # pdb.set_trace()


#     true_states = traj[:,:,:state_dim]

#     if task[-1] == 'A' and 'sim' not in task and 'acrobot' not in task:
#         transfered_traj = transfer(traj, state_dim)

#     state = transfered_traj[:,0,:state_dim]

#     actions = transfered_traj[..., state_dim:state_dim+action_dim]

#     # states = NN.predict_trajectory(state, actions)
#     NN.model.coeff = 1
#     states = NN.predict_trajectory(state, actions)

#     states = states.detach().numpy()

#     for test_traj in range(4):

#         plt.figure(1)
#         plt.scatter(traj[test_traj, 0, 0], traj[test_traj, 0, 1], marker="*", label='start')
#         plt.plot(traj[test_traj, :, 0], traj[test_traj, :, 1], color='blue', label='Ground Truth', marker='.')
#         plt.plot(states[test_traj, :, 0], states[test_traj, :, 1], color='red', label='NN Prediction')
#         plt.axis('scaled')
#         plt.title('Bayesian NN Prediction -- pos Space')
#         plt.legend()
#         plt.show()

