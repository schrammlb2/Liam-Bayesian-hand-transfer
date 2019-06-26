import torch 
import numpy as np
from common.pt_build_model import *
from common.data_normalization import *
import pickle


class RecurrentNet:
	def __init__(self, nn_type, state_dim, action_dim, dropout_p, save_path = None, cuda=None, file_name = 'model.pkl'):
		input_dim = state_dim + action_dim
		output_dim = state_dim
		self.model = pt_build_model(nn_type, input_dim, output_dim, dropout_p)
		if cuda == None:
			self.cuda = torch.cuda.is_available()
		else: self.cuda = cuda

		self.dtype = torch.float

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.save_path = save_path
		self.file_name = file_name

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

		return (x_mean_arr,	x_std_arr, y_mean_arr, y_std_arr)

	def run_traj(self, model, traj, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = 50, return_states = False):
	    true_states = traj[:,:self.state_dim]
	    state = traj[0][:self.state_dim]
	    states = []
	    if self.cuda:
	        state = state.cuda()
	        true_states = true_states.cuda()

	    mse_fn = torch.nn.MSELoss()

	    def softmax(states):
	        mse_fn = torch.nn.MSELoss(reduction='none')
	        mse = mse_fn(states[:,:2], true_states[:states.shape[0],:2])
	        mse = torch.sum(mse, 1)  #Sum two position losses at each time step to get the Euclidean distance
	        return torch.logsumexp(mse, 0)


	    for i, point in enumerate(traj):
	        states.append(state)
	        action = point[self.state_dim:self.state_dim+self.action_dim]
	        if self.cuda: action = action.cuda()

	        inpt = torch.cat((state, action), 0)

	        inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)
	        state_delta = model(inpt)
	        denorm_state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)

	        state= denorm_state_delta + state
	        #May need random component here to prevent overfitting
	        # states = torch.cat((states,state.view(1, self.state_dim)), 0)
	        if threshold and i%10:
	            with torch.no_grad():
	                mse = mse_fn(state[:2], true_states[i,:2])
	            if mse > threshold:
	                states = torch.stack(states, 0)
	                # return mse, 0, i
	                # pdb.set_trace()
	                # return softmax(states)*10, 0, i
	                return mse_fn(state[:2], true_states[i,:2]), 0, i


	    states = torch.stack(states, 0)
	    if return_states:
	        return states

	    loss_type = 'soft maximum'
	    # loss_type = 'sum'
	    if loss_type == 'soft maximum':
	        return softmax(states), 1, len(traj)
	    else:
	        mse_fn = torch.nn.MSELoss()
	        pos_loss = mse_fn(states[:,:2], true_states[:,:2])
	        

	    return pos_loss, 1, len(traj)

	def train(self, episodes, DATA, held_out=.1):


		x_data = DATA[:, :self.state_dim + self.action_dim]
		y_data = DATA[:, -self.state_dim:] - DATA[:, :self.state_dim]
		(x_mean_arr,	x_std_arr, y_mean_arr, y_std_arr) = self.get_norms(x_data, y_data)

		thresh = 10
		print('beginning run')

		opt = torch.optim.Adam(self.model.parameters(), lr=.001)

		np.random.shuffle(episodes)
		val_data = episodes[int(len(episodes)*(1-held_out)):]
		episodes = episodes[:int(len(episodes)*(1-held_out))]
		for epoch in range(500):
		    print('Epoch: ' + str(epoch))
		    np.random.shuffle(episodes)
		    total_loss = 0
		    total_completed = 0
		    total_distance = 0

		    for i, episode in enumerate(episodes):
		        if i % 30 == 0:
		            print(i)
		        loss, completed, dist = self.run_traj(self.model, episode, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = thresh)
		        opt.zero_grad()
		        loss.backward()
		        opt.step()

		    for i, episode in enumerate(val_data):
		        loss, completed, dist = self.run_traj(self.model, episode, x_mean_arr, x_std_arr, y_mean_arr, y_std_arr, threshold = 50)
		        total_loss += loss.data
		        total_completed += completed
		        total_distance += dist

		    thresh = 150
		    print('Loss: ' + str(total_loss/len(val_data)))
		    print('completed: ' + str(total_completed/len(val_data)))
		    print('Average time before divergence: ' + str(total_distance/len(val_data)))
		    if self.save_path:
			    with open(self.save_path+'/'+ self.file_name, 'wb') as pickle_file:
			        torch.save(self.model, pickle_file)
