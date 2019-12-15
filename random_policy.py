import gym 
import numpy as np
import pickle
import pdb
import matplotlib.pyplot as plt
import torch 

# base = '/home/william/Desktop/transfer/Research/Boularias/Liam-Bayesian-hand-transfer/data/'
base = 'data/'

def write_file_names(env_name):
	baseA = base + env_name + '/' + env_name + '_A'
	baseB = base + env_name + '/' + env_name + '_B'
	# baseB = base + env_name + '_B/'
	dct = {
	'train_A' : (baseA + '_train'),
	'train_B' : (baseB + '_train'),
	'test_A' : (baseA + '_test'),
	'test_B' : (baseB + '_test'),
	}
	return dct

acrobot_dict = {
	'task' : 'acrobot', 
	'state_dim': 4, 
	'action_dim': 1
}
cartpole_dict = {
	'task' : 'cartpole', 
	'state_dim': 4, 
	'action_dim': 1
}

bipedal_dict = {
	'task' : 'bipedal', 
	'state_dim': 24, 
	'action_dim': 4
}

hand_dict = {
	'task' : 'hand', 
	'state_dim': 4, 
	'action_dim': 6
}

hand_dict.update(write_file_names('hand'))

acrobot_dict.update(write_file_names('acrobot'))
cartpole_dict.update(write_file_names('cartpole'))


train_duration = 600
test_duration = 400

def build_env(env_name, version): 
	if env_name == 'cartpole':
		env = gym.make('CartPole-v0')
		env = gym.envs.classic_control.cartpole.CartpoleEnv()
		# env.tau = 
		if version == 'B':
			env.length *= .8
			env.polemass_length *= .8

		return env

	elif env_name == 'acrobot':
		env = gym.make('Acrobot-v1')
		env = gym.envs.classic_control.acrobot.AcrobotEnv()
		env.dt = .02
		# env.torque_noise_max = .01
		# env.torque_noise_max = 0
		if version == 'B':
			# env.LINK_LENGTH_1 = 1.5
			env.LINK_LENGTH_1 = 1.05
			env.LINK_COM_POS_1 = env.LINK_LENGTH_1/2
			# env.LINK_LENGTH_2 = .75
			env.LINK_LENGTH_2 = .95
			env.LINK_COM_POS_2 = env.LINK_LENGTH_2/2

		return env
			
	else:
		return

def observe(env):
	state_dim = task_dict['state_dim']
	rand = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(state_dim), torch.eye(state_dim))
	scale = .0001
	# scale = 0
	err = rand.sample()*scale
	return env.state+err.numpy()

def run_env(env_name, version, episodes=800, duration = train_duration, action_dur = 10):
	outstring = env_name + version + '\neps: ' + str(episodes) + '\nduration: ' + str(duration)
	print('running : ' + outstring)
	env = build_env(env_name, version)

	eps = []
	for i in range(episodes):
		env.reset()
		ep = []
		# old_state = env.state
		old_state = observe(env)
		action = env.action_space.sample()
		for j in range(duration):
			# env.render()
			if j%action_dur == 0:
				action = env.action_space.sample()

			env.step(action)
			rec_action = np.array([action])
			state = observe(env)
			output = np.concatenate([old_state, rec_action, state])
			ep.append(output)

			old_state = state

		ep = np.stack(ep, axis=0)
		eps.append(ep)


	return eps



env_name = 'acrobot'

task_dict = acrobot_dict

# env_name = 'hand'
# task_dict = hand_dict

with open(base + env_name + '_task', 'wb') as pickle_file:
	pickle.dump(task_dict, pickle_file)
# with open('data/cartpole_task') as pickle_file:
# 	pickle.dump(cartpole_dict, pickle_file)


task_A_train = run_env(env_name , 'A')
with open(task_dict['train_A'], 'wb') as pickle_file:
	pickle.dump(task_A_train, pickle_file)

task_A_test = run_env(env_name , 'A', episodes = 20, duration=test_duration)
with open(task_dict['test_A'], 'wb') as pickle_file:
	pickle.dump(task_A_test, pickle_file)


task_B_train = run_env(env_name , 'B')
with open(task_dict['train_B'], 'wb') as pickle_file:
	pickle.dump(task_B_train, pickle_file)

task_B_test = run_env(env_name , 'B', episodes = 20, duration=test_duration)
with open(task_dict['test_B'], 'wb') as pickle_file:
	pickle.dump(task_B_test, pickle_file)			
