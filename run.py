import os


# methods = ['retrain', 'constrained_restart', 'constrained_retrain', 'linear_transform', 'nonlinear_transform']
methods = ['retrain']#, 'traj_transfer']
# methods = ['traj_transfer']
# methods = ['traj_transfer_timeless']
methods = ['traj_transfer_timeless_recurrent', 'traj_transfer_timeless', 'retrain_naive']
# methods = ['linear_transform']
suffixes = 4


held_out_list = [.99,.98,.97,.96,.95,.94,.93,.92,.91,.9,.8,.7,.6,.5,.4,.3,.2,.1]
held_out_list = [.99,.98,.97]
# held_out_list = [.998,.997,.996,.995,.994,.992,.992,.991]#,.99]
# held_out_list = [.99, .98,.97,.96,.95,.94,.93,.92,.91,.9]#,.8,.7,.6,.5,.4,.3,.2,.1]
# held_out_list = [.998,.997,.996,.995,.994,.992,.992,.991,.99,.98,.97,.96,.95,.94,.93,.92,.91,.9,.8,.7,.6,.5,.4,.3,.2,.1]
# real_base_command = 'python3 regression_gym.py acrobot_A '  
# transfer_base_command = 'python3 regression_gym.py acrobot_transferB2A '
real_base_command = 'python3 regression_gym.py acrobot_B '  
transfer_base_command = 'python3 regression_gym.py acrobot_transferA2B '

train_base = True
train_transfer = True

for suffix in range(suffixes):
	for heldout in held_out_list:
		command = real_base_command + str(heldout) + ' _ ' + str(suffix)
		print('Executing: \'' + command + '\'')

		if train_base:
			os.system(command)

		for method in methods:
			command = transfer_base_command + str(heldout) + ' ' + method  + ' ' +  str(suffix)
			print('Executing: \'' + command + '\'')

			if train_transfer:
				os.system(command)
