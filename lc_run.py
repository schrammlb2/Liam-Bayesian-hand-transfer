import os

methods = ['retrain', 'constrained_restart', 'constrained_restrain' 'linear_transform', 'nonlinear_transform']


# held_out_list = [.99,.98,.97,.96,.95,.94,.93,.92,.91,.9,.8,.7,.6,.5,.4,.3,.2,.1]
held_out_list = [.99,.98,.97,.96,.95]
real_base_command = 'python3 learning_curve.py _ '  
transfer_base_command = 'python3 regression_clean.py transferB2A '

for heldout in held_out_list:
	for method in methods:
		command = transfer_base_command + str(heldout) + ' _ ' + method
		print('Executing: \'' + command + '\'')
		os.system(command)