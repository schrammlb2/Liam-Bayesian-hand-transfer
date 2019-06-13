import pdb
import pickle
import numpy as np
import copy 

filename='data/robotic_hand_real/B/testpaths_cyl35_red_d_v0_RAW.pkl'
new_filename = 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl'

with open(filename, 'rb') as pickle_file:
    trajectory = pickle.load(pickle_file, encoding='latin1')

new_traj = copy.deepcopy(trajectory)
test_paths = trajectory[1]
action_seq = trajectory[0]
# pdb.set_trace()
for path_inx in range(6):# Number of test path -> 0 to 5
	R = test_paths[path_inx]
	A = action_seq[path_inx]
	R = R[:,[0,1,11,12]]
	A = np.concatenate((A, np.tile(R[0,:], (A.shape[0], 1))), axis=1) # This is only if you want to include the start state in the action - try with and without
	new_traj[0][path_inx] = R
	new_traj[1][path_inx] = A[:-1]
	# pdb.set_trace()

with open(new_filename, 'wb') as new_file:
	pickle.dump(new_traj, new_file)#, encoding='latin1')