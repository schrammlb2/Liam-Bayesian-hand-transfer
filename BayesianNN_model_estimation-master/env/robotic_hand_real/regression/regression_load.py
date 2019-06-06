import numpy as np
import pickle
from common.BNN import BNN


with open('../../../data/robotic_hand_real/t42_cyl45_data_discrete_v0_d12_m1.obj', 'rb') as pickle_file:
    data_matrix, state_dim, action_dim, _, _ = pickle.load(pickle_file, encoding='latin1')
DATA = np.asarray(data_matrix)

x_data = DATA[:, :14]
y_data = DATA[:, 16:18] - DATA[:, 2:4]

if __name__ == "__main__":
    neural_network = BNN(nn_type='1')
    neural_network.add_dataset(x_data, y_data, held_out_percentage=0.1)
    neural_network.build_neural_net()
    save_path = '../../../save_model/robotic_hand_real/load'
    neural_network.train(save_path=save_path, normalization=True, normalization_type='z_score', decay='True')