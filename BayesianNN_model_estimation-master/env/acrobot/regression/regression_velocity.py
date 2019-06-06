import numpy as np
import pickle
from common.BNN import BNN

with open('../../../data/acrobot/acrobot_data_v2_d4', 'rb') as pickle_file:
    DATA = pickle.load(pickle_file)
DATA = np.asarray(DATA)[:]

x_data = DATA[:, :5]
y_data = DATA[:, 7:9] - DATA[:, 2:4]

if __name__ == "__main__":
    neural_network = BNN(nn_type='1')
    neural_network.add_dataset(x_data, y_data, held_out_percentage=0.1)
    neural_network.build_neural_net()
    save_path = '../../../save_model/acrobot/vel'
    neural_network.train(save_path=save_path, normalization=True, normalization_type='z_score')





