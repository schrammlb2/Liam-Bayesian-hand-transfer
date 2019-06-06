import scipy.io
from common.BNN import BNN

DATA = scipy.io.loadmat('../../../data/robotic_hand_simulator/sim_data_discrete_v13_d4_m10.mat')['D']
x_data = DATA[:, :6]
y_data = DATA[:, 6:8] - DATA[:, :2]


if __name__ == "__main__":
    neural_network = BNN(nn_type='0')
    neural_network.add_dataset(x_data, y_data, held_out_percentage=0.1)
    neural_network.build_neural_net()
    save_path = '../../../save_model/robotic_hand_simulator/d4_s10_pos'
    neural_network.train(save_path=save_path, normalization=True, normalization_type='z_score')