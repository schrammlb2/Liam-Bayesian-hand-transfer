import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pickle
import math
from common.data_normalization import z_score_normalize, z_score_denormalize
from common.BNN import BNN

tf.keras.backend.set_floatx('float64')  # for input weights of NN
ang_model_path = '../../../save_model/acrobot/ang'  # use to import weights and normalization arrays
vel_model_path = '../../../save_model/acrobot/vel'
test_traj = 1

PRE_START = 0  # prediction start point
PRE_END = None  # prediction end point, None means predict whole trajectory
VAR_ANG = [0.00001, 0.00001]
VAR_VEL = [0.00001, 0.00001]


with open('../../../data/acrobot/test_trajectory/acrobot_ao_rrt_traj' + str(test_traj), 'rb') as pickle_file:
    gt = pickle.load(pickle_file)

with open('../../../data/acrobot/test_trajectory/acrobot_ao_rrt_plan' + str(test_traj), 'rb') as pickle_file:
    acts = pickle.load(pickle_file)


def wrap_acrobot_action(plan_act):
    act_seq = []
    for act_pair in plan_act:
        a = act_pair[0]
        apply_num = int(act_pair[1] * 100)
        for i in range(apply_num):
            act_seq.append(a)
    return act_seq


def map_angle(state):
    if state[0] > math.pi:
        state[0] = - math.pi + (state[0] - math.pi)
    if state[0] < -math.pi:
        state[0] = state[0] + 2 * math.pi
    if state[1] > math.pi:
        state[1] = - math.pi + (state[1] - math.pi)
    if state[1] < -math.pi:
        state[1] = state[1] + 2 * math.pi
    return state


with open(ang_model_path+'/normalization_arr/normalization_arr', 'rb') as pickle_file:
    x_nor_arr, y_nor_arr_ang = pickle.load(pickle_file)

with open(vel_model_path+'/normalization_arr/normalization_arr', 'rb') as pickle_file:
    y_nor_arr_vel = pickle.load(pickle_file)[1]


acts = wrap_acrobot_action(acts)[PRE_START:PRE_END]
validation_data = np.asarray(gt[:-1])
validation_data = np.append(validation_data, np.asarray(acts).reshape((len(validation_data), 1)), axis=1)


'''Neural net structure'''
neural_net_ang = tf.keras.Sequential([
    tfp.layers.DenseFlipout(200, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.05),
    tfp.layers.DenseFlipout(200, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.05),
    tfp.layers.DenseFlipout(2),
])

neural_net_vel = tf.keras.Sequential([
    tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.1),
    tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.1),
    tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.1),
    tfp.layers.DenseFlipout(2),
])

x = tf.placeholder(tf.float64, shape=[None, 5])
y_ang_mean = neural_net_ang(x)
y_vel_mean = neural_net_vel(x)
y_ang_distribution = tfp.distributions.Normal(loc=y_ang_mean, scale=VAR_ANG)
y_vel_distribution = tfp.distributions.Normal(loc=y_vel_mean, scale=VAR_VEL)
y_ang_delta_pre = y_ang_distribution.sample()
y_vel_delta_pre = y_vel_distribution.sample()


if __name__ == "__main__":
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        neural_net_ang.load_weights(ang_model_path+"/weights/BNN_weights")  # load NN parameters
        neural_net_vel.load_weights(vel_model_path+"/weights/BNN_weights")
        angs = []  # prediction in angle space
        vels = []  # prediction in velocity space
        angs.append(validation_data[0][:2])
        vels.append(validation_data[0][2:4])
        state = np.array(validation_data[0])
        nor_state = z_score_normalize(np.asarray([state]), x_nor_arr[0], x_nor_arr[1])
        for i in range(len(validation_data)-1):
            (ang_delta, vel_delta) = sess.run((y_ang_delta_pre, y_vel_delta_pre), feed_dict={x: nor_state})
            ang_delta = z_score_denormalize(ang_delta, y_nor_arr_ang[0], y_nor_arr_ang[1])[0]  # denormalize
            vel_delta = z_score_denormalize(vel_delta, y_nor_arr_vel[0], y_nor_arr_vel[1])[0]
            next_ang = state[:2] + ang_delta
            next_vel = state[2:4] + vel_delta
            angs.append(next_ang)
            vels.append(next_vel)
            state = np.append(np.append(next_ang, next_vel), validation_data[i + 1][4:5])
            state = map_angle(state)
            nor_state = z_score_normalize(np.asarray([state]), x_nor_arr[0], x_nor_arr[1])

angs = np.asarray(angs)
vels = np.asarray(vels)

plt.figure(1)
plt.scatter(validation_data[0, 0], validation_data[0, 1], marker="*", label='start')
plt.plot(validation_data[:, 0], validation_data[:, 1], color='blue', label='Ground Truth', marker='.')
plt.plot(angs[:, 0], angs[:, 1], color='red', label='NN Prediction')
plt.axis('scaled')
plt.title('Bayesian NN Prediction -- Angle Space')
plt.legend()
plt.show()

plt.figure(2)
plt.scatter(validation_data[0, 2], validation_data[0, 3], marker="*", label='start')
plt.plot(validation_data[:, 2], validation_data[:, 3], color='blue', label='Ground Truth', marker='.')
plt.plot(vels[:, 0], vels[:, 1], color='red', label='NN Prediction')
plt.axis('scaled')
plt.title('Bayesian NN Prediction -- velocity Space')
plt.legend()
plt.show()
