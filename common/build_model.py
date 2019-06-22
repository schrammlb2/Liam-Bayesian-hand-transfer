import tensorflow as tf
import tensorflow_probability as tfp
from common.data_normalization import min_max_normalize, z_score_normalize
import numpy as np
import pickle

def build_model(nn_type, output_dim, dropout_p):
	if nn_type == '0':
	    neural_net = tf.keras.Sequential([
	        tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
	        tf.keras.layers.Dropout(rate=dropout_p),
	        tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
	        tf.keras.layers.Dropout(rate=dropout_p),
	        tfp.layers.DenseFlipout(output_dim),
	    ])
	elif nn_type == '1':
	    neural_net = tf.keras.Sequential([
	        tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
	        tf.keras.layers.Dropout(rate=dropout_p),
	        tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
	        tf.keras.layers.Dropout(rate=dropout_p),
	        tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
	        tf.keras.layers.Dropout(rate=dropout_p),
	        tfp.layers.DenseFlipout(output_dim),
	    ])

	elif nn_type == '2':
	    neural_net = tf.keras.Sequential([
	        tf.keras.layers.Dense(64, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        tf.keras.layers.Dense(64, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        tf.keras.layers.Dense(32, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        # tfp.layers.DenseFlipout(32, activation=tf=nn.selu),

	        tf.keras.layers.Dense(32, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        tf.keras.layers.Dense(2),
	    ])
	elif nn_type == '3':
	    neural_net = tf.keras.Sequential([
	        tf.keras.layers.Dense(64, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        tf.keras.layers.Dense(64, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        tf.keras.layers.Dense(16, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        # tfp.layers.DenseFlipout(32, activation=tf=nn.selu),
	        tf.keras.layers.Dense(2),
	    ])
	elif nn_type == '4':
	    neural_net = tf.keras.Sequential([
	        tf.keras.layers.Dense(128, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        tf.keras.layers.Dense(64, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        tf.keras.layers.Dense(32, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        tf.keras.layers.Dense(32, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        tf.keras.layers.Dense(16, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        tf.keras.layers.Dense(16, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        tf.keras.layers.Dense(16, activation=tf.nn.selu),
	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        tf.keras.layers.Dense(16, activation=tf.nn.selu),
	        # tf.keras.layers.AlphaDropout(rate=dropout_p),
	        # tfp.layers.DenseFlipout(output_dim),

	        tf.keras.layers.AlphaDropout(rate=dropout_p),
	        tf.keras.layers.Dense(2),
	    ])
	return neural_net