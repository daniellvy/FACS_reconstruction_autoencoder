import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from tqdm.autonotebook import tqdm
from tqdm import tqdm
%matplotlib inline
from IPython import display
import pandas as pd

from google3.pyglib import gfile

tf.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Reshape,
    RepeatVector,
    TimeDistributed,
    Activation,
    LSTM
)



timesteps = 50
BATCH_SIZE=32


class AE(tf.keras.Model):
    """a basic autoencoder class for tensorflow
    Extends:
        tf.keras.Model
    """
    def __init__(self, **kwargs):
        super(AE, self).__init__()
        self.__dict__.update(kwargs)
         
        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

    @tf.function
    def encode(self, x):
        return self.enc(x)

    @tf.function
    def decode(self, z):
        return self.dec(z)
    
    @tf.function
    def compute_loss(self, x):
        z = self.encode(x)
        _x = self.decode(z)
        ae_loss = tf.reduce_mean(tf.square(x - _x))
        return ae_loss
    
    @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)

    def call(self, inputs):
      return self.decode(self.encode(inputs))

    @tf.function
    def train(self, train_x):    
        gradients = self.compute_gradients(train_x)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))



# from tensorflow.keras.layers import UnifiedLSTM as LSTM

N_Z = 32
AE_UNITS = 128
enc = [
    Reshape(target_shape=(timesteps, columns)),
    # Bidirectional(LSTM(units=AE_UNITS, activation="relu")),
    LSTM(units=AE_UNITS, activation="relu"),
    Dense(units=AE_UNITS),
    Activation('tanh'),
    Dense(units=N_Z),
    Activation('tanh'),
]

dec = [
    Dense(units=AE_UNITS),
    Activation('tanh'),
    RepeatVector(timesteps),
    LSTM(units=AE_UNITS, activation="tanh", return_sequences=True),
    Dense(units=columns),
    Activation('tanh')
]

columns_list = [' AU01_r',
 ' AU02_r',
 ' AU04_r',
 ' AU05_r',
 ' AU06_r',
 ' AU07_r',
 ' AU09_r',
 ' AU10_r',
 ' AU12_r',
 ' AU14_r',
 ' AU15_r',
 ' AU17_r',
 ' AU20_r',
 ' AU23_r',
 ' AU25_r',
 ' AU26_r',
 ' AU45_r',
]

columns = len(column_list)
from keras.models import load_model
def load_model(weights_path):
	# optimizers
	learning_rate = 1e-4
	optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.9999, epsilon=1e-8)
	# model
	model = AE(
    		enc = enc,
    		dec = dec,
    		optimizer = optimizer,
	)
	model.build(input_shape=(BATCH_SIZE, timesteps, columns))
	model.load_weights(weights_path)
	return model
