import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
import tensorflow as tf
from collections import deque
from keras import losses
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, Adamax
from keras.initializers import Zeros, Ones
from environment_train import RetailEnvironment
import pandas as pd

class DQN_Agent:
    def __init__(self, DQN_config):
        self.DQN_config = DQN_config
        self.state_size = DQN_config['state_size']
        self.action_size = DQN_config['action_size']
        self.gamma = DQN_config['gamma']
        self.epsilon_decay = DQN_config['epsilon_decay']
        self.epsilon_min = DQN_config['epsilon_min']
        self.learning_rate = DQN_config['learning_rate']
        self.epochs = DQN_config['epochs']
        self.env = DQN_config['env']
        self.batch_size = DQN_config['batch_size']
        self.update = DQN_config['update']
        self.epoch_counter = 0
        self.epsilon = 1.0
        self.iteration = DQN_config['iteration']
        self.x = DQN_config['x']
        self.model = self.build_model()
        self.target_model = self.build_model()
        
        self.memory = deque(maxlen=20000)
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(32, activation = 'relu'))
        model.add(Dense(self.action_size, activation = 'linear'))
        
        model.compile(loss = losses.mean_squared_error, optimizer = Adam(lr = self.learning_rate))
        return model
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        print(state)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])