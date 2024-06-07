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

class DQNAgent:
    def __init__(self, DQN_config):
        self.DQN_config = DQN_config
        self.state_size = DQN_config['state_size']
        self.action_size = DQN_config['action_size']
        self.DiscountFactor = DQN_config['DiscountFactor']
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
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.memory = deque(maxlen=20000)
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self.learning_rate))
        return model
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print('***** Target network updated *****')
        
    def replay(self):
        MiniBatch = random.sample(self.memory, self.batch_size)
        
        current_states = np.array([experience[0] for experience in MiniBatch])
        current_qs = self.model.predict(current_states)
        
        new_states = np.array([experience[3] for experience in MiniBatch])
        future_qs = self.target_model.predict(new_states)
        
        x = []
        y = []
        
        for experience, current_q, future_q in zip(MiniBatch, current_qs, future_qs):
            _, action, reward, _, done = experience
            if not done:
                max_fut_q = np.max(future_q)
                new_q = reward + self.DiscountFactor * max_fut_q
            else:
                new_q = reward
            
            current_q[0][action] = new_q
            x.append(experience[0][0])
            y.append(current_q[0])
        
        self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size, verbose=0, shuffle=False)
        
        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # self.epsilon = max(self.epsilon_min, self.epsilon)
        
        # update target network
        if self.epoch_counter % self.update == 0:
            self.update_target_model()
    
    def train(self):
        # TO DO: implement the training loop
        pass
    
    def save_model(self, filename):
        self.model.save(filename)
    
    def load_model(self, filename):
        self.model = load_model(filename)
        self.target_model = load_model(filename)