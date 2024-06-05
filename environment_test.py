import numpy as np
import random
import math
import copy

class RetailEnvironment:
    def __init__(self, config):
        self.config = config
        self.Life_time = config['life_time']
        self.Lead_time = config['lead_time']
        self.mean_demand = config['mean_demand']
        self.coefficient_of_variation = config['coefficient_of_variation']
        self.max_order = config['max_order']
        self.order_cost = config['order_cost']
        self.outdated_cost = config['outdated_cost']
        self.lost_sales_cost = config['lost_sales_cost']
        self.holding_cost = config['holding_cost']
        self.use_FIFO = config['use_FIFO']
        self.use_LIFO = config['use_LIFO']
        self.simulation_time = config['simulation_time']
        self.warmup_period = config['warmup_period']
        
        self.demand = 0
        self.action = 0
        self.current_time = 0
        self.reward = 0
        
        self.shape = 1 / (self.coefficient_of_variation ** 2)
        self.scale = self.mean_demand / self.shape
        
        self.state = [0] * (self.Life_time + self.Lead_time - 1)
        self.render_state = self.state.copy()
        
        self.action_space = list(range(self.max_order + 1))
        
        np.random.seed(17)
        print('environment created')
    def step(self, action):
        self.action = action
        self.demand = round(np.random.gamma(self.shape, self.scale))
        demand = self.demand
        
        next_state = self.state[1:] + [action]
        self.render_state = next_state.copy()
        
        print(next_state)
        
        calc_state = next_state.copy()
        if self.use_FIFO:
            for i in range(self.Life_time):
                demand, next_state[-i-1] = self.update_demand(demand, calc_state[-i-1])
        if self.use_LIFO:
            for i in range(self.Lead_time, self.Lead_time + self.Life_time):
                demand, next_state[i] = self.update_demand(demand, calc_state[i])
        
        calc_state = next_state.copy()
        next_state = [0] + calc_state[:-1]
        
    def update_demand(self, demand, state):
        demand_used = min(demand, state)
        demand -= demand_used
        state -= demand_used
        return demand, max(state, 0)
    
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(config_dict)
