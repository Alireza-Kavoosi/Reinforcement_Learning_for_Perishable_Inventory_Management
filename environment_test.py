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
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(config_dict)


config = {
    'life_time': 10,
    'lead_time': 5,
    'mean_demand': 10.0,
    'coefficient_of_variation': 0.5,
    'max_order': 20,
    'order_cost': 10.0,
    'outdated_cost': 5.0,
    'lost_sales_cost': 20.0,
    'holding_cost': 1.0,
    'use_FIFO': True,
    'use_LIFO': False,
    'simulation_time': 100,
    'warmup_period': 20
}

env = RetailEnvironment.from_dict(config)