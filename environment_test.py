import numpy as np
import random
import math
import copy

class RetailEnvironment:
    def __init__(self, env_config):
        self.env_config = env_config
        self.Life_time = env_config['life_time']
        self.Lead_time = env_config['lead_time']
        self.mean_demand = env_config['mean_demand']
        self.coefficient_of_variation = env_config['coefficient_of_variation']
        self.max_order = env_config['max_order']
        self.order_cost = env_config['order_cost']
        self.outdated_cost = env_config['outdated_cost']
        self.lost_sales_cost = env_config['lost_sales_cost']
        self.holding_cost = env_config['holding_cost']
        self.use_FIFO = env_config['use_FIFO']
        self.use_LIFO = env_config['use_LIFO']
        self.simulation_time = env_config['simulation_time']
        self.perish_time = env_config['perish_time'] 
        self.warmup_period = env_config['warmup_period']
        
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
        
        costs = [
            action * self.order_cost, # order cost
            calc_state[-1] * self.cost_outdate, # outdate cost
            demand * self.cost_lost, # lost sales cost
            sum(next_state[i] * self.holding_cost for i in range(self.Life_time+1, self.Lead_time+self.Life_time))
        ] # holding cost
        
        perish_cost = sum(min(next_state[i], 0)* self.perish_time for i in range(self.Lead_time, self.Lead_time+self.Life_time))
        costs.append(perish_cost)
        
        self.reward = -sum(costs) if self.current_time >= self.warmup_period else 0
        self.current_time += 1
        self.state = next_state[1:self.Lead_time + self.Life_time]
        
    def update_demand(self, demand, state):
        demand_used = min(demand, state)
        demand -= demand_used
        state -= demand_used
        return demand, max(state, 0)
    
    def is_finished(self):
        return self.current_time == self.simulation_time
    
    def reset(self):
        self.current_time = 0
        self.state = [0] * (self.Lead_time+ self.Life_time-1)
        print('reset environment...')
        return self.state, self.current_time
    
    def render(self):
        print('---------------------------------------------------')
        print(f'*****   Period {self.current_time}   *****')
        inventory_on_hand = [self.render_state[i] for i in range(self.Lead_time, self.Lead_time + self.Life_time)]
        inventory_in_pipeline = self.render_state[:self.Lead_time]  # + [self.action] if you want to include self.action
        print(f'Inventory on hand: {inventory_on_hand}')
        print(f'Order placed: {self.action}')
        print(f'Orders in pipeline: {inventory_in_pipeline}')
        print(f'Demand encountered: {self.demand}')
        print(f'Costs: {self.reward}')
        
    def random_action(self):
        return random.sample(self.action_space, 1)[0]
        
    @classmethod
    def from_dict(cls, env_config_dict):
        return cls(env_config_dict)
