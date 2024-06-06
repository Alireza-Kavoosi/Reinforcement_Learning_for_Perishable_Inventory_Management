from environment_train import RetailEnvironment
from DQN import DQNAgent

env_config = {
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

DQN_config = {
    'gamma' : 0.99
}
env = RetailEnvironment.from_dict(env_config)