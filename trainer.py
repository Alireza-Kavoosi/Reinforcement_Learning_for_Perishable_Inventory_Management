from environment_train import RetailEnvironment
from DQN import DQNAgent

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
    'warmup_period': 20,
    'perish_time': 5
}

DQN_config = {
    'gamma' : 0.99
}
env = RetailEnvironment.from_dict(config)
agent = DQNAgent(DQN_config)

state = env.reset()
done = False
while not done:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    agent.remember(state, action, reward, next_state, done)
    state = next_state
