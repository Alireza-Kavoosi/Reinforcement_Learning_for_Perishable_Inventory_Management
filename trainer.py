<<<<<<< HEAD
from environment_train import RetailEnvironment

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

=======
from environment_test import RetailEnvironment

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

>>>>>>> 7938344f1c344193ab9cb96334396895fd6401b0
env = RetailEnvironment.from_dict(config)