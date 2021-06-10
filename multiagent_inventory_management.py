from inv_mangement_env import MultiAgentInvManagement
import ray
from ray.rllib import agents
from ray import tune
import numpy as np
import matplotlib.pyplot as plt


def env_creator(env_config):

    return MultiAgentInvManagement(env_config)


test_env = MultiAgentInvManagement({})

env_name = "MultiAgentInventoryManagement"
env_config = {}
tune.register_env(env_name, env_creator(env_config))

print(test_env.reset())
print(test_env.step({'retailer':10, 'wholesaler':20, 'distributor':20, 'factory':10}))
print(test_env.step({'retailer':10, 'wholesaler':20, 'distributor':20, 'factory':10}))
print(test_env.step({'retailer':10, 'wholesaler':20, 'distributor':20, 'factory':10}))
print(test_env.step({'retailer':10, 'wholesaler':20, 'distributor':20, 'factory':10}))