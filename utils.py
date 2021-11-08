"""
Helper functions for getting agents and rl configs
"""


from ray.rllib import agents
import os
import numpy as np
import copy

def get_config(algorithm, num_periods, seed=52):
    if algorithm == 'ppo':
        config = agents.ppo.DEFAULT_CONFIG.copy()
        config["model"] = {
            "vf_share_layers": False,
            "fcnet_activation": 'relu',
            "fcnet_hiddens": [64, 64]
        }
        # Should use a critic as a baseline (otherwise don't use value baseline; required for using GAE).
        config["use_critic"] = True
        # If true, use the Generalized Advantage Estimator (GAE)
        config["use_gae"] = True
        # The GAE (lambda) parameter.
        config["lambda"] = 0.95
        config["gamma"] = 0.99
        # Initial coefficient for KL divergence.
        config["kl_coeff"] = 0.2
        # Number of timesteps collected for each SGD round. This defines the size of each SGD epoch.
        config["train_batch_size"] = num_periods*100
        # Number of SGD iterations in each outer loop (i.e., number of epochs to execute per train batch).
        config["num_sgd_iter"] = 10
        # Total SGD batch size across all devices for SGD. This defines the minibatch size within each epoch.
        config["sgd_minibatch_size"] = 120
        # Whether to shuffle sequences in the batch when training (recommended).
        config["shuffle_sequences"] = True
        # Coefficient of the value function loss. IMPORTANT: you must tune this if you set vf_share_layers=True inside your model's config.
        config["vf_loss_coeff"] = 1.0
        # Coefficient of the entropy regularizer.
        config["entropy_coeff"] = 0.0
        # Decay schedule for the entropy regularizer.
        config["entropy_coeff_schedule"] = None
        # PPO clip parameter.
        config["clip_param"] = 0.2
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        config["vf_clip_param"] = 10_000.0
        # If specified, clip the global norm of gradients by this amount.
        config["grad_clip"] = None
        # Target value for KL divergence.
        config["kl_target"] = 0.01
        # Seed
        config["seed"] = seed
        # Learning rate
        config["lr"] = 1e-5
        # Framework
        config["framework"] = 'torch'

        config["batch_mode"] = "complete_episodes"
        config["normalize_actions"] = False

    else:
        raise Exception('Not Implemented')

    return config


def get_trainer(algorithm, rl_config, environment):
    if algorithm == 'ddpg':
        agent = agents.ddpg.DDPGTrainer(config=rl_config, env=environment)

    elif algorithm == 'ppo':
        agent = agents.ppo.PPOTrainer(config=rl_config, env=environment)

    elif algorithm == 'sac':
        agent = agents.sac.SACTrainer(config=rl_config, env=environment)

    else:
        raise Exception('Not Implemented')

    return agent

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_connections(connections):
    for node, children in connections.items():
        if children:
            for child in children:
                if child < node:
                    raise Exception("Downstream node cannot have a smaller index number than upstream node")

def create_network(connections):
    num_nodes = max(connections.keys())
    network = np.zeros((num_nodes + 1, num_nodes + 1))
    for parent, children in connections.items():
        if children:
            for child in children:
                network[parent][child] = 1

    return network


def get_stage(node, network):
    reached_root = False
    stage = 0
    counter = 0
    if node == 0:
        return 0
    while not reached_root:
        for i in range(len(network)):
            if network[i][node] == 1:
                stage += 1
                node = i
                if node == 0:
                    return stage
        counter += 1
        if counter > len(network):
            raise Exception("Infinite Loop")



def get_retailers(network):
    retailers = []
    for i in range(len(network)):
        if not any(network[i]):
            retailers.append(i)

    return retailers

