from environments.MAIM_env import MultiAgentInvManagement
from ray import tune
import ray
import numpy as np
import random
from ray.tune.schedulers import PopulationBasedTraining
from models.RNN_Model import RNNModel, SharedRNNModel
from ray.rllib.models import ModelCatalog
import torch
#%% Environment Configuration

# Set script seed
SEED = 52
np.random.seed(seed=SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Environment creator function for environment registration
def env_creator(configuration):
    env = MultiAgentInvManagement(configuration)
    return env

# Register Custom models
ModelCatalog.register_custom_model(
        "rnn_model", RNNModel)

ModelCatalog.register_custom_model(
        "shared_rnn_model", SharedRNNModel)

# Environment Configuration
num_stages = 4
num_periods = 30
customer_demand = np.ones(num_periods) * 5
mu = 5
lower_upper = (1, 5)
init_inv = np.ones(num_stages)*10
inv_target = np.ones(num_stages) * 0
inv_max = np.ones(num_stages) * 30
price = np.array([5, 4, 3, 2, 1])
stock_cost = np.array([0.35, 0.3, 0.4, 0.2])
backlog_cost = np.array([0.5, 0.7, 0.6, 0.9])
delay = np.array([1, 2, 3, 1], dtype=np.int8)
standardise_state = True
standardise_actions = True
a = -1
b = 1
time_dependency = False
use_lstm = False
prev_actions = False
prev_demand = False
prev_length = 1
independent = False

demand_distribution = "poisson"

if demand_distribution == "custom":
    parameter = "customer_demand"
    parameter_value = customer_demand
elif demand_distribution == 'poisson':
    parameter = "mu"
    parameter_value = mu
elif demand_distribution == "uniform":
    parameter = "lower_upper"
    parameter_value = lower_upper

# Agent/Policy ids of the 3-stage and 4-stage configurations
agent_ids = []
for i in range(num_stages):
    agent_id = "stage_" + str(i)
    agent_ids.append(agent_id)

env_name = "MultiAgentInventoryManagement"
tune.register_env(env_name, env_creator)

env_config = {
    "num_stages": num_stages,
    "num_periods": num_periods,
    "init_inv": init_inv,
    "price": price,
    "stock_cost": stock_cost,
    "backlog_cost": backlog_cost,
    "demand_dist": demand_distribution,
    "inv_target": inv_target,
    "inv_max": inv_max,
    "seed": SEED,
    "delay": delay,
    parameter: parameter_value,
    "standardise_state": standardise_state,
    "standardise_actions": standardise_actions,
    "a": a,
    "b": b,
    "time_dependency": time_dependency,
    "prev_demand": prev_demand,
    "prev_actions": prev_actions,
    "prev_length": prev_length,
    "independent": independent
}
CONFIG = env_config.copy()

# Test environment
test_env = MultiAgentInvManagement(env_config)
obs_space = test_env.observation_space
act_space = test_env.action_space
num_agents = test_env.num_agents

#%% Agent Configuration
# Define policies to train
policy_graphs = {}
for i in range(num_agents):
    policy_graphs[agent_ids[i]] = None, obs_space, act_space, {}


# Policy Mapping function to map each agent to appropriate stage policy

def policy_mapping_fn(agent_id, episode, **kwargs):
    for i in range(num_stages):
        if agent_id.startswith(agent_ids[i]):
            return agent_ids[i]

# Postprocess the perturbed config to ensure it's still valid
def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config

hp_mutations = dict()
hp_mutations["lambda"] = lambda: random.uniform(0.9, 1.0)
hp_mutations["gamma"] = lambda: random.uniform(0.95, 0.99)
hp_mutations["kl_coeff"] = lambda: random.uniform(0.1, 0.6)
hp_mutations["kl_target"] = lambda: random.uniform(0.003, 0.03)
hp_mutations["entropy_coeff"] = lambda: random.uniform(0, 0.01)
hp_mutations["clip_param"] = lambda: random.uniform(0.1, 0.4)
hp_mutations["lr"] = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
hp_mutations["num_sgd_iter"] = lambda: random.randint(3, 30)
hp_mutations["train_batch_size"] = lambda: random.randint(num_periods*50, num_periods*300)
hp_mutations["sgd_minibatch_size"] = lambda: random.randint(64, 256)
hp_mutations["vf_clip_param"] = lambda: random.randint(20, 2000)
hp_mutations["model"] = dict()
if use_lstm:
    hp_mutations["model"]["custom_model_config"] = dict()
    hp_mutations["model"]["custom_model_config"]["fc_size"] = [16, 32, 64, 128, 256]
    hp_mutations["model"]["custom_model_config"]["lstm_state_size"] = [64, 128, 256]

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=5,
    resample_probability=0.4,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations=hp_mutations,
    custom_explore_fn=explore)


rl_config = dict()
rl_config["env"] = "MultiAgentInventoryManagement"
rl_config["multiagent"] = dict()
rl_config["multiagent"]["policies"] = policy_graphs
rl_config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn
rl_config["multiagent"]["replay_mode"] = "independent"
rl_config["env_config"] = CONFIG
rl_config["batch_mode"] = "complete_episodes"
rl_config["normalize_actions"] = False
rl_config["framework"] = "torch"
rl_config["seed"] = SEED
rl_config["use_critic"] = True
rl_config["use_gae"] = True
rl_config["num_gpus"] = 0
rl_config["num_workers"] = 2
rl_config["num_cpus_per_worker"] = 1
rl_config["shuffle_sequences"] = True
rl_config["vf_loss_coeff"] = 1
rl_config["grad_clip"] = None
rl_config["model"] = dict()
rl_config["model"]["fcnet_activation"] = "relu"
rl_config["model"]["vf_share_layers"] = False
# These params are tuned from a fixed starting value.
rl_config["vf_clip_param"] = 1000
rl_config["lambda"] = 0.95
rl_config["gamma"] = 0.99
rl_config["clip_param"] = 0.2
rl_config["lr"] = 1e-5
rl_config["kl_coeff"] = 0.2
rl_config["kl_target"] = 0.01
rl_config["entropy_coeff"] = 0
rl_config["env_config"]["prev_length"] = prev_length
rl_config["sgd_minibatch_size"] = 128
rl_config["num_sgd_iter"] = 10
rl_config["train_batch_size"] = num_periods*100
# These params start off randomly drawn from a set.
rl_config["model"]["fcnet_hiddens"] = [tune.choice([64, 128, 256]), tune.choice([64, 128, 256])]
if use_lstm:
    rl_config["model"]["custom_model"] = "rnn_model"
    rl_config["model"]["max_seq_len"] = num_periods
    rl_config["model"]["custom_model_config"] = dict()
    rl_config["model"]["custom_model_config"]["use_initial_fc"] = True
    rl_config["model"]["custom_model_config"]["fc_size"] = 64
    rl_config["model"]["custom_model_config"]["lstm_state_size"] = tune.choice([128, 256])

ray.init(num_cpus=5, num_gpus=0)
analysis = tune.run(
        "PPO",
        name="pbt_MAInvManagement_test",
        scheduler=pbt,
        num_samples=8,
        metric="episode_reward_mean",
        mode="max",
        stop={"training_iteration": 200},
        config=rl_config,
        max_failures=5)

print("best hyperparameters: ", analysis.best_config)
if use_lstm:
    np.save(('MA_rnn_hyperparams.npy'), analysis)
    np.save(('MA_rnn_hyperparams_config.npy'), analysis.best_config)
else:
    np.save(('MA_hyperparams.npy'), analysis)
    np.save(('MA_hyperparams_config.npy'), analysis.best_config)