from environments.MAIM_env import MultiAgentInvManagement
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
import numpy as np
import matplotlib.pyplot as plt
from utils import get_config, get_trainer
from CC_Model import CentralizedCriticModel, FillInActions, central_critic_observer
from gym.spaces import Dict

#%% Environment and Agent Configuration

# Set script seed
SEED = 52
np.random.seed(seed=SEED)

# Environment creator function for environment registration
def env_creator(configuration):
    env = MultiAgentInvManagement(configuration)
    return env


# Environment Configuration
num_stages = 2
num_periods = 30
customer_demand = np.ones(num_periods) * 5
mu = 5
lower_upper = (1, 5)
init_inv = np.ones(num_stages)*10
inv_target = np.ones(num_stages) * 0
inv_max = np.ones(num_stages) * 30
price = np.array([3, 2, 1])
stock_cost = np.array([0.4, 0.4])
backlog_cost = np.array([0.55, 0.45])
delay = np.array([0, 0], dtype=np.int32)

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
    "customer_demand": customer_demand,
    "init_inv": init_inv,
    "price": price,
    "stock_cost": stock_cost,
    "backlog_cost": backlog_cost,
    "demand_dist": demand_distribution,
    "inv_target": inv_target,
    "inv_max": inv_max,
    "delay": delay,
    "independent": False,
    "seed": 52,
    parameter: parameter_value
}
CONFIG = env_config.copy()

# Test environment
test_env = MultiAgentInvManagement(env_config)
obs_space = test_env.observation_space
act_space = test_env.action_space
num_agents = test_env.num_agents
cc_obs_space = Dict({
    "own_obs": obs_space,
    "opponent_obs": obs_space,
    "opponent_action": act_space,
})


# Define policies to train
policy_graphs = {}
for i in range(num_agents):
    policy_graphs[agent_ids[i]] = None, cc_obs_space, act_space, {}


# Policy Mapping function to map each agent to appropriate stage policy

def policy_mapping_fn(agent_id):
    for i in range(num_stages):
        if agent_id.startswith(agent_ids[i]):
            return agent_ids[i]

ModelCatalog.register_custom_model(
        "cc_model", CentralizedCriticModel)

# Algorithm used
algorithm = 'ppo'
# Training Set-up
ray.init(ignore_reinit_error=True, local_mode=True, num_cpus=4)
rl_config = get_config(algorithm, num_periods=num_periods)
rl_config["multiagent"] = {
    "policies": policy_graphs,
    "policy_mapping_fn": policy_mapping_fn,
    "replay_mode": "independent",
    "observation_fn": central_critic_observer
}
rl_config["num_workers"] = 0
rl_config["normalize_actions"] = False
rl_config["callbacks"] = FillInActions
rl_config["env"] = "MultiAgentInventoryManagement"
rl_config["env_config"] = CONFIG
rl_config["framework"] = 'torch'
rl_config["lr"] = 1e-5
rl_config["seed"] = 52
rl_config["batch_mode"] = "complete_episodes"
rl_config["model"]["custom_model"] = "cc_model"

agent = get_trainer(algorithm, rl_config, "MultiAgentInventoryManagement")

#%% Training

# Training
iters = 5
validation_interval = 10
num_validation = 100
results = []
mean_eval_rewards = np.zeros((num_stages, iters//validation_interval))
std_eval_rewards = np.zeros((num_stages, iters//validation_interval))
valid_demand = test_env.dist.rvs(size=(num_validation, test_env.num_periods), **test_env.dist_param)
eval_num = 0
eval_episode = []
for i in range(iters):
    res = agent.train()
    results.append(res)
    if (i + 1) % 1 == 0:
        print('\rIter: {}\tReward: {:.2f}'.format(
            i + 1, res['episode_reward_mean']), end='')
    '''
    if (i + 1) % validation_interval == 0:
        eval_episode.append(res['episodes_total'])
        reward_array = np.zeros((num_stages, num_validation))
        for j in range(num_validation):
            demand = valid_demand[j]
            obs = test_env.reset(customer_demand=demand)
            done = False
            while not done:
                action = {}
                for m in range(num_stages):
                    stage_policy = agent_ids[m]
                    action[stage_policy] = agent.compute_action(obs[stage_policy], policy_id=stage_policy)
                obs, r, dones, info = test_env.step(action)
                done = dones['__all__']
                for m in range(num_stages):
                    reward_array[m, j] += r[agent_ids[m]]

        mean_eval_rewards[:, eval_num] = np.mean(reward_array, axis=1)
        std_eval_rewards[:, eval_num] = np.std(reward_array, axis=1)
        eval_num += 1
    '''


ray.shutdown()
