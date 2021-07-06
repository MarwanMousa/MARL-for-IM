from environments.MAIM_env import MultiAgentInvManagement
import ray
from ray.rllib import agents
from ray import tune
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import json
from utils import get_config, get_trainer

#%% Environment and Agent Configuration

# Environment creator function for environment registration
def env_creator(configuration):
    env = MultiAgentInvManagement(configuration)
    return env


# Environment Configuration
num_stages = 3
num_periods = 30
customer_demand = np.ones(num_periods) * 5
mu = 5
lower_upper = (1, 5)
init_inv = np.ones(num_stages)*10
inv_target = np.ones(num_stages) * 0
inv_max = np.ones(num_stages) * 30
price = np.array([4, 3, 2, 1])
stock_cost = np.array([0.4, 0.4, 0.4])
backlog_cost = np.array([0.55, 0.5, 0.45])
delay = np.array([0, 0, 0], dtype=np.int8)

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
if num_stages == 4:
    agent_ids = ["retailer", "wholesaler", "distributor", "factory"]
elif num_stages == 3:
    agent_ids = ["retailer", "distributor", "factory"]

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


# Define policies to train
policy_graphs = {}
for i in range(num_agents):
    policy_graphs[agent_ids[i]] = None, obs_space, act_space, {}


# Policy Mapping function to map each agent to appropriate stage policy
def policy_mapping_fn(agent_id):
    if agent_id.startswith("retailer"):
        return "retailer"
    elif agent_id.startswith("wholesaler"):
        return "wholesaler"
    elif agent_id.startswith("distributor"):
        return "distributor"
    elif agent_id.startswith("factory"):
        return "factory"

# Algorithm used
algorithm = 'ddpg'
# Training Set-up
ray.init(ignore_reinit_error=True, local_mode=True, num_cpus=4)
rl_config = get_config(algorithm, num_periods=num_periods)
rl_config["multiagent"] = {
    "policies": policy_graphs,
    "policy_mapping_fn": policy_mapping_fn,
    "replay_mode": "independent"
}
rl_config["num_workers"] = 0
rl_config["normalize_actions"] = False
rl_config["env_config"] = CONFIG
rl_config["framework"] = 'torch'
rl_config["lr"] = 1e-5
rl_config["seed"] = 52

agent = get_trainer(algorithm, rl_config, "MultiAgentInventoryManagement")
#agent = agents.ddpg.DDPGTrainer(config=rl_config, env=MultiAgentInvManagement)

#%% Training

# Training
iters = 30
results = []
mean_eval_rewards = np.zeros((num_stages, iters//10))
std_eval_rewards = np.zeros((num_stages, iters//10))
eval_num = 0
eval_episode = []
for i in range(iters):
    res = agent.train()
    results.append(res)
    if (i + 1) % 1 == 0:
        #chkpt_file = agent.save('/Users/marwanmousa/University/MSc_AI/Individual_Project/MARL-and-DMPC-for-OR/checkpoints/multi_agent')
        print('\rIter: {}\tReward: {:.2f}'.format(
            i + 1, res['episode_reward_mean']), end='')

    if (i + 1) % 10 == 0:
        eval_episode.append(res['episodes_total'])
        for j in range(30):
            np.random.seed(seed=j)
            demand = test_env.dist.rvs(size=test_env.num_periods, **test_env.dist_param)
            obs = test_env.reset(customer_demand=demand)
            reward_array = np.zeros((num_stages, 30))
            done = False
            while not done:
                action = {}
                for m in range(num_stages):
                    stage_policy = agent_ids[m]
                    action[stage_policy] = np.round(agent.compute_action(obs[stage_policy], policy_id=stage_policy), 0).astype(int)
                obs, r, dones, info = test_env.step(action)
                done = dones['__all__']
                for m in range(num_stages):
                    reward_array[m, j] += r[agent_ids[m]]

        mean_eval_rewards[:, eval_num] = np.mean(reward_array, axis=1)
        std_eval_rewards[:, eval_num] = np.std(reward_array, axis=1)
        eval_num += 1
        np.random.seed(seed=52)

ray.shutdown()

training_time = datetime.datetime.now().strftime("D%dM%m_h%Hm%M")
save_path = '/Users/marwanmousa/University/MSc_AI/Individual_Project/MARL-and-DMPC-for-OR/figures/multi_agent/' \
            + training_time
json_config = save_path + '/env_config.json'
os.makedirs(os.path.dirname(json_config), exist_ok=True)
with open(json_config, 'w') as fp:
    for key, value in CONFIG.items():
        if isinstance(value, np.ndarray):
            CONFIG[key] = CONFIG[key].tolist()
    json.dump(CONFIG, fp)

#%% Reward Plots
p = 100
# Unpack values from each iteration
rewards = np.hstack([i['hist_stats']['episode_reward']
                     for i in results])

mean_rewards = np.array([np.mean(rewards[i - p:i + 1])
                         if i >= p else np.mean(rewards[:i + 1])
                         for i, _ in enumerate(rewards)])

std_rewards = np.array([np.std(rewards[i - p:i + 1])
                        if i >= p else np.std(rewards[:i + 1])
                        for i, _ in enumerate(rewards)])

policy_rewards = {}
policy_mean_rewards = {}
policy_std_rewards = {}
for j in range(num_stages):
    policy_agent = agent_ids[j]
    stat = 'policy_' + policy_agent + '_reward'
    policy_rewards[policy_agent] = np.hstack([i['hist_stats'][stat] for i in results])
    temp = policy_rewards[policy_agent]
    policy_mean_rewards[policy_agent] = np.array([np.mean(temp[i - p:i + 1])
                                                  if i >= p else np.mean(temp[:i + 1])
                                                  for i, _ in enumerate(temp)])
    policy_std_rewards[policy_agent] = np.array([np.std(temp[i - p:i + 1])
                                                 if i >= p else np.std(temp[:i + 1])
                                                 for i, _ in enumerate(temp)])


fig, ax = plt.subplots()
ax.fill_between(np.arange(len(mean_rewards)),
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 label='Standard Deviation', alpha=0.3)
ax.plot(mean_rewards, label='Mean Rewards')
ax.plot(eval_episode, np.sum(mean_eval_rewards, axis=0))
ax.set_ylabel('Rewards')
ax.set_xlabel('Episode')
ax.set_title('Aggregate Training Rewards')
ax.legend()

rewards_name = save_path + '/training_rewards.png'
plt.savefig(rewards_name, dpi=200)
plt.show()
plt.show()

colours = ['r', 'g', 'b', 'k']
fig, ax = plt.subplots()
for i in range(num_agents):
    policy_agent = agent_ids[i]
    ax.plot(policy_mean_rewards[policy_agent], colours[i], label=agent_ids[i])
    ax.fill_between(np.arange(len(policy_mean_rewards[policy_agent])),
                    policy_mean_rewards[policy_agent] - policy_std_rewards[policy_agent],
                    policy_mean_rewards[policy_agent] + policy_std_rewards[policy_agent])
    ax.plot(eval_episode, mean_eval_rewards[i, :])
    ax.set_title('Learning Curve (Rewards)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rewards')
    ax.legend()

rewards_name_policy = save_path + '/training_rewards_policy.png'
plt.savefig(rewards_name, dpi=200)
plt.show()


#%% Test rollout

# run until episode ends
episode_reward = 0
done = False
obs = test_env.reset()
dict_obs = {}
dict_info = {}
dict_actions = {}
dict_rewards = {}
period = 0

# Dict initialisation
for i in range(num_stages):
    stage_policy = agent_ids[i]
    dict_obs[stage_policy] = {}
    dict_info[stage_policy] = {}
    dict_obs[stage_policy]['inventory'] = np.zeros(num_periods + 1)
    dict_obs[stage_policy]['backlog'] = np.zeros(num_periods + 1)
    dict_obs[stage_policy]['order_u'] = np.zeros(num_periods + 1)
    dict_info[stage_policy]['demand'] = np.zeros(num_periods)
    dict_info[stage_policy]['ship'] = np.zeros(num_periods)
    dict_info[stage_policy]['acquisition'] = np.zeros(num_periods)
    dict_info[stage_policy]['actual order'] = np.zeros(num_periods)
    dict_obs[stage_policy]['inventory'][0] = obs[stage_policy][0]
    dict_obs[stage_policy]['backlog'][0] = obs[stage_policy][1]
    dict_obs[stage_policy]['order_u'][0] = obs[stage_policy][2]
    dict_actions[stage_policy] = np.zeros(num_periods)
    dict_rewards[stage_policy] = np.zeros(num_periods)
    dict_rewards['Total'] = np.zeros(num_periods)

while not done:
    action = {}
    for i in range(num_stages):
        stage_policy = agent_ids[i]
        action[stage_policy] = agent.compute_action(obs[stage_policy], policy_id=stage_policy)
    obs, reward, dones, info = test_env.step(action)
    done = dones['__all__']
    for i in range(num_stages):
        stage_policy = agent_ids[i]
        episode_reward += reward[stage_policy]

        dict_obs[stage_policy]['inventory'][period + 1] = obs[stage_policy][0]
        dict_obs[stage_policy]['backlog'][period + 1] = obs[stage_policy][1]
        dict_obs[stage_policy]['order_u'][period + 1] = obs[stage_policy][2]
        dict_info[stage_policy]['demand'][period] = info[stage_policy]['demand']
        dict_info[stage_policy]['ship'][period] = info[stage_policy]['ship']
        dict_info[stage_policy]['acquisition'][period] = info[stage_policy]['acquisition']
        dict_info[stage_policy]['actual order'][period] = info[stage_policy]['actual order']
        dict_actions[stage_policy][period] = action[stage_policy]
        dict_rewards[stage_policy][period] = reward[stage_policy]
        dict_rewards['Total'][period] += reward[stage_policy]

    period += 1

#%% Plots
fig, axs = plt.subplots(4, num_stages, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs = axs.ravel()

for i in range(num_stages):
    stage_policy = agent_ids[i]
    axs[i].plot(dict_obs[stage_policy]['inventory'], label='Inventory')
    axs[i].plot(dict_obs[stage_policy]['backlog'], label='Backlog')
    axs[i].plot(dict_obs[stage_policy]['order_u'], label='Unfulfilled orders')
    axs[i].plot([0, 0], [inv_target[i], inv_target[i]], label='Target Inventory', color='r')
    axs[i].legend()
    axs[i].set_title(stage_policy)
    axs[i].set_ylabel('Products')
    axs[i].set_xlim(0, num_periods)

    axs[i + num_stages].plot(dict_info[stage_policy]['actual order'], label='Replenishment order', color='k')
    axs[i + num_stages].plot(np.arange(0, num_periods), dict_info[stage_policy]['acquisition'], label='Acquisition')
    axs[i + num_stages].legend()
    axs[i + num_stages].set_ylabel('Products')
    axs[i + num_stages].set_xlim(0, num_periods)

    axs[i + num_stages * 2].plot(np.arange(0, num_periods), dict_info[stage_policy]['demand'], label='demand')
    axs[i + num_stages * 2].plot(np.arange(0, num_periods), dict_info[stage_policy]['ship'], label='shipment')

    axs[i + num_stages * 2].legend()
    axs[i + num_stages * 2].set_ylabel('Products')
    axs[i + num_stages * 2].set_xlim(0, num_periods)

    axs[i + num_stages * 3].plot(np.arange(1, num_periods + 1), dict_rewards[stage_policy], label='periodic profit')
    axs[i + num_stages * 3].plot(np.arange(1, num_periods + 1), np.cumsum(dict_rewards[stage_policy]), label='cumulative profit')
    axs[i + num_stages * 3].plot([0, num_periods], [0, 0], color='k')
    axs[i + num_stages * 3].legend()
    axs[i + num_stages * 3].set_xlabel('Period')
    axs[i + num_stages * 3].set_ylabel('Profit')
    axs[i + num_stages * 3].set_xlim(0, num_periods)

test_name = save_path + '/test_rollout.png'
plt.savefig(test_name, dpi=200)
plt.show()

# Rewards plots
fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
ax.plot(np.arange(1, num_periods+1), dict_rewards['Total'], label='periodic profit')
ax.plot(np.arange(1, num_periods+1), np.cumsum(dict_rewards['Total']), label='cumulative profit')
ax.plot([0, num_periods], [0, 0], color='k')
ax.legend()
ax.set_title('Aggregate Rewards')
ax.set_xlabel('Period')
ax.set_ylabel('Rewards')
ax.set_xlim(0, num_periods)

test_rewards_name = save_path + '/test_rollout_rewards.png'
plt.savefig(test_rewards_name, dpi=200)
plt.show()