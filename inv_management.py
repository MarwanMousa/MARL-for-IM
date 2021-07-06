from environments.IM_env import InvManagement
import ray
from ray.rllib import agents
from ray import tune
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import os
from scipy.optimize import minimize
from utils import get_config, get_trainer
from scipy.stats import poisson, randint
from base_restock_policy import optimize_inventory_policy, dfo_func, base_stock_policy
#%% Environment and Agent Configuration

# Environment creator function for environment registration
def env_creator(configuration):
    env = InvManagement(configuration)
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

env_name = "InventoryManagement"
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
    "seed": 52,
    "delay": delay,
    parameter: parameter_value
}
CONFIG = env_config.copy()

# Test environment
test_env = InvManagement(env_config)

# Algorithm used
algorithm = 'ppo'
# Training Set-up
ray.init(ignore_reinit_error=True, local_mode=True, num_cpus=1)
rl_config = get_config(algorithm, num_periods)
rl_config["num_workers"] = 0
rl_config["batch_mode"] = "complete_episodes"
rl_config["normalize_actions"] = False
rl_config["env_config"] = CONFIG
rl_config["framework"] = 'torch'
rl_config["lr"] = 1e-5
rl_config["seed"] = 52
rl_config["env"] = "InventoryManagement"
#rl_config["evaluation_interval"] = 40
#rl_config["evaluation_num_episodes"] = 5
#rl_config["evaluation_num_workers"] = 1
#rl_config["evaluation_config"] = {
#    "explore": False
#}

agent = get_trainer(algorithm, rl_config, "InventoryManagement")

#%% RL Training

# Training
iters = 800
results = []
mean_eval_rewards = []
std_eval_rewards = []
eval_episode = []
for i in range(iters):
    res = agent.train()
    results.append(res)
    if (i + 1) % 1 == 0:
        print('\rIter: {}\tReward: {:.2f}'.format(
            i + 1, res['episode_reward_mean']), end='')
    if (i + 1) % 10 == 0:
        eval_episode.append(res['episodes_total'])
        list_eval_rewards = []
        for j in range(30):
            np.random.seed(seed=j)
            demand = test_env.dist.rvs(size=test_env.num_periods, **test_env.dist_param)
            s = test_env.reset(customer_demand=demand)
            reward = 0
            done = False
            while not done:
                a = np.round(agent.compute_action(s), 0).astype(int)
                s, r, done, _ = test_env.step(a)
                reward += r
            list_eval_rewards.append(reward)
        mean_eval_rewards.append(np.mean(list_eval_rewards))
        std_eval_rewards.append(np.std(list_eval_rewards))
        np.random.seed(seed=52)

    # chkpt_file = agent.save('/Users/marwanmousa/University/MSc_AI/Individual_Project/MARL-and-DMPC-for-OR/checkpoints/multi_agent')

ray.shutdown()

training_time = datetime.datetime.now().strftime("D%dM%m_h%Hm%M")
save_path = '/Users/marwanmousa/University/MSc_AI/Individual_Project/MARL-and-DMPC-for-OR/figures/single_agent/' + training_time
json_env_config = save_path + '/env_config.json'
os.makedirs(os.path.dirname(json_env_config), exist_ok=True)
with open(json_env_config, 'w') as fp:
    for key, value in CONFIG.items():
        if isinstance(value, np.ndarray):
            CONFIG[key] = CONFIG[key].tolist()
    json.dump(CONFIG, fp)

RL_config = {}
RL_config["algorithm"] = algorithm
RL_config["lr"] = rl_config["lr"]
RL_config["seed"] = rl_config["seed"]
json_rl_config = save_path + '/rl_config.json'
with open(json_rl_config, 'w') as fp:
    json.dump(RL_config, fp)

#%% Derivative Free Optimization
policy, out = optimize_inventory_policy(test_env, dfo_func)
print("Re-order levels: {}".format(policy))
print("DFO Info:\n{}".format(out))

eps = 1000
dfo_rewards = []
for i in range(eps):
    np.random.seed(seed=i)
    demand = test_env.dist.rvs(size=test_env.num_periods, **test_env.dist_param)
    test_env.reset(customer_demand=demand)
    dfo_reward = 0
    done = False
    while not done:
        dfo_action = base_stock_policy(policy, test_env)
        s, r, done, _ = test_env.step(dfo_action)
        dfo_reward += r
    dfo_rewards.append(dfo_reward)

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

mean_dfo_rewards = np.mean(dfo_rewards)
std_dfo_rewards = np.std(dfo_rewards)


fig, ax = plt.subplots()
ax.fill_between(np.arange(len(mean_rewards)),
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 alpha=0.3)
ax.plot(mean_rewards, label='Mean Rewards')
ax.fill_between(np.arange(len(mean_rewards)),
                 np.ones(len(mean_rewards)) * (mean_dfo_rewards - std_dfo_rewards),
                 np.ones(len(mean_rewards)) * (mean_dfo_rewards + std_dfo_rewards),
                 alpha=0.3)
ax.plot(np.arange(len(mean_rewards)), np.ones(len(mean_rewards)) * (mean_dfo_rewards) , label='Mean DFO Rewards')
ax.plot(eval_episode, mean_eval_rewards, label='Mean Eval Rewards')
ax.fill_between(eval_episode,
                np.array(mean_eval_rewards) - np.array(std_eval_rewards),
                np.array(mean_eval_rewards) + np.array(std_eval_rewards),
                alpha=0.3)
ax.set_ylabel('Rewards')
ax.set_xlabel('Episode')
ax.set_title('Aggregate Training Rewards')
ax.legend()

rewards_name = save_path + '/training_rewards.png'
plt.savefig(rewards_name, dpi=200)
plt.show()

#%% Test rollout

# run until episode ends
episode_reward = 0
done = False
array_obs = np.zeros((num_stages, 4, num_periods + 1))
array_actions = np.zeros((num_stages, num_periods))
array_profit = np.zeros((num_stages, num_periods))
array_profit_sum = np.zeros(num_periods)
array_demand = np.zeros((num_stages, num_periods))
array_ship = np.zeros((num_stages, num_periods))
array_acquisition = np.zeros((num_stages, num_periods))
array_rewards = np.zeros(num_periods)
period = 0

test_seed = 420
np.random.seed(seed=test_seed)
test_demand = test_env.dist.rvs(size=test_env.num_periods, **test_env.dist_param)
obs = test_env.reset(customer_demand=test_demand)
array_obs[:, :, 0] = obs

while not done:
    action = np.round(agent.compute_action(obs), 0).astype(int)
    obs, reward, done, info = test_env.step(action)
    array_obs[:, :, period + 1] = obs
    array_actions[:, period] = action
    array_rewards[period] = reward
    array_profit[:, period] = info['profit']
    array_profit_sum[period] = np.sum(info['profit'])
    array_demand[:, period] = info['demand']
    array_ship[:, period] = info['ship']
    array_acquisition[:, period] = info['acquisition']
    episode_reward += reward
    period += 1

#%% Plots
fig, axs = plt.subplots(4, num_stages, figsize=(20, 8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs = axs.ravel()

for i in range(num_stages):
    axs[i].plot(array_obs[i, 0, :], label='Inventory')
    axs[i].plot(array_obs[i, 1, :], label='Backlog')
    axs[i].plot(array_obs[i, 2, :], label='Unfulfilled orders')
    axs[i].plot([0, num_periods + 1], [inv_target[i], inv_target[i]], label='Target Inventory', color='r')
    axs[i].legend()
    title = 'Stage ' + str(i)
    axs[i].set_title(title)
    axs[i].set_ylabel('Products')
    axs[i].set_xlim(0, num_periods)

    axs[i+num_stages].plot(np.arange(0, num_periods), array_acquisition[i, :], label='Acquisition')
    axs[i+num_stages].plot(np.arange(0, num_periods), array_actions[i, :], label='Replenishment Order', color='k')
    axs[i+num_stages].legend()
    axs[i+num_stages].set_ylabel('Products')
    axs[i+num_stages].set_xlim(0, num_periods)

    axs[i+num_stages*2].plot(np.arange(0, num_periods), array_demand[i, :], label='demand')
    axs[i+num_stages*2].plot(np.arange(0, num_periods), array_ship[i, :], label='shipment')
    axs[i+num_stages*2].legend()
    axs[i+num_stages*2].set_ylabel('Products')
    axs[i+num_stages*2].set_xlim(0, num_periods)

    axs[i+num_stages*3].plot(np.arange(1, num_periods+1), array_profit[i, :], label='periodic profit')
    axs[i+num_stages*3].plot(np.arange(1, num_periods + 1), np.cumsum(array_profit[i, :]), label='cumulative profit')
    axs[i+num_stages*3].plot([0, num_periods], [0, 0], color='k')
    axs[i+num_stages*3].set_xlabel('Period')
    axs[i+num_stages*3].set_ylabel('Profit')
    axs[i+num_stages*3].legend()
    axs[i+num_stages*3].set_xlim(0, num_periods)

test_name = save_path + '/test_rollout.png'
plt.savefig(test_name, dpi=200)
plt.show()

# Rewards plots
fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
ax.plot(np.arange(1, num_periods+1), array_rewards, label='periodic reward')
ax.plot(np.arange(1, num_periods+1), np.cumsum(array_rewards), label='cumulative reward')
ax.plot(np.arange(1, num_periods+1), array_profit_sum, label='periodic profit')
ax.plot(np.arange(1, num_periods+1), np.cumsum(array_profit_sum), label='cumulative profit')
ax.plot([0, num_periods], [0, 0], color='k')
ax.set_title('Aggregate Rewards')
ax.set_xlabel('Period')
ax.set_ylabel('Rewards/profit')
ax.legend()
ax.set_xlim(0, num_periods)

test_rewards_name = save_path + '/test_rollout_rewards.png'
plt.savefig(test_rewards_name, dpi=200)
plt.show()

#%% DFO Test run

# run until episode ends
dfo_episode_reward = 0

obs = test_env.reset()
dfo_array_obs = np.zeros((num_stages, 4, num_periods + 1))
dfo_array_actions = np.zeros((num_stages, num_periods))
dfo_array_profit = np.zeros((num_stages, num_periods))
dfo_array_profit_sum = np.zeros(num_periods)
dfo_array_demand = np.zeros((num_stages, num_periods))
dfo_array_ship = np.zeros((num_stages, num_periods))
dfo_array_acquisition = np.zeros((num_stages, num_periods))
dfo_array_rewards = np.zeros(num_periods)
period = 0

test_env.reset(customer_demand=test_demand)
dfo_episode_reward = 0
done = False
while not done:
    dfo_action = base_stock_policy(policy, test_env)
    dfo_obs, dfo_reward, done, dfo_info = test_env.step(dfo_action)
    dfo_array_obs[:, :, period + 1] = dfo_obs
    dfo_array_actions[:, period] = dfo_action
    dfo_array_rewards[period] = dfo_reward
    dfo_array_profit[:, period] = dfo_info['profit']
    dfo_array_profit_sum[period] = np.sum(dfo_info['profit'])
    dfo_array_demand[:, period] = dfo_info['demand']
    dfo_array_ship[:, period] = dfo_info['ship']
    dfo_array_acquisition[:, period] = dfo_info['acquisition']
    dfo_episode_reward += reward
    period += 1

#%% DFO Plots
fig, axs = plt.subplots(4, num_stages, figsize=(20, 8), facecolor='w', edgecolor='k')
fig.suptitle('DFO', fontsize=16)
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs = axs.ravel()

for i in range(num_stages):
    axs[i].plot(dfo_array_obs[i, 0, :], label='Inventory')
    axs[i].plot(dfo_array_obs[i, 1, :], label='Backlog')
    axs[i].plot(dfo_array_obs[i, 2, :], label='Unfulfilled orders')
    axs[i].plot([0, num_periods + 1], [inv_target[i], inv_target[i]], label='Target Inventory', color='r')
    axs[i].legend()
    title = 'Stage ' + str(i)
    axs[i].set_title(title)
    axs[i].set_ylabel('Products')
    axs[i].set_xlim(0, num_periods)

    axs[i+num_stages].plot(np.arange(0, num_periods), dfo_array_acquisition[i, :], label='Acquisition')
    axs[i+num_stages].plot(np.arange(0, num_periods), dfo_array_actions[i, :], label='Replenishment Order', color='k')
    axs[i+num_stages].legend()
    axs[i+num_stages].set_ylabel('Products')
    axs[i+num_stages].set_xlim(0, num_periods)

    axs[i+num_stages*2].plot(np.arange(0, num_periods), dfo_array_demand[i, :], label='demand')
    axs[i+num_stages*2].plot(np.arange(0, num_periods), dfo_array_ship[i, :], label='shipment')
    axs[i+num_stages*2].legend()
    axs[i+num_stages*2].set_ylabel('Products')
    axs[i+num_stages*2].set_xlim(0, num_periods)

    axs[i+num_stages*3].plot(np.arange(1, num_periods+1), dfo_array_profit[i, :], label='periodic profit')
    axs[i+num_stages*3].plot(np.arange(1, num_periods+1), np.cumsum(dfo_array_profit[i, :]), label='cumulative profit')
    axs[i+num_stages*3].plot([0, num_periods], [0, 0], color='k')
    axs[i+num_stages*3].set_xlabel('Period')
    axs[i+num_stages*3].set_ylabel('Profit')
    axs[i+num_stages*3].legend()
    axs[i+num_stages*3].set_xlim(0, num_periods)

plt.show()

# Rewards plots
fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
ax.plot(np.arange(1, num_periods+1), dfo_array_rewards, label='periodic reward')
ax.plot(np.arange(1, num_periods+1), np.cumsum(dfo_array_rewards), label='cumulative reward')
ax.plot(np.arange(1, num_periods+1), dfo_array_profit_sum, label='periodic profit')
ax.plot(np.arange(1, num_periods+1), np.cumsum(dfo_array_profit_sum), label='cumulative profit')
ax.plot([0, num_periods], [0, 0], color='k')
ax.set_title('Aggregate Rewards DFO')
ax.set_xlabel('Period')
ax.set_ylabel('Rewards/profit')
ax.legend()
ax.set_xlim(0, num_periods)

plt.show()