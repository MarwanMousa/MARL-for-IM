from environments.IM_env import InvManagement
import ray
from ray.rllib import agents
from ray import tune
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import os
from utils import get_config, get_trainer
from scipy.stats import poisson, randint
from base_restock_policy import optimize_inventory_policy, dfo_func, base_stock_policy
#%% Environment Configuration

# Set script seed
SEED = 52
np.random.seed(seed=SEED)

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
backlog_cost = np.array([0.6, 0.6, 0.6])
delay = np.array([1, 1, 1], dtype=np.int8)
standardise_state = True
standardise_actions = True
a = -1
b = 1
time_dependency = True
use_lstm = False

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
    "seed": SEED,
    "delay": delay,
    parameter: parameter_value,
    "standardise_state": standardise_state,
    "standardise_actions": standardise_actions,
    "a": a,
    "b": b,
    "time_dependency": time_dependency,
}
CONFIG = env_config.copy()
# Configuration for the base-restock policy with DFO that has no state-action standardisation
DFO_CONFIG = env_config.copy()
DFO_CONFIG["standardise_state"] = False
DFO_CONFIG["standardise_actions"] = False
DFO_CONFIG["time_dependent_states"] = False
# Test environment
test_env = InvManagement(env_config)
DFO_env = InvManagement(DFO_CONFIG)


#%% Derivative Free Optimization
policy, out = optimize_inventory_policy(DFO_env, dfo_func)
print("Re-order levels: {}".format(policy))
print("DFO Info:\n{}".format(out))

eps = 1000
dfo_valid_demand = DFO_env.dist.rvs(size=(eps, DFO_env.num_periods), **DFO_env.dist_param)
dfo_rewards = []
for i in range(eps):
    demand = dfo_valid_demand[i]
    DFO_env.reset(customer_demand=demand)
    dfo_reward = 0
    done = False
    while not done:
        dfo_action = base_stock_policy(policy, DFO_env)
        s, r, done, _ = DFO_env.step(dfo_action)
        dfo_reward += r
    dfo_rewards.append(dfo_reward)

#%% Agent Configuration

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
rl_config["seed"] = SEED
rl_config["env"] = "InventoryManagement"
rl_config["model"]["use_lstm"] = use_lstm
rl_config["model"]["max_seq_len"] = num_periods
rl_config["model"]["lstm_cell_size"] = 64
rl_config["model"]["lstm_use_prev_action"] = True
rl_config["model"]["lstm_use_prev_reward"] = True

agent = get_trainer(algorithm, rl_config, "InventoryManagement")
#%% RL Training

# Training
iters = 100  # Number of training iterations
validation_interval = 10  # Run validation after how many training iterations
num_validation = 100  # How many validation runs i.e. different realisation of demand
# Create validation demand
valid_demand = test_env.dist.rvs(size=(num_validation, test_env.num_periods), **test_env.dist_param)
results = []
mean_eval_rewards = []
std_eval_rewards = []
eval_episode = []

# Start Training
for i in range(iters):
    res = agent.train()
    results.append(res)
    if (i + 1) % 1 == 0:
        print('\rIter: {}\tReward: {:.2f}'.format(
            i + 1, res['episode_reward_mean']), end='')
    if (i + 1) % validation_interval == 0:
        eval_episode.append(res['episodes_total'])
        list_eval_rewards = []
        for j in range(num_validation):
            demand = valid_demand[j, :]
            obs = test_env.reset(customer_demand=demand)
            episode_reward = 0
            done = False
            if use_lstm:
                reward = 0
                action = np.zeros(num_stages)
                state = agent.get_policy().get_initial_state()
            while not done:
                if use_lstm:
                    action, state, _ = agent.compute_action(obs, state=state, prev_action=action, prev_reward=reward)
                else:
                    action = agent.compute_action(obs)
                obs, reward, done, _ = test_env.step(action)
                episode_reward += reward
            list_eval_rewards.append(episode_reward)
        mean_eval_rewards.append(np.mean(list_eval_rewards))
        std_eval_rewards.append(np.std(list_eval_rewards))

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

dfo_rewards_mean = np.mean(dfo_rewards)
dfo_rewards_std = np.std(dfo_rewards)


fig, ax = plt.subplots()
ax.fill_between(np.arange(len(mean_rewards)),
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 alpha=0.3)
ax.plot(mean_rewards, label='Mean Rewards')
ax.fill_between(np.arange(len(mean_rewards)),
                 np.ones(len(mean_rewards)) * (dfo_rewards_mean - dfo_rewards_std),
                 np.ones(len(mean_rewards)) * (dfo_rewards_mean + dfo_rewards_std),
                 alpha=0.3)
ax.plot(np.arange(len(mean_rewards)), np.ones(len(mean_rewards)) * (dfo_rewards_mean) , label='Mean DFO Rewards')
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

num_tests = 1000
# run until episode ends
episode_reward = 0
done = False
if time_dependency:
    array_obs = np.zeros((num_stages, 4 + np.max(delay), num_periods + 1))
else:
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
test_demand = test_env.dist.rvs(size=(num_tests + 1, test_env.num_periods), **test_env.dist_param)
obs = test_env.reset(customer_demand=test_demand[0, :])
array_obs[:, :, 0] = obs
if use_lstm:
    reward = 0
    action = np.zeros(num_stages)
    state = agent.get_policy().get_initial_state()

while not done:
    if use_lstm:
        action, state, _ = agent.compute_action(obs, state=state, prev_action=action, prev_reward=reward)
    else:
        action = agent.compute_action(obs)
    obs, reward, done, info = test_env.step(action)
    array_obs[:, :, period + 1] = obs
    if standardise_actions:
        array_actions[:, period] = np.round(test_env.rev_scale(action, np.zeros(test_env.num_stages), test_env.order_max,
                                                      test_env.a, test_env.b), 0)
    else:
        array_actions[:, period] = np.round(action, 0)
    array_rewards[period] = reward
    array_profit[:, period] = info['profit']
    array_profit_sum[period] = np.sum(info['profit'])
    array_demand[:, period] = info['demand']
    array_ship[:, period] = info['ship']
    array_acquisition[:, period] = info['acquisition']
    episode_reward += reward
    period += 1

if standardise_state:
    for i in range(num_periods + 1):
        array_obs[:, 0, i] = test_env.rev_scale(array_obs[:, 0, i], np.zeros(test_env.num_stages), test_env.inv_max,
                                                test_env.a, test_env.b)
        array_obs[:, 1, i] = test_env.rev_scale(array_obs[:, 1, i], np.zeros(test_env.num_stages), test_env.inv_max,
                                                test_env.a, test_env.b)
        array_obs[:, 2, i] = test_env.rev_scale(array_obs[:, 2, i], np.zeros(test_env.num_stages), test_env.inv_max,
                                                test_env.a, test_env.b)
        array_obs[:, 3, i] = test_env.rev_scale(array_obs[:, 3, i], np.zeros(test_env.num_stages), test_env.inv_max,
                                                test_env.a, test_env.b)
        if time_dependency:
            array_obs[:, 4:4 + np.max(delay), i] = test_env.rev_scale(array_obs[:, 4:4 + np.max(delay), i],
                                                                   np.zeros((test_env.num_stages, test_env.max_delay)),
                                                                      np.tile(test_env.inv_max.reshape((-1, 1)),
                                                                              (1, test_env.max_delay)),
                                                    test_env.a, test_env.b)

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
'''
# run until episode ends
dfo_episode_reward = 0

dfo_array_obs = np.zeros((num_stages, 4, num_periods + 1))
dfo_array_actions = np.zeros((num_stages, num_periods))
dfo_array_profit = np.zeros((num_stages, num_periods))
dfo_array_profit_sum = np.zeros(num_periods)
dfo_array_demand = np.zeros((num_stages, num_periods))
dfo_array_ship = np.zeros((num_stages, num_periods))
dfo_array_acquisition = np.zeros((num_stages, num_periods))
dfo_array_rewards = np.zeros(num_periods)
period = 0

dfo_obs = DFO_env.reset(customer_demand=test_demand[0, :])
dfo_array_obs[:, :, 0] = dfo_obs
done = False

while not done:
    dfo_action = base_stock_policy(policy, DFO_env)
    dfo_obs, dfo_reward, done, dfo_info = DFO_env.step(dfo_action)
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
'''
#%% Test runs on final agent
list_test_rewards = []
stage_rewards = np.zeros((num_stages, num_tests))
for i in range(num_tests):
    demand = test_demand[i + 1, :]
    obs = test_env.reset(customer_demand=demand)
    episode_reward = 0
    done = False
    if use_lstm:
        reward = 0
        action = np.zeros(num_stages)
        state = agent.get_policy().get_initial_state()
    while not done:
        if use_lstm:
            action, state, _ = agent.compute_action(obs, state=state, prev_action=action, prev_reward=reward)
        else:
            action = agent.compute_action(obs)
        obs, reward, done, info = test_env.step(action)
        for m in range(num_stages):
            stage_rewards[m, i] += info["profit"][m]
        episode_reward += reward
    list_test_rewards.append(episode_reward)

test_reward_mean = np.mean(list_test_rewards)
test_reward_std = np.std(list_test_rewards)
stage_test_reward_mean = np.mean(stage_rewards, axis=1)
stage_test_reward_std = np.std(stage_rewards, axis=1)

print(f"\nThe mean DFO reward is {dfo_rewards_mean} with standard deviation {dfo_rewards_std}")
print(f"\nOn {num_tests} runs, the mean reward is: {test_reward_mean}, with standard deviation {test_reward_std}")
for m in range(num_stages):
    print(f"\nFor stage {m}, the mean reward is: {stage_test_reward_mean[m]}, "
          f"with standard deviation {stage_test_reward_std[m]}")