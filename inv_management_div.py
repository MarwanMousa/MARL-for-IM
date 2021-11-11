from environments.IM_div_env import InvManagementDiv
import ray
from ray import tune
import numpy as np
import json
from utils import get_config, get_trainer, ensure_dir, check_connections, create_network
from base_restock_policy import optimize_inventory_policy, dfo_func, base_stock_policy
from models.RNN_Model import RNNModel
from ray.rllib.models import ModelCatalog
from hyperparams import get_hyperparams
import time
import matplotlib.pyplot as plt
from matplotlib import rc

#%% Environment Configuration

train_agent = False
save_agent = True
save_path = "checkpoints/single_agent/div_1"
load_path = "checkpoints/single_agent/div_1"
LP_load_path = "LP_results/div_1/"
load_iteration = str(500)
load_agent_path = load_path + '/checkpoint_000' + load_iteration + '/checkpoint-' + load_iteration

# Define plot settings
rc('font', **{'family': 'serif', 'serif': ['Palatino'], 'size': 13})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["figure.dpi"] = 200

# Set script seed
SEED = 52
np.random.seed(seed=SEED)

# Environment Configuration
num_nodes = 4
connections = {
    0: [1],
    1: [2, 3],
    2: [],
    3: [],
    }
check_connections(connections)
network = create_network(connections)
num_periods = 30
customer_demand = np.ones(num_periods) * 5
mu = 5
lower_upper = (1, 5)
init_inv = np.ones(num_nodes)*10
inv_target = np.ones(num_nodes) * 0
inv_max = np.ones(num_nodes) * 30
stock_cost = np.array([0.35, 0.3, 0.4, 0.4])
backlog_cost = np.array([0.5, 0.7, 0.6, 0.6])
delay = np.array([1, 2, 1, 1], dtype=np.int8)
time_dependency = False
use_lstm = False
prev_actions = True
prev_demand = True
prev_length = 1

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


# Environment creator function for environment registration
def env_creator(configuration):
    env = InvManagementDiv(configuration)
    return env


env_name = "InventoryManagementDiv"
tune.register_env(env_name, env_creator)

env_config = {
    "num_nodes": num_nodes,
    "connections": connections,
    "num_periods": num_periods,
    "init_inv": init_inv,
    "stock_cost": stock_cost,
    "backlog_cost": backlog_cost,
    "demand_dist": demand_distribution,
    "inv_target": inv_target,
    "inv_max": inv_max,
    "seed": SEED,
    "delay": delay,
    parameter: parameter_value,
    "time_dependency": time_dependency,
    "prev_demand": prev_demand,
    "prev_actions": prev_actions,
    "prev_length": prev_length,
}

# Loading in hyperparameters from hyperparameter search
use_optimal = True
configuration_name = "S_3"

if use_optimal:
    o_config = get_hyperparams(configuration_name)
    temp_env = o_config["env_config"]
    time_dependency = temp_env["time_dependency"]
    prev_demand = temp_env["prev_demand"]
    prev_actions = temp_env["prev_actions"]
    prev_length = temp_env["prev_length"]
    env_config["time_dependency"] = time_dependency
    env_config["prev_demand"] = prev_demand
    env_config["prev_actions"] = prev_actions
    env_config["prev_length"] = prev_length

CONFIG = env_config.copy()
# Configuration for the base-restock policy with DFO that has no state-action standardisation
DFO_CONFIG = env_config.copy()
DFO_CONFIG["standardise_state"] = False
DFO_CONFIG["standardise_actions"] = False
DFO_CONFIG["time_dependency"] = False
DFO_CONFIG["prev_actions"] = False
DFO_CONFIG["prev_demand"] = False

# Test environment
test_env = InvManagementDiv(env_config)
DFO_env = InvManagementDiv(DFO_CONFIG)

ModelCatalog.register_custom_model(
        "rnn_model", RNNModel)

#%% Agent Configuration

# Algorithm used
algorithm = 'ppo'
# Training Set-up
ray.init(ignore_reinit_error=True, local_mode=True, num_cpus=1)
rl_config = get_config(algorithm, num_periods, SEED)
rl_config["env_config"] = CONFIG
rl_config["env"] = "InventoryManagementDiv"
if use_lstm:
    rl_config["model"]["custom_model"] = "rnn_model"
    rl_config["model"]["fcnet_hiddens"] = [128, 128]
    rl_config["model"]["max_seq_len"] = num_periods
    rl_config["model"]["custom_model_config"] = {"fc_size": 64,
                                                 "use_initial_fc": True,
                                                 "lstm_state_size": 128}

if use_optimal:
    rl_config = o_config
    rl_config["env_config"] = CONFIG
#%% RL Training

# Get trainer
agent = get_trainer(algorithm, rl_config, "InventoryManagementDiv")

if train_agent:
    # Training
    iters = 500  # Number of training iterations
    min_iter_save = 300
    checkpoint_interval = 20
    results = []

    # Start Training
    for i in range(iters):
        res = agent.train()
        results.append(res)
        if (i + 1) % 1 == 0:
            print('\rIter: {}\tReward: {:.2f}'.format(
                i + 1, res['episode_reward_mean']), end='')

        if (i+1) % checkpoint_interval == 0 and i >= min_iter_save and save_agent:
            ensure_dir(save_path)
            agent.save(save_path)
else:
    agent.restore(load_agent_path)
    results_load_path = load_path + '/results.npy'
    results = np.load(results_load_path, allow_pickle=True)

ray.shutdown()


if save_agent:
    json_env_config = save_path + '/env_config.json'
    ensure_dir(json_env_config)
    with open(json_env_config, 'w') as fp:
        for key, value in CONFIG.items():
            if isinstance(value, np.ndarray):
                CONFIG[key] = CONFIG[key].tolist()
        json.dump(CONFIG, fp)
    results_save_path = save_path + '/results.npy'
    np.save(results_save_path, results)

#%% Get Test demand
num_tests = 1000
test_seed = 420
np.random.seed(seed=test_seed)
test_demand = test_env.dist.rvs(size=(num_tests, (len(test_env.retailers)), test_env.num_periods), **test_env.dist_param)
noisy_demand = False
noise_threshold = 30/100
noisy_delay = False
noisy_delay_threshold = 30/100
if noisy_demand:
    for i in range(num_tests):
        for j in range(num_periods):
            double_demand = np.random.uniform(0, 1)
            zero_demand = np.random.uniform(0, 1)
            if double_demand <= noise_threshold:
                test_demand[i, j] = 2 *test_demand[i, j]
            if zero_demand <= noise_threshold:
                test_demand[i, j] = 0

#%% Derivative Free Optimization
init_policy = np.ones(num_nodes)*25
policy, out = optimize_inventory_policy(DFO_env, dfo_func, init_policy=init_policy, demand=test_demand[0, :])
print("Re-order levels: {}".format(policy))
print("DFO Info:\n{}".format(out))

dfo_rewards = []
for i in range(num_tests):
    demand = test_demand[i, :]
    DFO_env.reset(customer_demand=demand, noisy_delay=noisy_delay, noisy_delay_threshold=noisy_delay_threshold)
    dfo_reward = 0
    done = False
    while not done:
        dfo_action = base_stock_policy(policy, DFO_env)
        s, r, done, _ = DFO_env.step(dfo_action)
        dfo_reward += r
    dfo_rewards.append(dfo_reward)

print(f'Mean DFO rewards is {np.mean(dfo_rewards)}')
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
oracle_rewards_mean = np.load(LP_load_path + 'Oracle/reward_mean.npy')
oracle_rewards_std = np.load(LP_load_path + 'Oracle/reward_std.npy')
SHLP_rewards_mean = np.load(LP_load_path + 'SHLP/reward_mean.npy')
SHLP_rewards_std = np.load(LP_load_path + 'SHLP/reward_std.npy')

fig, ax = plt.subplots()
# Plot rewards
ax.fill_between(np.arange(len(mean_rewards)),
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 alpha=0.3)
ax.plot(mean_rewards, label='Mean RL rewards')

# Plot DFO rewards
ax.fill_between(np.arange(len(mean_rewards)),
                 np.ones(len(mean_rewards)) * (dfo_rewards_mean - dfo_rewards_std),
                 np.ones(len(mean_rewards)) * (dfo_rewards_mean + dfo_rewards_std),
                 alpha=0.3)
ax.plot(np.arange(len(mean_rewards)), np.ones(len(mean_rewards)) * (dfo_rewards_mean), label='Mean DFO rewards')

# Plot Oracle rewards
ax.fill_between(np.arange(len(mean_rewards)),
                 np.ones(len(mean_rewards)) * (oracle_rewards_mean - oracle_rewards_std),
                 np.ones(len(mean_rewards)) * (oracle_rewards_mean + oracle_rewards_std),
                 alpha=0.3)
ax.plot(np.arange(len(mean_rewards)), np.ones(len(mean_rewards)) * (oracle_rewards_mean), label='Mean Oracle rewards')

# Plot SHLP rewards
ax.fill_between(np.arange(len(mean_rewards)),
                 np.ones(len(mean_rewards)) * (SHLP_rewards_mean - SHLP_rewards_std),
                 np.ones(len(mean_rewards)) * (SHLP_rewards_mean + SHLP_rewards_std),
                 alpha=0.3)
ax.plot(np.arange(len(mean_rewards)), np.ones(len(mean_rewards)) * (SHLP_rewards_mean), label='Mean SHLP rewards')


ax.set_ylabel('Rewards')
ax.set_xlabel('Episode')
ax.legend()

if save_agent:
    rewards_name = save_path + '/training_rewards.png'
    plt.savefig(rewards_name, dpi=200)

plt.show()

#%% Test rollout

# run until episode ends
episode_reward = 0
done = False
if time_dependency and not prev_demand and not prev_actions:
    array_obs = np.zeros((num_nodes, 3 + np.max(delay), num_periods + 1))
elif time_dependency and not prev_demand and prev_actions:
    array_obs = np.zeros((num_nodes, 3 + np.max(delay) + prev_length, num_periods + 1))
elif time_dependency and prev_demand and not prev_actions:
    array_obs = np.zeros((num_nodes, 3 + np.max(delay) + prev_length, num_periods + 1))
elif time_dependency and prev_demand and prev_actions:
    array_obs = np.zeros((num_nodes, 3 + np.max(delay) + prev_length*2, num_periods + 1))
elif not time_dependency and prev_demand and prev_actions:
    array_obs = np.zeros((num_nodes, 3 + prev_length*2, num_periods + 1))
elif not time_dependency and prev_demand and not prev_actions:
    array_obs = np.zeros((num_nodes, 3 + prev_length, num_periods + 1))
elif not time_dependency and not prev_demand and prev_actions:
    array_obs = np.zeros((num_nodes, 3 + prev_length, num_periods + 1))
else:
    array_obs = np.zeros((num_nodes, 3, num_periods + 1))

array_actions = np.zeros((num_nodes, num_periods))
array_profit = np.zeros((num_nodes, num_periods))
array_profit_sum = np.zeros(num_periods)
array_demand = np.zeros((num_nodes, num_periods))
array_ship = np.zeros((num_nodes, num_periods))
array_acquisition = np.zeros((num_nodes, num_periods))
array_rewards = np.zeros(num_periods)
period = 0


obs = test_env.reset(customer_demand=test_demand[0, :], noisy_delay=noisy_delay, noisy_delay_threshold=noisy_delay_threshold)
array_obs[:, :, 0] = obs
if use_lstm:
    reward = 0
    action = np.zeros(num_nodes)
    state = agent.get_policy().get_initial_state()

while not done:
    if use_lstm:
        action, state, _ = agent.compute_single_action(obs, state=state, prev_action=action, prev_reward=reward)
    else:
        action = agent.compute_single_action(obs)
    obs, reward, done, info = test_env.step(action)
    array_obs[:, :, period + 1] = obs
    array_actions[:, period] = np.round(test_env.rev_scale(action, np.zeros(test_env.num_nodes), test_env.order_max,
                                        test_env.a, test_env.b), 0)
    array_actions[:, period] = np.maximum(array_actions[:, period], np.zeros(num_nodes))

    array_rewards[period] = reward
    array_profit[:, period] = info['profit']
    array_profit_sum[period] = np.sum(info['profit'])
    array_demand[:, period] = info['demand']
    array_ship[:, period] = info['ship']
    array_acquisition[:, period] = info['acquisition']
    episode_reward += reward
    period += 1

#%% Rescaling

for i in range(num_periods + 1):
    array_obs[:, 0, i] = test_env.rev_scale(array_obs[:, 0, i], np.zeros(test_env.num_nodes), test_env.inv_max,
                                            test_env.a, test_env.b)
    array_obs[:, 1, i] = test_env.rev_scale(array_obs[:, 1, i], np.zeros(test_env.num_nodes), test_env.inv_max,
                                            test_env.a, test_env.b)
    array_obs[:, 2, i] = test_env.rev_scale(array_obs[:, 2, i], np.zeros(test_env.num_nodes), test_env.inv_max,
                                            test_env.a, test_env.b)
    if time_dependency and not prev_demand and not prev_actions:
        array_obs[:, 3:3 + np.max(delay), i] = test_env.rev_scale(array_obs[:, 3:3 + np.max(delay), i],
                                                                  np.zeros((test_env.num_nodes, test_env.max_delay)),
                                                                  np.tile(test_env.inv_max.reshape((-1, 1)),
                                                                          (1, test_env.max_delay)),
                                                                  test_env.a, test_env.b)
    elif time_dependency and not prev_demand and prev_actions:
        array_obs[:, 3:3 + prev_length, i] = test_env.rev_scale(array_obs[:, 3:3 + prev_length, i],
                                                                  np.zeros(
                                                                      (test_env.num_nodes, test_env.prev_length)),
                                                                  np.tile(test_env.order_max.reshape((-1, 1)),
                                                                          (1, test_env.prev_length)),
                                                                  test_env.a, test_env.b)
        array_obs[:, 3+prev_length:3+prev_length+np.max(delay), i] = test_env.rev_scale(array_obs[:, 3+prev_length:3+prev_length+np.max(delay) + np.max(delay), i],
                                                                  np.zeros(
                                                                      (test_env.num_nodes, test_env.max_delay)),
                                                                  np.tile(test_env.inv_max.reshape((-1, 1)),
                                                                          (1, test_env.max_delay)),
                                                                  test_env.a, test_env.b)
    elif time_dependency and prev_demand and not prev_actions:
        array_obs[:, 3:3 + prev_length, i] = test_env.rev_scale(array_obs[:, 3:3 + prev_length, i],
                                                                np.zeros(
                                                                    (test_env.num_nodes, test_env.prev_length)),
                                                                np.tile(test_env.demand_max.reshape((-1, 1)),
                                                                        (1, test_env.prev_length)),
                                                                test_env.a, test_env.b)
        array_obs[:, 3 + prev_length:3 + prev_length + np.max(delay), i] = test_env.rev_scale(
            array_obs[:, 3 + prev_length:3 + prev_length + np.max(delay) + np.max(delay), i],
            np.zeros(
                (test_env.num_nodes, test_env.max_delay)),
            np.tile(test_env.inv_max.reshape((-1, 1)),
                    (1, test_env.max_delay)),
            test_env.a, test_env.b)
    elif time_dependency and prev_demand and prev_actions:
        array_obs[:, 3:3 + prev_length, i] = test_env.rev_scale(array_obs[:, 3:3 + prev_length, i],
                                                                np.zeros(
                                                                    (test_env.num_nodes, test_env.prev_length)),
                                                                np.tile(test_env.demand_max.reshape((-1, 1)),
                                                                        (1, test_env.prev_length)),
                                                                test_env.a, test_env.b)

        array_obs[:, 3 + prev_length:3 + prev_length * 2, i] = test_env.rev_scale(
            array_obs[:, 3 + prev_length:3 + prev_length * 2, i],
            np.zeros(
                (test_env.num_nodes, test_env.prev_length)),
            np.tile(test_env.inv_max.reshape((-1, 1)),
                    (1, test_env.prev_length)),
            test_env.a, test_env.b)

        array_obs[:, 3 + prev_length*2:3 + prev_length*2 + np.max(delay), i] = test_env.rev_scale(
            array_obs[:, 3 + prev_length*2:3 + prev_length*2 + np.max(delay) + np.max(delay), i],
            np.zeros(
                (test_env.num_nodes, test_env.max_delay)),
            np.tile(test_env.inv_max.reshape((-1, 1)),
                    (1, test_env.max_delay)),
            test_env.a, test_env.b)
    elif not time_dependency and prev_demand and not prev_actions:
        array_obs[:, 3:3 + prev_length, i] = test_env.rev_scale(array_obs[:, 3:3 + prev_length, i],
                                                                np.zeros(
                                                                    (test_env.num_nodes, test_env.prev_length)),
                                                                np.tile(test_env.demand_max.reshape((-1, 1)),
                                                                        (1, test_env.prev_length)),
                                                                test_env.a, test_env.b)

    elif not time_dependency and prev_demand and prev_actions:
        array_obs[:, 3:3 + prev_length, i] = test_env.rev_scale(array_obs[:, 3:3 + prev_length, i],
                                                                np.zeros(
                                                                    (test_env.num_nodes, test_env.prev_length)),
                                                                np.tile(test_env.demand_max.reshape((-1, 1)),
                                                                        (1, test_env.prev_length)),
                                                                test_env.a, test_env.b)

        array_obs[:, 3 + prev_length:3 + prev_length * 2, i] = test_env.rev_scale(
            array_obs[:, 3 + prev_length:3 + prev_length * 2, i],
            np.zeros(
                (test_env.num_nodes, test_env.prev_length)),
            np.tile(test_env.inv_max.reshape((-1, 1)),
                    (1, test_env.prev_length)),
            test_env.a, test_env.b)


#%% Plots
fig, axs = plt.subplots(3, num_nodes, figsize=(18, 9), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.06, wspace=.16)

axs = axs.ravel()

for i in range(num_nodes):
    axs[i].plot(array_obs[i, 0, :], label='Inventory', lw=2)
    axs[i].plot(array_obs[i, 1, :], label='Backlog', color='tab:red', lw=2)
    title = 'Node ' + str(i)
    axs[i].set_title(title)
    axs[i].set_xlim(0, num_periods)
    axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if i == 0:
        axs[i].legend()
        axs[i].set_ylabel('Products')

    axs[i+num_nodes].plot(array_actions[i, :], label='Replenishment order', color='k', lw=2)
    axs[i+num_nodes].plot(array_demand[i, :], label='Demand', color='tab:orange', lw=2)
    axs[i+num_nodes].set_xlim(0, num_periods)
    axs[i+num_nodes].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if i == 0:
        axs[i+num_nodes].legend()
        axs[i + num_nodes].set_ylabel('Products')

    axs[i+num_nodes*2].plot(array_profit[i, :], label='Periodic profit', color='tab:green', lw=2)
    axs[i+num_nodes*2].plot(np.cumsum(array_profit[i, :]), label='Cumulative profit', color='salmon', lw=2)
    axs[i+num_nodes*2].plot([0, num_periods], [0, 0], color='k')
    axs[i+num_nodes*2].set_xlabel('Period')
    axs[i+num_nodes*2].set_xlim(0, num_periods)
    if i == 0:
        axs[i + num_nodes * 2].legend()
        axs[i + num_nodes * 2].set_ylabel('Profit')

if save_agent:
    test_name = save_path + '/test_rollout.png'
    plt.savefig(test_name, dpi=200)
plt.show()

#%% DFO Test run

# run until episode ends
dfo_episode_reward = 0

dfo_array_obs = np.zeros((num_nodes, 3, num_periods + 1))
dfo_array_actions = np.zeros((num_nodes, num_periods))
dfo_array_profit = np.zeros((num_nodes, num_periods))
dfo_array_profit_sum = np.zeros(num_periods)
dfo_array_demand = np.zeros((num_nodes, num_periods))
dfo_array_ship = np.zeros((num_nodes, num_periods))
dfo_array_acquisition = np.zeros((num_nodes, num_periods))
dfo_array_rewards = np.zeros(num_periods)
period = 0

dfo_obs = DFO_env.reset(customer_demand=test_demand[0, :], noisy_delay=noisy_delay, noisy_delay_threshold=noisy_delay_threshold)
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
fig, axs = plt.subplots(3, num_nodes, figsize=(18, 9), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.06, wspace=.16)

axs = axs.ravel()

for i in range(num_nodes):
    axs[i].plot(dfo_array_obs[i, 0, :], label='Inventory', lw=2)
    axs[i].plot(dfo_array_obs[i, 1, :], label='Backlog', color='tab:red', lw=2)
    title = 'DFO Node ' + str(i)
    axs[i].set_title(title)
    axs[i].set_xlim(0, num_periods)
    axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if i == 0:
        axs[i].legend()
        axs[i].set_ylabel('Products')

    axs[i+num_nodes].plot(dfo_array_actions[i, :], label='Replenishment order', color='k', lw=2)
    axs[i+num_nodes].plot(dfo_array_demand[i, :], label='Demand', color='tab:orange', lw=2)
    axs[i+num_nodes].set_xlim(0, num_periods)
    axs[i+num_nodes].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if i == 0:
        axs[i + num_nodes].legend()
        axs[i + num_nodes].set_ylabel('Products')

    axs[i+num_nodes*2].plot(dfo_array_profit[i, :], label='Periodic profit', color='tab:green', lw=2)
    axs[i+num_nodes*2].plot(np.cumsum(dfo_array_profit[i, :]), label='Cumulative profit', color='salmon', lw=2)
    axs[i+num_nodes*2].plot([0, num_periods], [0, 0], color='k')
    axs[i+num_nodes*2].set_xlabel('Period')
    axs[i+num_nodes*2].set_xlim(0, num_periods)
    if i == 0:
        axs[i + num_nodes * 2].legend()
        axs[i + num_nodes * 2].set_ylabel('Profit')

if save_agent:
    test_name = save_path + '/dfo_test_rollout.png'
    plt.savefig(test_name, dpi=200)
plt.show()

#%% Test runs on final agent
reward_list = []
inventory_list = []
backlog_list = []
customer_backlog_list = []
stage_rewards = np.zeros((num_nodes, num_tests))
profit = np.zeros((num_tests, num_periods))

start_time = time.time()
for i in range(num_tests):
    demand = test_demand[i, :]
    obs = test_env.reset(customer_demand=demand, noisy_delay=noisy_delay, noisy_delay_threshold=noisy_delay_threshold)
    episode_reward = 0
    done = False
    t = 0
    total_inventory = 0
    total_backlog = 0
    customer_backlog = 0
    if use_lstm:
        reward = 0
        action = np.zeros(num_nodes)
        state = agent.get_policy().get_initial_state()
    while not done:
        if use_lstm:
            action, state, _ = agent.compute_single_action(obs, state=state, prev_action=action, prev_reward=reward)
        else:
            action = agent.compute_single_action(obs)

        obs, reward, done, info = test_env.step(action)
        profit[i, t] = reward
        # Get re-scaled inv and backlog
        inv = test_env.rev_scale(obs[:, 0], np.zeros(test_env.num_nodes), test_env.inv_max, test_env.a, test_env.b)
        bl = test_env.rev_scale(obs[:, 1], np.zeros(test_env.num_nodes), test_env.inv_max, test_env.a, test_env.b)

        total_inventory += sum(inv)
        total_backlog += sum(bl)
        customer_backlog += bl[0]

        for m in range(num_nodes):
            stage_rewards[m, i] += info["profit"][m]

        episode_reward += reward

        t += 1

    reward_list.append(episode_reward)
    inventory_list.append(total_inventory)
    backlog_list.append(total_backlog)
    customer_backlog_list.append(customer_backlog)

single_time = time.time() - start_time

stage_test_reward_mean = np.mean(stage_rewards, axis=1)
stage_test_reward_std = np.std(stage_rewards, axis=1)

single_reward_mean = np.mean(reward_list)
single_reward_std = np.std(reward_list)
inventory_level_mean = np.mean(inventory_list)
inventory_level_std = np.std(inventory_list)
backlog_level_mean = np.mean(backlog_list)
backlog_level_std = np.std(backlog_list)
customer_backlog_mean = np.mean(customer_backlog_list)
customer_backlog_std = np.std(customer_backlog_list)

print(f"\nThe mean DFO reward is {dfo_rewards_mean} with standard deviation {dfo_rewards_std}")
print(f"\nOn {num_tests} runs, the mean reward is: {single_reward_mean}, with standard deviation {single_reward_std}")
print(f'Mean inventory level is: {inventory_level_mean} with std: {inventory_level_std}')
print(f'Mean backlog level is: {backlog_level_mean} with std: {backlog_level_std}')
print(f'Mean customer backlog level is: {customer_backlog_mean } with std: {customer_backlog_std}')
print(f'Took {single_time}s for {num_tests} inference')
for m in range(num_nodes):
    print(f"\nFor stage {m}, the mean reward is: {stage_test_reward_mean[m]}, "
          f"with standard deviation {stage_test_reward_std[m]}")

if save_agent:
    np.save(save_path+'/dfo_mean.npy', dfo_rewards_mean)
    np.save(save_path+'/dfo_std.npy', dfo_rewards_std)
    np.save(save_path+'/reward_mean.npy', single_reward_mean)
    np.save(save_path+'/reward_std.npy', single_reward_std)
    np.save(save_path+'/inventory_mean.npy', inventory_level_mean)
    np.save(save_path+'/inventory_std.npy', inventory_level_std)
    np.save(save_path+'/backlog_mean.npy', backlog_level_mean)
    np.save(save_path+'/backlog_std.npy', backlog_level_std)
    np.save(save_path+'/customer_backlog_mean', customer_backlog_mean)
    np.save(save_path+'/customer_backlog_std', customer_backlog_std)
    np.save(save_path+'/profit', profit)
    np.save(save_path+'/time', single_time)