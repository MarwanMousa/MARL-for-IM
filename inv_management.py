from environments.IM_env import InvManagement
import ray
from ray.rllib import agents
from ray import tune
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import os

#%% Environment and Agent Configuration

# Environment creator function for environment registration
def env_creator(configuration):
    env = InvManagement(configuration)
    return env


# Environment Configuration
num_stages = 3
num_periods = 50
customer_demand = np.ones(num_periods) * 5
mu = 5
lower_upper = (1, 5)
init_inv = np.ones(num_stages)*20
inv_target = np.ones(num_stages) * 5
inv_max = np.ones(num_stages) * 100
price = np.array([3.5, 3, 2, 1])
stock_cost = np.array([0.1, 0.2, 0.3])
backlog_cost = np.array([0.2, 0.7, 0.5])

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
    parameter: parameter_value
}
CONFIG = env_config.copy()

# Test environment
test_env = InvManagement(env_config)


# Training Set-up
ray.init(ignore_reinit_error=True, local_mode=True)
rl_config = agents.ppo.DEFAULT_CONFIG.copy()
rl_config["num_workers"] = 4
rl_config["normalize_actions"] = False
rl_config["env_config"] = CONFIG
rl_config["framework"] = 'torch'
rl_config["model"] = {
        "vf_share_layers": False,
        "fcnet_activation": 'relu',
        "fcnet_hiddens": [256, 256]
    }
rl_config["lr"] = 1e-5
rl_config["seed"] = 52
rl_config['vf_clip_param'] = 10_000
agent = agents.ppo.PPOTrainer(config=rl_config, env=InvManagement)

#%% Training

# Training
iters = 200
results = []
for i in range(iters):
    res = agent.train()
    results.append(res)
    if (i + 1) % 5 == 0:
        #chkpt_file = agent.save('/Users/marwanmousa/University/MSc_AI/Individual_Project/MARL-and-DMPC-for-OR/checkpoints/multi_agent')
        print('\rIter: {}\tReward: {:.2f}'.format(
            i + 1, res['episode_reward_mean']), end='')

ray.shutdown()

training_time = datetime.datetime.now().strftime("D%dM%m_h%Hm%M")
save_path = '/Users/marwanmousa/University/MSc_AI/Individual_Project/MARL-and-DMPC-for-OR/figures/single_agent/' + training_time
json_config = save_path + '/env_config.json'
os.makedirs(os.path.dirname(json_config), exist_ok=True)
with open(json_config, 'w') as fp:
    for key, value in CONFIG.items():
        if isinstance(value, np.ndarray):
            CONFIG[key] = CONFIG[key].tolist()
    json.dump(CONFIG, fp)
#%% Reward Plots
p = 200
# Unpack values from each iteration
rewards = np.hstack([i['hist_stats']['episode_reward']
                     for i in results])

mean_rewards = np.array([np.mean(rewards[i - p:i + 1])
                         if i >= p else np.mean(rewards[:i + 1])
                         for i, _ in enumerate(rewards)])

std_rewards = np.array([np.std(rewards[i - p:i + 1])
                        if i >= p else np.std(rewards[:i + 1])
                        for i, _ in enumerate(rewards)])


fig, ax = plt.subplots()
ax.fill_between(np.arange(len(mean_rewards)),
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 label='Standard Deviation', alpha=0.3)
ax.plot(mean_rewards, label='Mean Rewards')
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
obs = test_env.reset()
array_obs = np.zeros((num_stages, 3, num_periods + 1))
array_actions = np.zeros((num_stages, num_periods))
array_profit = np.zeros((num_stages, num_periods))
array_demand = np.zeros((num_stages, num_periods))
array_ship = np.zeros((num_stages, num_periods))
array_acquisition = np.zeros((num_stages, num_periods))
array_rewards = np.zeros(num_periods)
period = 0

array_obs[:, :, 0] = obs

while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = test_env.step(action)
    array_obs[:, :, period + 1] = obs
    array_actions[:, period] = action
    array_rewards[period] = reward
    array_profit[:, period] = info['profit']
    array_demand[:, period] = info['demand']
    array_ship[:, period] = info['ship']
    array_acquisition[:, period] = info['acquisition']
    array_rewards[period] = reward
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
    axs[i].plot([0, num_periods + 1], [inv_target[i], inv_target[i]], label='Target Inventory', color='k')
    axs[i].legend()
    title = 'Stage ' + str(i)
    axs[i].set_title(title)
    axs[i].set_ylabel('Products')
    axs[i].set_xlim(0, num_periods)

    axs[i+num_stages].plot(np.arange(1, num_periods+1), array_acquisition[i, :], label='Acquisition')
    axs[i+num_stages].plot(np.arange(0, num_periods), array_actions[i, :], label='Replenishment Order', color='k')
    axs[i+num_stages].legend()
    axs[i+num_stages].set_ylabel('Products')
    axs[i+num_stages].set_xlim(0, num_periods)

    axs[i+num_stages*2].plot(np.arange(1, num_periods+1), array_demand[i, :], label='demand')
    axs[i+num_stages*2].plot(np.arange(1, num_periods+1), array_ship[i, :], label='shipment')
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
ax.plot(np.arange(1, num_periods+1), array_rewards, label='periodic profit')
ax.plot(np.arange(1, num_periods+1), np.cumsum(array_rewards), label='cumulative profit')
ax.plot([0, num_periods], [0, 0], color='k')
ax.set_title('Aggregate Rewards')
ax.set_xlabel('Period')
ax.set_ylabel('Rewards/profit')
ax.legend()
ax.set_xlim(0, num_periods)

test_rewards_name = save_path + '/test_rollout_rewards.png'
plt.savefig(test_rewards_name, dpi=200)
plt.show()
