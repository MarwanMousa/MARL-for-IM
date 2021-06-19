from environments.IM_env import InvManagement
import ray
from ray.rllib import agents
from ray import tune
import numpy as np
import matplotlib.pyplot as plt

#%% Environment and Agent Configuration

# Environment creator function for environment registration
def env_creator(configuration):
    env = InvManagement(configuration)
    return env


# Environment Configuration
num_stages = 3
num_periods = 50
customer_demand = np.ones(num_periods) * 5
init_inv = np.ones(num_stages)*50
price = [3.5, 3, 2, 1]
stock_cost = [0.1, 0.3, 0.3]
backlog_cost = [0.3, 0.5, 0.5]

# Agent/Policy ids of the 3-stage and 4-stage configurations


env_name = "InventoryManagement"
tune.register_env(env_name, env_creator)

env_config = {
    "num_stages": num_stages,
    "num_periods": num_periods,
    "customer_demand": customer_demand,
    "init_inv": init_inv,
    "price": price,
    "stock_cost": stock_cost,
    "backlog_cost": backlog_cost,
    "independent": False
}
CONFIG = env_config.copy()

# Test environment
test_env = InvManagement(env_config)


# Training Set-up
ray.init(ignore_reinit_error=True, local_mode=True)
rl_config = agents.ppo.DEFAULT_CONFIG.copy()
rl_config["num_workers"] = 3
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
fig, axs = plt.subplots(3, num_stages, figsize=(20, 8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs = axs.ravel()

for i in range(num_stages):
    axs[i].plot(array_obs[i, 0, :], label='Inventory')
    axs[i].plot(array_obs[i, 1, :], label='Backlog')
    axs[i].plot(np.arange(1, 51), array_obs[i, 2, 1:], label='Unfulfilled orders')
    axs[i].plot(np.arange(1, 51), array_actions[i, :], label='Replenishment Order', color='k', alpha=0.5)
    axs[i].legend()
    title = 'Stage ' + str(i)
    axs[i].set_title(title)
    axs[i].set_ylabel('Products')
    axs[i].set_xlim(0, num_periods)

    axs[i+num_stages].plot(np.arange(1, 51), array_demand[i, :], label='demand')
    axs[i+num_stages].plot(np.arange(1, 51), array_ship[i, :], label='shipment')
    axs[i+num_stages].plot(np.arange(1, 51), array_acquisition[i, :], label='Acquisition')
    axs[i+num_stages].legend()
    axs[i+num_stages].set_ylabel('Products')
    axs[i+num_stages].set_xlim(0, num_periods)

    axs[i+num_stages*2].plot(np.arange(1, 51), array_profit[i, :], label='profit')
    axs[i+num_stages*2].plot([0, num_periods], [0, 0], color='k')
    axs[i+num_stages*2].set_xlabel('Period')
    axs[i+num_stages*2].set_ylabel('Profit')
    axs[i+num_stages*2].set_xlim(0, num_periods)


plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='w', edgecolor='k')
ax.plot(np.arange(1, 51), array_rewards)
ax.plot([0, num_periods], [0, 0], color='k')
ax.set_title('Aggregate Rewards')
ax.set_xlabel('Period')
ax.set_ylabel('Rewards/profit')
ax.set_xlim(0, num_periods)

plt.show()