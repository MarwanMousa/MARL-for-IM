from environments.MAIM_env import MultiAgentInvManagement
import ray
from ray import tune
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import json
from utils import get_config, get_trainer

#%% Environment and Agent Configuration

# Set script seed
SEED = 52
np.random.seed(seed=SEED)

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
backlog_cost = np.array([0.6, 0.6, 0.6])
delay = np.array([1, 1, 1], dtype=np.int8)
independent = True
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
    "independent": independent,
    "seed": 52,
    parameter: parameter_value,
    "standardise_state": standardise_state,
    "standardise_actions": standardise_actions,
    "a": a,
    "b": b,
    "time_dependency": time_dependency,
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
    for i in range(num_stages):
        if agent_id.startswith(agent_ids[i]):
            return agent_ids[i]

# Algorithm used
algorithm = 'ppo'
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
rl_config["batch_mode"] = "complete_episodes"
rl_config["model"]["use_lstm"] = use_lstm
rl_config["model"]["max_seq_len"] = num_periods
rl_config["model"]["lstm_use_prev_action"] = False
rl_config["model"]["lstm_use_prev_reward"] = False


agent = get_trainer(algorithm, rl_config, "MultiAgentInventoryManagement")

#%% Training

# Training
iters = 100
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
        #chkpt_file = agent.save('/Users/marwanmousa/University/MSc_AI/Individual_Project/MARL-and-DMPC-for-OR/checkpoints/multi_agent')
        print('\rIter: {}\tReward: {:.2f}'.format(
            i + 1, res['episode_reward_mean']), end='')

    if (i + 1) % validation_interval == 0:
        eval_episode.append(res['episodes_total'])
        reward_array = np.zeros((num_stages, num_validation))
        for j in range(num_validation):
            demand = valid_demand[j]
            obs = test_env.reset(customer_demand=demand)
            done = False
            if use_lstm:
                reward = {}
                action = {}
                state = {}
                for m in range(num_stages):
                    sp = agent_ids[m]
                    reward[sp] = 0
                    action[sp] = 0
                    state[sp] = agent.get_policy(sp).get_initial_state()
            while not done:
                if use_lstm:
                    for m in range(num_stages):
                        sp = agent_ids[m]
                        action[sp], state[sp], _ = agent.compute_action(obs[sp], state=state[sp],
                                                                        prev_action=action[sp], prev_reward=reward[sp],
                                                                        policy_id=sp)
                else:
                    action = {}
                    for m in range(num_stages):
                        sp = agent_ids[m]
                        action[sp] = agent.compute_action(obs[sp], policy_id=sp)

                obs, reward, dones, info = test_env.step(action)
                done = dones['__all__']
                for m in range(num_stages):
                    reward_array[m, j] += reward[agent_ids[m]]

        mean_eval_rewards[:, eval_num] = np.mean(reward_array, axis=1)
        std_eval_rewards[:, eval_num] = np.std(reward_array, axis=1)
        eval_num += 1

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
ax.plot(eval_episode, np.sum(mean_eval_rewards, axis=0), label='Mean Validation Rewards')
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
    ax.plot(policy_mean_rewards[policy_agent], color=colours[i], label='Mean Training Rewards ' + policy_agent)
    ax.fill_between(np.arange(len(policy_mean_rewards[policy_agent])),
                    policy_mean_rewards[policy_agent] - policy_std_rewards[policy_agent],
                    policy_mean_rewards[policy_agent] + policy_std_rewards[policy_agent],
                    color=colours[i], alpha=0.3)
    ax.plot(eval_episode, mean_eval_rewards[i, :], label='Mean Validation Rewards ' + policy_agent)
    ax.fill_between(eval_episode,
                    mean_eval_rewards[i, :] - std_eval_rewards[i, :],
                    mean_eval_rewards[i, :] + std_eval_rewards[i, :],
                    alpha=0.3)
    ax.set_title('Learning Curve (Rewards)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rewards')
    ax.legend()

rewards_name_policy = save_path + '/training_rewards_policy.png'
plt.savefig(rewards_name, dpi=200)
plt.show()


#%% Test rollout

num_tests = 1000
# run until episode ends
episode_reward = 0
done = False
test_seed = 420
np.random.seed(seed=test_seed)
test_demand = test_env.dist.rvs(size=(num_tests + 1, test_env.num_periods), **test_env.dist_param)
obs = test_env.reset(customer_demand=test_demand[0, :])
dict_obs = {}
dict_info = {}
dict_actions = {}
dict_rewards = {}
period = 0

# Dict initialisation
for i in range(num_stages):
    sp = agent_ids[i]
    dict_obs[sp] = {}
    dict_info[sp] = {}
    dict_obs[sp]['inventory'] = np.zeros(num_periods + 1)
    dict_obs[sp]['backlog'] = np.zeros(num_periods + 1)
    dict_obs[sp]['order_u'] = np.zeros(num_periods + 1)
    dict_obs[sp]['time_dependent_s'] = np.zeros((num_periods + 1, test_env.max_delay))
    dict_info[sp]['demand'] = np.zeros(num_periods)
    dict_info[sp]['ship'] = np.zeros(num_periods)
    dict_info[sp]['acquisition'] = np.zeros(num_periods)
    dict_info[sp]['actual order'] = np.zeros(num_periods)
    dict_info[sp]['profit'] = np.zeros(num_periods)
    if standardise_state:
        dict_obs[sp]['inventory'][0] = test_env.rev_scale(obs[sp][0], 0, test_env.inv_max[i], test_env.a, test_env.b)
        dict_obs[sp]['backlog'][0] = test_env.rev_scale(obs[sp][1], 0, test_env.inv_max[i], test_env.a, test_env.b)
        dict_obs[sp]['order_u'][0] = test_env.rev_scale(obs[sp][2], 0, test_env.order_max[i], test_env.a, test_env.b)
        if time_dependency:
            dict_obs[sp]['time_dependent_s'][0] = test_env.rev_scale(obs[sp][4:4 + test_env.max_delay],
                                                                               np.zeros(test_env.max_delay),
                                                                               np.ones(test_env.max_delay)*test_env.inv_max[i],
                                                                               test_env.a, test_env.b)
    else:
        dict_obs[sp]['inventory'][0] = obs[sp][0]
        dict_obs[sp]['backlog'][0] = obs[sp][1]
        dict_obs[sp]['order_u'][0] = obs[sp][2]
        if time_dependency:
            dict_obs[sp]['time_dependent_s'][0] = obs[sp][4:4 + test_env.max_delay]
    dict_actions[sp] = np.zeros(num_periods)
    dict_rewards[sp] = np.zeros(num_periods)
    dict_rewards['Total'] = np.zeros(num_periods)

if use_lstm:
    reward = {}
    action = {}
    state = {}
    for m in range(num_stages):
        sp = agent_ids[m]
        reward[sp] = 0
        action[sp] = 0
        state[sp] = agent.get_policy(sp).get_initial_state()
while not done:
    if use_lstm:
        for i in range(num_stages):
            sp = agent_ids[i]
            action[sp], state[sp], _ = agent.compute_action(obs[sp], state=state[sp], prev_action=action[sp],
                                                            prev_reward=reward[sp], policy_id=sp)
    else:
        action = {}
        for i in range(num_stages):
            sp = agent_ids[i]
            action[sp] = agent.compute_action(obs[sp], policy_id=sp)
    obs, reward, dones, info = test_env.step(action)
    done = dones['__all__']
    for i in range(num_stages):
        sp = agent_ids[i]
        episode_reward += reward[sp]
        if standardise_state:
            dict_obs[sp]['inventory'][period + 1] = test_env.rev_scale(obs[sp][0], 0, test_env.inv_max[i],
                                                                       test_env.a, test_env.b)
            dict_obs[sp]['backlog'][period + 1] = test_env.rev_scale(obs[sp][1], 0, test_env.inv_max[i],
                                                                     test_env.a, test_env.b)
            dict_obs[sp]['order_u'][period + 1] = test_env.rev_scale(obs[sp][2], 0, test_env.order_max[i],
                                                                     test_env.a, test_env.b)
            if time_dependency:
                dict_obs[sp]['time_dependent_s'][period + 1] = test_env.rev_scale(
                    obs[sp][4:4 + test_env.max_delay],
                    np.zeros(test_env.max_delay),
                    np.ones(test_env.max_delay) * test_env.inv_max[i],
                    test_env.a, test_env.b)
        else:
            dict_obs[sp]['inventory'][period + 1] = obs[sp][0]
            dict_obs[sp]['backlog'][period + 1] = obs[sp][1]
            dict_obs[sp]['order_u'][period + 1] = obs[sp][2]
            if time_dependency:
                dict_obs[sp]['time_dependent_s'][period + 1] = obs[sp][4:4 + test_env.max_delay]
        dict_info[sp]['demand'][period] = info[sp]['demand']
        dict_info[sp]['ship'][period] = info[sp]['ship']
        dict_info[sp]['acquisition'][period] = info[sp]['acquisition']
        dict_info[sp]['actual order'][period] = info[sp]['actual order']
        dict_info[sp]['profit'][period] = info[sp]['profit']
        if standardise_actions:
            dict_actions[sp][period] = np.round(
                test_env.rev_scale(action[sp], 0, test_env.order_max[i],
                                                                    test_env.a, test_env.b), 0)
        else:
            dict_actions[sp][period] = np.round(action[sp], 0)
        dict_rewards[sp][period] = reward[sp]
        dict_rewards['Total'][period] += reward[sp]

    period += 1

#%% Plots
fig, axs = plt.subplots(4, num_stages, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=.3)

axs = axs.ravel()

for i in range(num_stages):
    sp = agent_ids[i]
    axs[i].plot(dict_obs[sp]['inventory'], label='Inventory')
    axs[i].plot(dict_obs[sp]['backlog'], label='Backlog')
    axs[i].plot(dict_obs[sp]['order_u'], label='Unfulfilled orders')
    axs[i].plot([0, 0], [inv_target[i], inv_target[i]], label='Target Inventory', color='r')
    axs[i].legend()
    axs[i].set_title(sp)
    axs[i].set_ylabel('Products')
    axs[i].set_xlim(0, num_periods)

    axs[i + num_stages].plot(dict_info[sp]['actual order'], label='Replenishment order', color='k')
    axs[i + num_stages].plot(np.arange(0, num_periods), dict_info[sp]['acquisition'], label='Acquisition')
    axs[i + num_stages].legend()
    axs[i + num_stages].set_ylabel('Products')
    axs[i + num_stages].set_xlim(0, num_periods)

    axs[i + num_stages * 2].plot(np.arange(0, num_periods), dict_info[sp]['demand'], label='demand')
    axs[i + num_stages * 2].plot(np.arange(0, num_periods), dict_info[sp]['ship'], label='shipment')

    axs[i + num_stages * 2].legend()
    axs[i + num_stages * 2].set_ylabel('Products')
    axs[i + num_stages * 2].set_xlim(0, num_periods)

    axs[i + num_stages * 3].plot(np.arange(1, num_periods + 1), dict_info[sp]['profit'], label='periodic profit')
    axs[i + num_stages * 3].plot(np.arange(1, num_periods + 1), np.cumsum(dict_info[sp]['profit']), label='cumulative profit')
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

#%% Test runs on final agent

list_test_rewards = []
stage_rewards = np.zeros((num_stages, num_tests))
for i in range(num_tests):
    demand = test_demand[i + 1, :]
    obs = test_env.reset(customer_demand=demand)
    episode_reward = 0
    done = False
    if use_lstm:
        reward = {}
        action = {}
        state = {}
        for m in range(num_stages):
            sp = agent_ids[m]
            reward[sp] = 0
            action[sp] = 0
            state[sp] = agent.get_policy(sp).get_initial_state()
    while not done:
        if use_lstm:
            for m in range(num_stages):
                sp = agent_ids[m]
                action[sp], state[sp], _ = agent.compute_action(obs[sp], state=state[sp], prev_action=action[sp],
                                                                prev_reward=reward[sp], policy_id=sp)
        else:
            action = {}
            for m in range(num_stages):
                sp = agent_ids[m]
                action[sp] = agent.compute_action(obs[sp], policy_id=sp)
        obs, reward, dones, info = test_env.step(action)
        done = dones['__all__']
        for m in range(num_stages):
            episode_reward += reward[agent_ids[m]]
            stage_rewards[m, i] += info[agent_ids[m]]['profit']
    list_test_rewards.append(episode_reward)

test_reward_mean = np.mean(list_test_rewards)
test_reward_std = np.std(list_test_rewards)
stage_test_reward_mean = np.mean(stage_rewards, axis=1)
stage_test_reward_std = np.std(stage_rewards, axis=1)

print(f"\nOn {num_tests} runs, the mean reward is: {test_reward_mean}, with standard deviation {test_reward_std}")
for m in range(num_stages):
    print(f"\nFor stage {m}, the mean profit is: {stage_test_reward_mean[m]}, "
          f"with standard deviation {stage_test_reward_std[m]}")