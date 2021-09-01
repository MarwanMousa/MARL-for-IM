from environments.MAIM_env import MultiAgentInvManagement
import ray
import json
from ray import tune
from ray.rllib.models import ModelCatalog
import numpy as np
import time
from utils import get_config, get_trainer, ensure_dir
from models.CC_Model import CentralizedCriticModel, CentralizedCriticModelRNN, FillInActions, central_critic_observer
from gym.spaces import Dict, Box
from hyperparams import get_hyperparams
import matplotlib.pyplot as plt
from matplotlib import rc
#%% Environment and Agent Configuration

train_agent = False
save_agent = True
save_path = "checkpoints/cc_agent/four_stage"
load_path = "checkpoints/cc_agent/four_stage"
LP_load_path = "LP_results/four_stage/"
load_iteration = str(600)
load_agent_path = load_path + '/checkpoint_000' + load_iteration + '/checkpoint-' + load_iteration

# Define plot settings
rc('font', **{'family': 'serif', 'serif': ['Palatino'], 'size': 13})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["figure.dpi"] = 200

# Set script seed
SEED = 52
np.random.seed(seed=SEED)

# Environment creator function for environment registration
def env_creator(configuration):
    env = MultiAgentInvManagement(configuration)
    return env

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
independent = False
standardise_state = True
standardise_actions = True
a = -1
b = 1
time_dependency = True
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
    "prev_demand": prev_demand,
    "prev_actions": prev_actions,
    "prev_length": prev_length,
}

# Loading in hyperparameters from hyperparameter search
use_optimal = True
configuration_name = "CC_5"

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

# Test environment
test_env = MultiAgentInvManagement(env_config)
obs_space = test_env.observation_space
act_space = test_env.action_space
num_agents = test_env.num_agents
size = obs_space.shape[0]
opponent_obs_space = Box(low=np.tile(obs_space.low, num_agents-1), high=np.tile(obs_space.high, num_agents-1),
                         dtype=np.float64, shape=(obs_space.shape[0]*(num_agents-1),))
opponent_act_space = Box(low=np.tile(act_space.low, num_agents-1), high=np.tile(act_space.high, num_agents-1),
                         dtype=np.float64, shape=(act_space.shape[0]*(num_agents-1),))
cc_obs_space = Dict({
    "own_obs": obs_space,
    "opponent_obs": opponent_obs_space,
    "opponent_action": opponent_act_space,
})


# Define policies to train
policy_graphs = {}
for i in range(num_agents):
    policy_graphs[agent_ids[i]] = None, cc_obs_space, act_space, {}


# Policy Mapping function to map each agent to appropriate stage policy

def policy_mapping_fn(agent_id, episode, **kwargs):
    for i in range(num_stages):
        if agent_id.startswith(agent_ids[i]):
            return agent_ids[i]

ModelCatalog.register_custom_model(
        "cc_model", CentralizedCriticModel)

ModelCatalog.register_custom_model(
        "cc_rnn_model", CentralizedCriticModelRNN)

# Algorithm used
algorithm = 'ppo'
# Training Set-up
ray.init(ignore_reinit_error=True, local_mode=True, num_cpus=2)
rl_config = get_config(algorithm, num_periods=num_periods, seed=SEED)
rl_config["multiagent"] = {
    "policies": policy_graphs,
    "policy_mapping_fn": policy_mapping_fn,
    "replay_mode": "lockstep",
    "observation_fn": central_critic_observer
}
rl_config["callbacks"] = FillInActions
rl_config["env"] = "MultiAgentInventoryManagement"
rl_config["env_config"] = CONFIG

if not use_lstm:
    rl_config["model"]["custom_model"] = "cc_model"
    rl_config["model"]["custom_model_config"] = {"state_size": obs_space.shape[0]}
else:
    rl_config["shuffle_sequences"] = False
    rl_config["model"]["custom_model"] = "cc_rnn_model"
    rl_config["model"]["max_seq_len"] = num_periods
    rl_config["model"]["custom_model_config"] = {"fc_size": 64,
                                                 "fc_value_size": 64,
                                                 "use_initial_fc": True,
                                                 "lstm_state_size": 128,
                                                 "state_size": obs_space.shape[0]}

# Overwrite default parameters
if use_optimal:
    rl_config = o_config
    rl_config["env_config"] = CONFIG
    rl_config["num_workers"] = 2
    rl_config["num_gpus"] = 0
    rl_config["multiagent"] = {
        "policies": policy_graphs,
        "policy_mapping_fn": policy_mapping_fn,
        "replay_mode": "lockstep",
        "observation_fn": central_critic_observer
    }
    rl_config["callbacks"] = FillInActions
    rl_config["model"]["custom_model_config"]["state_size"] = obs_space.shape[0]


agent = get_trainer(algorithm, rl_config, "MultiAgentInventoryManagement")

#%% Training
if train_agent:
    # Training
    iters = 800
    min_iter_save = 100
    checkpoint_interval = 10
    results = []
    for i in range(iters):
        res = agent.train()
        res["config"] = 0
        results.append(res)
        if (i + 1) % 1 == 0:
            print('\rIter: {}\tReward: {:.2f}'.format(
                i + 1, res['episode_reward_mean']), end='')

        if (i + 1) % checkpoint_interval == 0 and i > min_iter_save and save_agent:
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

decLP_rewards_mean = np.load(LP_load_path + 'DecLP/reward_mean.npy')
decLP_rewards_std = np.load(LP_load_path + 'DecLP/reward_std.npy')

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
                alpha=0.3)
ax.plot(mean_rewards, label='Mean Rewards')
# Plot decLP rewards
ax.fill_between(np.arange(len(mean_rewards)),
                np.ones(len(mean_rewards)) * (decLP_rewards_mean - decLP_rewards_std),
                np.ones(len(mean_rewards)) * (decLP_rewards_mean + decLP_rewards_std),
                alpha=0.3)
ax.plot(np.arange(len(mean_rewards)), np.ones(len(mean_rewards)) * (decLP_rewards_mean), label='Mean Decentralise LP rewards')

ax.set_ylabel('Rewards')
ax.set_xlabel('Episode')
ax.legend()

if save_agent:
    rewards_name = save_path + '/training_rewards.png'
    plt.savefig(rewards_name, dpi=200)

plt.show()


colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:olive', 'tab:cyan']
fig, ax = plt.subplots()
for i in range(num_agents):
    policy_agent = agent_ids[i]
    ax.plot(policy_mean_rewards[policy_agent], color=colours[i], label='Mean Training Rewards ' + policy_agent)
    ax.fill_between(np.arange(len(policy_mean_rewards[policy_agent])),
                    policy_mean_rewards[policy_agent] - policy_std_rewards[policy_agent],
                    policy_mean_rewards[policy_agent] + policy_std_rewards[policy_agent],
                    color=colours[i], alpha=0.3)
    ax.set_title('Learning Curve (Rewards)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rewards')
    ax.legend()

rewards_name_policy = save_path + '/training_rewards_policy.png'
plt.savefig(rewards_name_policy, dpi=200)
plt.show()

#%% Test rollout

num_tests = 1000
# run until episode ends
episode_reward = 0
done = False
test_seed = 420
np.random.seed(seed=test_seed)
test_demand = test_env.dist.rvs(size=(num_tests, test_env.num_periods), **test_env.dist_param)
noisy_demand = False
noise_threshold = 40/100

noisy_delay = False
noisy_delay_threshold = 50/100

if noisy_demand:
    for i in range(num_tests):
        for j in range(num_periods):
            double_demand = np.random.uniform(0, 1)
            zero_demand = np.random.uniform(0, 1)
            if double_demand <= noise_threshold:
                test_demand[i, j] = 2 * test_demand[i, j]
            if zero_demand <= noise_threshold:
                test_demand[i, j] = 0


obs = test_env.reset(customer_demand=test_demand[0, :], noisy_delay=noisy_delay, noisy_delay_threshold=noisy_delay_threshold)
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
        dict_obs[sp]['inventory'][0] = test_env.rev_scale(obs[sp][0], 0, test_env.inv_max[i],
                                                                    test_env.a, test_env.b)
        dict_obs[sp]['backlog'][0] = test_env.rev_scale(obs[sp][1], 0, test_env.inv_max[i],
                                                                  test_env.a, test_env.b)
        dict_obs[sp]['order_u'][0] = test_env.rev_scale(obs[sp][2], 0, test_env.order_max[i],
                                                                  test_env.a, test_env.b)
    else:
        dict_obs[sp]['inventory'][0] = obs[sp][0]
        dict_obs[sp]['backlog'][0] = obs[sp][1]
        dict_obs[sp]['order_u'][0] = obs[sp][2]
    dict_actions[sp] = np.zeros(num_periods)
    dict_rewards[sp] = np.zeros(num_periods)
    dict_rewards['Total'] = np.zeros(num_periods)

action = {}
# Initialise actions
for m in range(num_stages):
    action[agent_ids[m]] = 0
if use_lstm:
    reward = {}
    state = {}
    for m in range(num_stages):
        sp = agent_ids[m]
        reward[sp] = 0
        state[sp] = agent.get_policy(sp).get_initial_state()
while not done:
    for m in range(num_stages):
        sp = agent_ids[m]
        opponent_obs = np.zeros(size * (num_agents - 1))
        opponent_act = np.zeros(num_agents - 1)
        counter = 0
        for o in range(num_stages):
            if o != m:
                opponent_obs[counter * size:counter * size + size] = obs[agent_ids[o]]
                opponent_act[counter] = np.clip(action[agent_ids[o]], a, b)
                counter += 1
        CC_Obs = dict({"own_obs": obs[agent_ids[m]],
                       "opponent_obs": opponent_obs,
                       "opponent_action": opponent_act})
        if use_lstm:
            action[sp], state[sp], _ = agent.compute_single_action(CC_Obs, state=state[sp],
                                                                   prev_action=action[sp],
                                                                   prev_reward=reward[sp],
                                                                   policy_id=sp)
        else:
            action[sp] = agent.compute_single_action(CC_Obs, policy_id=sp)
    obs, reward, dones, info = test_env.step(action)
    done = dones['__all__']
    for i in range(num_stages):
        sp = agent_ids[i]
        episode_reward += reward[sp]
        if standardise_state:
            dict_obs[sp]['inventory'][period + 1] = test_env.rev_scale(obs[sp][0], 0,
                                                                             test_env.inv_max[i], test_env.a, test_env.b)
            dict_obs[sp]['backlog'][period + 1] = test_env.rev_scale(obs[sp][1], 0,
                                                                           test_env.inv_max[i], test_env.a, test_env.b)
            dict_obs[sp]['order_u'][period + 1] = test_env.rev_scale(obs[sp][2], 0,
                                                                           test_env.order_max[i], test_env.a, test_env.b)
        else:
            dict_obs[sp]['inventory'][period + 1] = obs[sp][0]
            dict_obs[sp]['backlog'][period + 1] = obs[sp][1]
            dict_obs[sp]['order_u'][period + 1] = obs[sp][2]
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
fig, axs = plt.subplots(3, num_stages, figsize=(18, 9), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.06, wspace=.16)
axs = axs.ravel()

for i in range(num_stages):
    sp = agent_ids[i]
    axs[i].plot(dict_obs[sp]['inventory'], label='Inventory', lw=2)
    axs[i].plot(dict_obs[sp]['backlog'], label='Backlog', color='tab:red', lw=2)
    title = 'Stage ' + str(i + 1)
    axs[i].set_title(title)
    axs[i].set_xlim(0, num_periods)
    axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if i == 0:
        axs[i].legend()
        axs[i].set_ylabel('Products')

    axs[i+num_stages].plot(dict_info[sp]['actual order'], label='Replenishment order', color='k', lw=2)
    axs[i+num_stages].plot(dict_info[sp]['demand'], label='Demand', color='tab:orange', lw=2)
    axs[i+num_stages].set_xlim(0, num_periods)
    axs[i + num_stages].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if i == 0:
        axs[i+num_stages].legend()
        axs[i + num_stages].set_ylabel('Products')

    axs[i+num_stages*2].plot(dict_info[sp]['profit'], label='Periodic profit', color='tab:green', lw=2)
    axs[i+num_stages*2].plot(np.cumsum(dict_info[sp]['profit']), label='Cumulative profit', color='salmon', lw=2)
    axs[i+num_stages*2].plot([0, num_periods], [0, 0], color='k')
    axs[i+num_stages*2].set_xlabel('Period')
    axs[i+num_stages*2].set_xlim(0, num_periods)
    if i == 0:
        axs[i + num_stages * 2].legend()
        axs[i + num_stages * 2].set_ylabel('Profit')

if save_agent:
    test_name = save_path + '/test_rollout.png'
    plt.savefig(test_name, dpi=200)
plt.show()
#%% Test runs on final agent
reward_list = []
inventory_list = []
backlog_list = []
customer_backlog_list = []
stage_rewards = np.zeros((num_stages, num_tests))
profit = np.zeros((num_tests, num_periods))

start_time = time.time()
for i in range(num_tests):
    demand = test_demand[i, :]
    obs = test_env.reset(customer_demand=demand, noisy_delay=noisy_delay, noisy_delay_threshold=noisy_delay_threshold)
    episode_reward = 0
    done = False
    action = {}
    # Initialise actions
    for m in range(num_stages):
        action[agent_ids[m]] = 0
    if use_lstm:
        reward = {}
        state = {}
        for m in range(num_stages):
            sp = agent_ids[m]
            reward[sp] = 0
            state[sp] = agent.get_policy(sp).get_initial_state()

    t = 0
    total_inventory = 0
    total_backlog = 0
    customer_backlog = 0
    while not done:
        for m in range(num_stages):
            sp = agent_ids[m]
            opponent_obs = np.zeros(size * (num_agents - 1))
            opponent_act = np.zeros(num_agents - 1)
            counter = 0
            for o in range(num_stages):
                if o != m:
                    opponent_obs[counter * size:counter * size + size] = obs[agent_ids[o]]
                    opponent_act[counter] = np.clip(action[agent_ids[o]], a, b)
                    counter += 1
            CC_Obs = dict({"own_obs": obs[agent_ids[m]],
                           "opponent_obs": opponent_obs,
                           "opponent_action": opponent_act})
            if use_lstm:
                action[sp], state[sp], _ = agent.compute_single_action(CC_Obs, state=state[sp],
                                                                       prev_action=action[sp],
                                                                       prev_reward=reward[sp],
                                                                       policy_id=sp)
            else:
                action[sp] = agent.compute_single_action(CC_Obs, policy_id=sp)
        obs, reward, dones, info = test_env.step(action)
        done = dones['__all__']

        total_step_reward = 0
        total_step_inv = 0
        total_step_bl = 0
        for m in range(num_stages):
            episode_reward += reward[agent_ids[m]]
            stage_rewards[m, i] += info[agent_ids[m]]['profit']
            total_step_reward += reward[agent_ids[m]]
            total_step_inv += test_env.rev_scale(obs[agent_ids[m]][0], 0, test_env.inv_max[m], test_env.a, test_env.b)
            total_step_bl += test_env.rev_scale(obs[agent_ids[m]][1], 0, test_env.inv_max[m], test_env.a, test_env.b)

        profit[i, t] = total_step_reward
        total_inventory += total_step_inv
        total_backlog += total_step_bl
        customer_backlog += test_env.rev_scale(obs[agent_ids[0]][1], 0, test_env.inv_max[0], test_env.a, test_env.b)
        t += 1

    reward_list.append(episode_reward)
    inventory_list.append(total_inventory)
    backlog_list.append(total_backlog)
    customer_backlog_list.append(customer_backlog)

cc_time = time.time() - start_time

stage_test_reward_mean = np.mean(stage_rewards, axis=1)
stage_test_reward_std = np.std(stage_rewards, axis=1)

cc_reward_mean = np.mean(reward_list)
cc_reward_std = np.std(reward_list)
inventory_level_mean = np.mean(inventory_list)
inventory_level_std = np.std(inventory_list)
backlog_level_mean = np.mean(backlog_list)
backlog_level_std = np.std(backlog_list)
customer_backlog_mean = np.mean(customer_backlog_list)
customer_backlog_std = np.std(customer_backlog_list)

print(f"\nOn {num_tests} runs, the mean reward is: {cc_reward_mean}, with standard deviation {cc_reward_std}")
print(f'Mean inventory level is: {inventory_level_mean} with std: {inventory_level_std}')
print(f'Mean backlog level is: {backlog_level_mean} with std: {backlog_level_std}')
print(f'Mean customer backlog level is: {customer_backlog_mean } with std: {customer_backlog_std}')
print(f'Took {cc_time}s for {num_tests} inference')
for m in range(num_stages):
    print(f"\nFor stage {m}, the mean profit is: {stage_test_reward_mean[m]}, "
          f"with standard deviation {stage_test_reward_std[m]}")

np.save(save_path+'/reward_mean.npy', cc_reward_mean)
np.save(save_path+'/reward_std.npy', cc_reward_std)
np.save(save_path+'/inventory_mean.npy', inventory_level_mean)
np.save(save_path+'/inventory_std.npy', inventory_level_std)
np.save(save_path+'/backlog_mean.npy', backlog_level_mean)
np.save(save_path+'/backlog_std.npy', backlog_level_std)
np.save(save_path+'/customer_backlog_mean', customer_backlog_mean)
np.save(save_path+'/customer_backlog_std', customer_backlog_std)
np.save(save_path+'/profit', profit)
np.save(save_path+'/time', cc_time)
