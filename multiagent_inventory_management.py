from inv_mangement_env import MultiAgentInvManagement
import ray
from ray.rllib import agents
from ray import tune
import numpy as np
import matplotlib.pyplot as plt


def env_creator(configuration):
    env = MultiAgentInvManagement(configuration)
    return env


# Environment Configuration
num_stages = 3
num_periods = 50
customer_demand = np.ones(num_periods) * 5
init_inv = np.ones(num_stages)*50

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
    "init_inv": init_inv
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


# Training Set-up
ray.init(ignore_reinit_error=True, local_mode=True)
rl_config = agents.ddpg.DEFAULT_CONFIG.copy()
rl_config["multiagent"] = {
    "policies": policy_graphs,
    "policy_mapping_fn": policy_mapping_fn
}
rl_config["num_workers"] = 3
rl_config["normalize_actions"] = False
rl_config["env_config"] = CONFIG
rl_config["framework"] = 'torch'
rl_config["model"] = {
        "vf_share_layers": False,
        "fcnet_activation": 'relu',
        "fcnet_hiddens": [256, 256, 256]
    }
rl_config["lr"] = 1e-5
rl_config["seed"] = 52
agent = agents.ddpg.DDPGTrainer(config=rl_config, env=MultiAgentInvManagement)

# Training
iters = 50
results = []
for i in range(iters):
    res = agent.train()
    results.append(res)
    if (i + 1) % 5 == 0:
        chkpt_file = agent.save('/Users/marwanmousa/University/MSc_AI/Individual_Project/MARL-and-DMPC-for-OR/checkpoints/multi_agent')
        print('\rIter: {}\tReward: {:.2f}'.format(
            i + 1, res['episode_reward_mean']), end='')

ray.shutdown()

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
ax.set_ylabel('Rewards')
ax.set_xlabel('Episode')
ax.set_title('Training Rewards')
ax.legend()
plt.show()

colours = ['r', 'g', 'b', 'k']
fig, ax = plt.subplots()
for i in range(num_agents):
    policy_agent = agent_ids[i]
    ax.plot(policy_mean_rewards[policy_agent], colours[i], label=agent_ids[i])
    ax.legend

plt.show()

'''
# run until episode ends
episode_reward = 0
done = False
agent.restore(chkpt_file)
obs = test_env.reset()
list_actions = []
list_obs = []
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = test_env.step(action)
    episode_reward += reward
    list_actions.append(action)
    list_obs.append(obs)
'''