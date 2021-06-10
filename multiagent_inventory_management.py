from inv_mangement_env import MultiAgentInvManagement
import ray
from ray.rllib import agents
from ray import tune
import numpy as np
import matplotlib.pyplot as plt


def env_creator(env_config):
    return MultiAgentInvManagement(env_config)


test_env = MultiAgentInvManagement({})
num_agents = 4
agent_ids = ["retailer", "wholesaler", "distributor", "factory"]

env_name = "MultiAgentInventoryManagement"
env_config = {}
tune.register_env(env_name, env_creator)

print(test_env.reset())
print(test_env.step({'retailer': 10, 'wholesaler': 20, 'distributor': 20, 'factory': 10}))
print(test_env.step({'retailer': 10, 'wholesaler': 20, 'distributor': 20, 'factory': 10}))


obs_space = test_env.observation_space
act_space = test_env.action_space
num_agents = test_env.num_agents


def gen_policy():
    return None, obs_space, act_space, {}


policy_graphs = {}
for i in range(num_agents):
    policy_graphs[agent_ids[i]] = gen_policy()


def policy_mapping_fn(agent_id):
    if agent_id.startswith("retailer"):
        return "retailer"
    elif agent_id.startswith("wholesaler"):
        return "wholesaler"
    elif agent_id.startswith("distributor"):
        return "distributor"
    elif agent_id.startswith("factory"):
        return "factory"


ray.init()

trainer = agents.ppo.PPOTrainer(env="MultiAgentInventoryManagement", config={
    "multiagent": {
        "policies": policy_graphs,
        "policy_mapping_fn": policy_mapping_fn
    },
    "num_workers": 4,
    "env_config": env_config,
    "framework": 'torch',
    "model": {
        "vf_share_layers": False,
        "fcnet_activation": 'relu',
        "fcnet_hiddens": [256, 256]
    },
    "lr": 1e-5
})

p = 10

results = []
for i in range(p):
    res = trainer.train()
    results.append(res)
    if (i + 1) % 5 == 0:
        print('\rIter: {}\tReward: {:.2f}'.format(
            i + 1, res['episode_reward_mean']), end='')
ray.shutdown()

# Unpack values from each iteration
rewards = np.hstack([i['hist_stats']['episode_reward']
                     for i in results])

mean_rewards = np.array([np.mean(rewards[i - p:i + 1])
                         if i >= p else np.mean(rewards[:i + 1])
                         for i, _ in enumerate(rewards)])
std_rewards = np.array([np.std(rewards[i - p:i + 1])
                        if i >= p else np.std(rewards[:i + 1])
                        for i, _ in enumerate(rewards)])

fig = plt.figure(constrained_layout=True, figsize=(20, 10))
gs = fig.add_gridspec(2, 4)
ax0 = fig.add_subplot(gs[:, :-2])
ax0.fill_between(np.arange(len(mean_rewards)),
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 label='Standard Deviation', alpha=0.3)
ax0.plot(mean_rewards, label='Mean Rewards')
ax0.set_ylabel('Rewards')
ax0.set_xlabel('Episode')
ax0.set_title('Training Rewards')
ax0.legend()