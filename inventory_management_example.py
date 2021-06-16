'''
Example Single-Agent environment for inventory management
Baseline code for this files can be found at:
https://www.datahubbs.com/how-to-use-deep-reinforcement-learning-to-improve-your-supply-chain/
'''
from or_gym.utils import create_env
import ray
from ray.rllib import agents
from ray import tune
import numpy as np
import matplotlib.pyplot as plt

def register_env(env_name, env_config={}):
    env = create_env(env_name)
    tune.register_env(env_name,
        lambda env_name: env(env_name,
            env_config=env_config))


# Environment and RL Configuration Settings
env_name = 'InvManagement-v1'
periods = 50
env_config = {"periods": periods}  # Change environment parameters here
rl_config = dict(
    env=env_name,
    num_workers=4,
    env_config=env_config,
    framework='torch',
    model=dict(
        vf_share_layers=False,
        fcnet_activation='relu',
        fcnet_hiddens=[256, 256]
    ),
    lr=1e-5,
    seed=52
)

# Register environment
register_env(env_name, env_config)

# Initialize Ray and Build Agent
ray.init(ignore_reinit_error=True, checkpoint_at_end=True)
agent = agents.ddpg.DDPGTrainer(env=env_name,
                              config=rl_config)

results = []
for i in range(50):
    res = agent.train()
    results.append(res)
    if (i + 1) % 5 == 0:
        print('\rIter: {}\tReward: {:.2f}'.format(
            i + 1, res['episode_reward_mean']), end='')
    if (i + 1) % 10 == 0:
        chkpt_file = agent.save(
            '/Users/marwanmousa/University/MSc_AI/Individual_Project/MARL-and-DMPC-for-OR/checkpoints/single_agent_example')
ray.shutdown()

# Unpack values from each iteration
rewards = np.hstack([i['hist_stats']['episode_reward']
                     for i in results])


p = 200
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
ax.set_title('Training Rewards')
ax.legend()
plt.show()


InvManagement = create_env('InvManagement-v1', env_config)
test_env = InvManagement(env_config)


# run until episode ends
episode_reward = 0
done = False
obs = test_env.reset()
list_actions = []
list_obs = []
for i in range(periods):
    action = agent.compute_action(obs)
    obs, reward, done, info = test_env.step(action)
    episode_reward += reward
    list_actions.append(action)
    list_obs.append(obs)


