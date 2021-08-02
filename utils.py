"""
Helper functions for getting agents and rl configs
"""


from ray.rllib import agents


def get_config(algorithm, num_periods):
    if algorithm == 'ddpg' or algorithm == 'apex' or algorithm == 'td3':
        config = agents.ddpg.DEFAULT_CONFIG.copy()
        config["twin_q"] = True
        config["policy_delay"] = 1
        config["smooth_target_policy"] = False
        config["evaluation_interval"] = 1000
        config["evaluation_num_episodes"] = 10
        config["actor_hiddens"] = [400, 300]
        config["actor_hidden_activation"] = "relu"
        config["critic_hiddens"] = [400, 300]
        config["critic_hidden_activation"] = "relu"
        config["n_step"] = 1
        config["exploration_config"] = {
            "type": "OrnsteinUhlenbeckNoise",
            # For how many timesteps should we return completely random actions,
            # before we start adding (scaled) noise?
            "random_timesteps": 2000,
            # The OU-base scaling factor to always apply to action-added noise.
            "ou_base_scale": 0.1,
            # The OU theta param.
            "ou_theta": 0.15,
            # The OU sigma param.
            "ou_sigma": 0.2,
            # The initial noise scaling factor.
            "initial_scale": 1.0,
            # The final noise scaling factor.
            "final_scale": 1.0,
            # Timesteps over which to anneal scale (from initial to final values).
            "scale_timesteps": 10000,
        }
        config["timesteps_per_iteration"] = num_periods*100
        config["buffer_size"] = int(1e4)
        config["prioritized_replay"] = True
        config["prioritized_replay_alpha"] = 0.6
        config["prioritized_replay_beta"] = 0.4
        config["prioritized_replay_beta_annealing_timesteps"] = 20_000
        config["final_prioritized_replay_beta"] = 0.4
        config["prioritized_replay_eps"] = 1e-6
        config["critic_lr"] = 1e-3
        config["actor_lr"] = 1e-3
        config["target_network_update_freq"] = 0
        # Update the target by \tau * policy + (1-\tau) * target_policy
        config["tau"] = 0.002
        config["grad_clip"] = None
        config["learning_starts"] = 2e3
        config["rollout_fragment_length"] = 1
        config["train_batch_size"] = 256

    elif algorithm == 'ppo':
        config = agents.ppo.DEFAULT_CONFIG.copy()
        config["model"] = {
            "vf_share_layers": False,
            "fcnet_activation": 'relu',
            "fcnet_hiddens": [64, 64]
        }
        # Should use a critic as a baseline (otherwise don't use value baseline; required for using GAE).
        config["use_critic"] = True
        # If true, use the Generalized Advantage Estimator (GAE)
        config["use_gae"] = True
        # The GAE (lambda) parameter.
        config["lambda"] = 0.95
        config["gamma"] = 0.99
        # Initial coefficient for KL divergence.
        config["kl_coeff"] = 0.2
        # Number of timesteps collected for each SGD round. This defines the size of each SGD epoch.
        config["train_batch_size"] = num_periods*100
        # Number of SGD iterations in each outer loop (i.e., number of epochs to execute per train batch).
        config["num_sgd_iter"] = 10
        # Total SGD batch size across all devices for SGD. This defines the minibatch size within each epoch.
        config["sgd_minibatch_size"] = 120
        # Whether to shuffle sequences in the batch when training (recommended).
        config["shuffle_sequences"] = True
        # Coefficient of the value function loss. IMPORTANT: you must tune this if you set vf_share_layers=True inside your model's config.
        config["vf_loss_coeff"] = 1.0
        # Coefficient of the entropy regularizer.
        config["entropy_coeff"] = 0.0
        # Decay schedule for the entropy regularizer.
        config["entropy_coeff_schedule"] = None
        # PPO clip parameter.
        config["clip_param"] = 0.2
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        config["vf_clip_param"] = 10_000.0
        # If specified, clip the global norm of gradients by this amount.
        config["grad_clip"] = None
        # Target value for KL divergence.
        config["kl_target"] = 0.01

    elif algorithm == 'sac':
        config = agents.sac.DEFAULT_CONFIG.copy()
        config["Q_model"] = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define custom Q-model(s).
            "custom_model_config": {},
        }
        config["policy_model"] = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define a custom policy model.
            "custom_model_config": {},
        }
        config["buffer_size"] = int(1e4)
        config["prioritized_replay"] = False
        config["learning_starts"] = int(2000)
        config["train_batch_size"] = int(256)
        config["target_network_update_freq"] = int(10)


    else:
        raise Exception('Not Implemented')

    return config


def get_trainer(algorithm, rl_config, environment):
    if algorithm == 'ddpg':
        agent = agents.ddpg.DDPGTrainer(config=rl_config, env=environment)

    elif algorithm == 'ppo':
        agent = agents.ppo.PPOTrainer(config=rl_config, env=environment)

    elif algorithm == 'sac':
        agent = agents.sac.SACTrainer(config=rl_config, env=environment)

    else:
        raise Exception('Not Implemented')

    return agent

