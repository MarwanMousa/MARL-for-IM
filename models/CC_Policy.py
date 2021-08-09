import numpy as np
from gym.spaces import Box, Discrete
import os

import ray
from ray import tune
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, \
    KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN


torch, nn = try_import_torch()

OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"


class CCPolicy_Model(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **customized_model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        # Get value network architecture
        full_state_size = customized_model_kwargs["full_state_size"]
        self.fc_net_value = customized_model_kwargs["fcnet_hiddens_value"]
        mlp = [full_state_size]
        for i in range(len(self.fc_net_value)):
            mlp.append(self.fc_net_value[i])

        # Base of the model
        self.model = TorchFC(obs_space, action_space, num_outputs,
                             model_config, name)

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        self.central_vf = nn.ModuleList([nn.Linear(mlp[i], mlp[i + 1]) for i in range(len(mlp) - 1)])
        self.central_vf_out = nn.Linear(mlp[len(mlp) - 1], 1)


    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        x = torch.cat([
            obs, opponent_obs, opponent_actions
        ], 1)

        for l in self.central_vf:
            x = nn.functional.relu(l(x))

        return torch.reshape(self.central_vf_out(x), [-1])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used

class CCPolicy_ModelRNN(TorchRNN, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **customized_model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)


        self.full_obs_size = customized_model_kwargs["full_state_size"]
        self.obs_size = customized_model_kwargs["state_size"]

        self.fc_action_size = customized_model_kwargs.pop("fc_action_size", 64)
        self.fc_value_size = customized_model_kwargs.pop("fc_value_size", 64)

        self.use_initial_fc = customized_model_kwargs.pop("use_initial_fc", True)
        self.lstm_state_size = customized_model_kwargs.pop("lstm_state_size", 256)

        # Base of the model
        self.init = nn.Linear(self.obs_size, 1)

        # Get policy model architecture
        fc_net = model_config["fcnet_hiddens"]
        mlp = [self.lstm_state_size]
        for i in range(len(fc_net)):
            mlp.append(fc_net[i])

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        if self.use_initial_fc:
            self.fc_action = nn.Linear(self.obs_size, self.fc_action_size)
            self.fc_value = nn.Linear(self.full_obs_size, self.fc_value_size)
            self.lstm_action = nn.LSTM(self.fc_action_size, self.lstm_state_size, batch_first=True)
            self.lstm_value = nn.LSTM(self.fc_value_size, self.lstm_state_size, batch_first=True)
        else:
            self.lstm_action = nn.LSTM(self.obs_size, self.lstm_state_size, batch_first=True)
            self.lstm_value = nn.LSTM(self.obs_size, self.lstm_state_size, batch_first=True)


        self.linears_action = nn.ModuleList([nn.Linear(mlp[i], mlp[i + 1]) for i in range(len(mlp) - 1)])
        self.linears_value = nn.ModuleList([nn.Linear(mlp[i], mlp[i + 1]) for i in range(len(mlp) - 1)])
        self.final_action = nn.Linear(mlp[len(mlp) - 1], num_outputs)
        self.final_value = nn.Linear(mlp[len(mlp) - 1], 1)

        self._input = None

    @override(ModelV2)
    def get_initial_state(self):
        h = [torch.zeros(self.lstm_state_size), torch.zeros(self.lstm_state_size)]
        return h

    @override(TorchRNN)
    def forward_rnn(self, input, state, seq_lens):
        # Store model-input for possible `value_function()` call.
        self._input = [input, state, seq_lens]

        x = input
        # Pre-RNN FCN
        if self.use_initial_fc:
            x = nn.functional.relu(self.fc_action(x))

        # RNN
        x, [h, c] = self.lstm_action(x, [torch.unsqueeze(state[0], 0),
                                         torch.unsqueeze(state[1], 0)])

        # Post-RNN FCN
        for l in self.linears_action:
            x = nn.functional.relu(l(x))

        # Get Action
        action_out = self.final_action(x)

        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        assert self._input is not None, "must call forward() first"
        try:
            x = torch.cat([obs, opponent_obs, opponent_actions], 1)
        except:
            obs = torch.unsqueeze(obs, 0)
            opponent_obs = torch.unsqueeze(opponent_obs, 0)
            opponent_actions = torch.unsqueeze(opponent_actions, 0)
            x = torch.cat([obs, opponent_obs, opponent_actions], 1)
        try:
            x = torch.reshape(x, (self._input[0].shape[0], self._input[0].shape[1], self.full_obs_size))
        except:
            x = torch.unsqueeze(x, 1)

        if self.use_initial_fc:
            x = nn.functional.relu(self.fc_value(x))

        x, [h, c] = self.lstm_value(x, [torch.unsqueeze(self._input[1][0], 0),
                                        torch.unsqueeze(self._input[1][1], 0)])

        for l in self.linears_value:
            x = nn.functional.relu(l(x))

        value_out = self.final_value(x)

        return torch.reshape(value_out, [-1])

    @override(ModelV2)
    def value_function(self):
        assert self._input is not None, "must call forward() first"
        return self.init(self._input[0])  # not used


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        if self.config["framework"] == "torch":
            self.compute_central_vf = self.model.central_value_function



# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):

    state_size = sample_batch['obs'].shape[1]
    if (hasattr(policy, "compute_central_vf")):
        assert other_agent_batches is not None
        agents = [*other_agent_batches]
        counter = 0
        action_encoder = ModelCatalog.get_preprocessor_for_space(
            Box(low=-1, high=1, dtype=np.float64, shape=(1,))
        )
        sample_batch[OPPONENT_OBS] = np.tile(np.zeros_like(sample_batch[SampleBatch.CUR_OBS]), 3)
        sample_batch[OPPONENT_ACTION] = np.tile(np.zeros_like(sample_batch[SampleBatch.ACTIONS]), 3)
        for agent in agents:
            opponent_batch = other_agent_batches[agent][1]
            sample_batch[OPPONENT_OBS][:, counter*state_size:counter*state_size + state_size] = opponent_batch['obs']
            opponent_actions = np.array([
                action_encoder.transform(np.clip(a, -1, 1))
                for a in opponent_batch['actions']
            ])
            sample_batch[OPPONENT_ACTION][:, counter] = np.squeeze(opponent_actions)
            counter += 1
        # also record the opponent obs and actions in the trajectory


        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            convert_to_torch_tensor(
                sample_batch[SampleBatch.CUR_OBS], policy.device),
            convert_to_torch_tensor(
                sample_batch[OPPONENT_OBS], policy.device),
            convert_to_torch_tensor(
                sample_batch[OPPONENT_ACTION], policy.device)) \
            .cpu().detach().numpy()
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OPPONENT_OBS] = np.tile(np.zeros_like(
            sample_batch[SampleBatch.CUR_OBS]), 3)
        sample_batch[OPPONENT_ACTION] = np.tile(np.zeros_like(
            sample_batch[SampleBatch.ACTIONS]), 3)
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = torch_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS], train_batch[OPPONENT_OBS],
        train_batch[OPPONENT_ACTION])
    #print("train batch")
    #print(train_batch['obs'].shape)
    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    TorchKLCoeffMixin.__init__(policy, config)
    TorchEntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                       config["entropy_coeff_schedule"])
    TorchLR.__init__(policy, config["lr"], config["lr_schedule"])


CCPPOPolicy = PPOTorchPolicy.with_updates(
    name="CCPPOPolicy",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_init=setup_torch_mixins,
    mixins=[
        TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
        CentralizedValueMixin
    ])


def get_policy_class(config):
        return CCPPOPolicy


CCTrainer = PPOTrainer.with_updates(
    name="CCPPOTrainer",
    default_policy=CCPPOPolicy,
    get_policy_class=get_policy_class,
)

if __name__ == "__main__":
    ray.init()
    args = 1

    ModelCatalog.register_custom_model(
        "cc_model", CCPolicy_Model)

    config = {
        "env": TwoStepGame,
        "batch_mode": "complete_episodes",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "multiagent": {
            "policies": {
                "pol1": (None, Discrete(6), TwoStepGame.action_space, {
                    "framework": args.framework,
                }),
                "pol2": (None, Discrete(6), TwoStepGame.action_space, {
                    "framework": args.framework,
                }),
            },
            "policy_mapping_fn": (
                lambda aid, **kwargs: "pol1" if aid == 0 else "pol2"),
        },
        "model": {
            "custom_model": "cc_model",
        },
        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(CCTrainer, config=config, stop=stop, verbose=1)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)