"""An example of implementing a centralized critic with ObservationFunction.
The advantage of this approach is that it's very simple and you don't have to
change the algorithm at all -- just use callbacks and a custom model.
However, it is a bit less principled in that you have to change the agent
observation spaces to include data that is only used at train time.
See also: centralized_critic.py for an alternative approach that instead
modifies the policy to add a centralized value function.
"""

import numpy as np
from gym.spaces import Box

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor

torch, nn = try_import_torch()


class CentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized value function.
    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).
    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **customized_model_kwargs):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)


        state_size = customized_model_kwargs["state_size"]
        self.action_model = TorchFC(
            Box(
                low=-np.ones(state_size),
                high=np.ones(state_size),
                dtype=np.float64,
                shape=(state_size,)),
            action_space,
            num_outputs,
            model_config,
            name + "_action")

        self.value_model = TorchFC(obs_space, action_space, 1, model_config,
                                   name + "_vf")
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        # Store model-input for possible `value_function()` call.
        self._model_in = [input_dict["obs_flat"], state, seq_lens]

        return self.action_model({
            "obs": input_dict["obs"]["own_obs"]
        }, state, seq_lens)

    def value_function(self):

        value_out, _ = self.value_model({
            "obs": self._model_in[0]
        }, self._model_in[1], self._model_in[2])
        return torch.reshape(value_out, [-1])

class CentralizedCriticModelRNN(TorchRNN, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **customized_model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)


        self.full_obs_size = get_preprocessor(obs_space)(obs_space).size
        self.obs_size = customized_model_kwargs["state_size"]
        self.fc_size = customized_model_kwargs.pop("fc_size", 64)
        self.fc_value_size = customized_model_kwargs.pop("fc_value_size", 64)
        self.use_initial_fc = customized_model_kwargs.pop("use_initial_fc", True)
        self.lstm_state_size = customized_model_kwargs.pop("lstm_state_size", 256)
        self.mlp_config = model_config["fcnet_hiddens"]
        mlp = [self.lstm_state_size]
        for i in range(len(self.mlp_config)):
            mlp.append(self.mlp_config[i])


        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        if self.use_initial_fc:
            self.fc_action = nn.Linear(self.obs_size, self.fc_size)
            self.fc_value = nn.Linear(self.full_obs_size, self.fc_value_size)
            self.lstm_action = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)
            self.lstm_value = nn.LSTM(self.fc_value_size, self.lstm_state_size, batch_first=True)
        else:
            self.lstm_action = nn.LSTM(self.obs_size, self.lstm_state_size, batch_first=True)
            self.lstm_value = nn.LSTM(self.obs_size, self.lstm_state_size, batch_first=True)

        if len(self.mlp_config) > 0:
            self.linears_action = nn.ModuleList([nn.Linear(mlp[i], mlp[i + 1]) for i in range(len(mlp) - 1)])
            self.linears_value = nn.ModuleList([nn.Linear(mlp[i], mlp[i + 1]) for i in range(len(mlp) - 1)])
            self.final_action = nn.Linear(mlp[len(mlp) - 1], num_outputs)
            self.final_value = nn.Linear(mlp[len(mlp) - 1], 1)
        else:
            self.final_action = nn.Linear(self.lstm_state_size, num_outputs)
            self.final_value = nn.Linear(self.lstm_state_size, 1)


        self._input = None

    @override(ModelV2)
    def get_initial_state(self):
        h = [torch.zeros(self.lstm_state_size), torch.zeros(self.lstm_state_size)]
        return h

    @override(TorchRNN)
    def forward_rnn(self, input, state, seq_lens):
        # Store model-input for possible `value_function()` call.
        self._input = [input, state, seq_lens]
        x = input[:, :, -self.obs_size:]
        # Pre-RNN FCN
        if self.use_initial_fc:
            x = nn.functional.relu(self.fc_action(x))

        # RNN
        x, [h, c] = self.lstm_action(x, [torch.unsqueeze(state[0], 0),
                                         torch.unsqueeze(state[1], 0)])

        # Post-RNN FCN
        if len(self.mlp_config) > 0:
            for l in self.linears_action:
                x = nn.functional.relu(l(x))

        # Get Action
        action_out = self.final_action(x)

        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    @override(ModelV2)
    def value_function(self):
        assert self._input is not None, "must call forward() first"

        x = self._input[0]
        if self.use_initial_fc:
            x = nn.functional.relu(self.fc_value(x))

        x, [h, c] = self.lstm_value(x, [torch.unsqueeze(self._input[1][0], 0),
                                        torch.unsqueeze(self._input[1][1], 0)])

        if len(self.mlp_config) > 0:
            for l in self.linears_value:
                x = nn.functional.relu(l(x))

        value_out = self.final_value(x)
        return torch.reshape(value_out, [-1])


class FillInActions(DefaultCallbacks):
    """Fills in the opponent actions info in the training batches."""

    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id,
                                  policies, postprocessed_batch,
                                  original_batches, **kwargs):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        action_encoder = ModelCatalog.get_preprocessor_for_space(
            Box(
                low=-1,
                high=1,
                dtype=np.float64,
                shape=(1,)
            )
        )
        agents = [*original_batches]
        agents.remove(agent_id)
        num_agents = len(agents)

        for i in range(num_agents):
            other_id = agents[i]

            # set the opponent actions into the observation
            _, opponent_batch = original_batches[other_id]
            opponent_actions = np.array([
                action_encoder.transform(np.clip(a, -1, 1))
                for a in opponent_batch[SampleBatch.ACTIONS]
            ])
            to_update[:, i] = np.squeeze(opponent_actions)  # <--------------------------


def central_critic_observer(agent_obs, **kw):
    """Rewrites the agent obs to include opponent data for training."""
    agents = [*agent_obs]
    num_agents = len(agents)
    obs_space = len(agent_obs[agents[0]])

    new_obs = dict()
    for agent in agents:
        new_obs[agent] = dict()
        new_obs[agent]["own_obs"] = agent_obs[agent]
        new_obs[agent]["opponent_obs"] = np.zeros((num_agents - 1)*obs_space)
        new_obs[agent]["opponent_action"] = np.zeros((num_agents - 1))
        i = 0
        for other_agent in agents:
            if agent != other_agent:
                new_obs[agent]["opponent_obs"][i*obs_space:i*obs_space + obs_space] = agent_obs[other_agent]
                i += 1

    return new_obs
