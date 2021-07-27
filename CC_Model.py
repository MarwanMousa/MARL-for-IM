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
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)


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
        #print(self._model_in[0])
        value_out, _ = self.value_model({
            "obs": self._model_in[0]
        }, self._model_in[1], self._model_in[2])
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
        #print('before updating')
        #print(to_update[0, :])
        #print(to_update[1, :])
        for i in range(num_agents):
            other_id = agents[i]

            # set the opponent actions into the observation
            _, opponent_batch = original_batches[other_id]
            opponent_actions = np.array([
                action_encoder.transform(np.clip(a, -1, 1))
                for a in opponent_batch[SampleBatch.ACTIONS]
            ])
            to_update[:, i] = np.squeeze(opponent_actions)  # <--------------------------
            #print('after updating')
            #print(to_update[0, :])
            #print(to_update[1, :])


def central_critic_observer(agent_obs, **kw):
    """Rewrites the agent obs to include opponent data for training."""
    agents = [*agent_obs]
    #print(agents)
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

    #print('new_obs')
    #print(new_obs)
    return new_obs
