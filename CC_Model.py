"""An example of implementing a centralized critic with ObservationFunction.
The advantage of this approach is that it's very simple and you don't have to
change the algorithm at all -- just use callbacks and a custom model.
However, it is a bit less principled in that you have to change the agent
observation spaces to include data that is only used at train time.
See also: centralized_critic.py for an alternative approach that instead
modifies the policy to add a centralized value function.
"""

import numpy as np
from gym.spaces import Dict, Discrete, Box
import argparse
import os

from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC

torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=100,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=7.99,
    help="Reward at which we stop training.")

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
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.action_model = TorchFC(
            Box(
                low=-np.zeros(4),
                high=np.array([30, np.inf, 30, 30]),
                dtype=np.float,
                shape=(4,)),  # one-hot encoded Discrete(6)
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


class FillInActions(DefaultCallbacks):
    """Fills in the opponent actions info in the training batches."""

    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id,
                                  policies, postprocessed_batch,
                                  original_batches, **kwargs):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 'stage_1' if agent_id == 'stage_0' else 'stage_0'
        action_encoder = ModelCatalog.get_preprocessor_for_space(
            Box(
                low=-np.inf,
                high=np.inf,
                dtype=np.float32,
                shape=(1,)
            )
        )

        # set the opponent actions into the observation
        _, opponent_batch = original_batches[other_id]
        #print(opponent_batch[SampleBatch.ACTIONS])
        #for a in opponent_batch[SampleBatch.ACTIONS]:
            #print('Look here')
            #print(action_encoder.transform(a))
        opponent_actions = np.array([
            action_encoder.transform(a)
            for a in opponent_batch[SampleBatch.ACTIONS]
        ])
        to_update[:, -2:] = opponent_actions


def central_critic_observer(agent_obs, **kw):
    """Rewrites the agent obs to include opponent data for training."""

    new_obs = {
        "stage_0": {
            "own_obs": agent_obs["stage_0"],
            "opponent_obs": agent_obs["stage_1"],
            "opponent_action": np.zeros(1, dtype=np.int32),  # filled in by FillInActions
        },
        "stage_1": {
            "own_obs": agent_obs["stage_1"],
            "opponent_obs": agent_obs["stage_0"],
            "opponent_action": np.zeros(1, dtype=np.int32),  # filled in by FillInActions
        },
    }
    return new_obs


if __name__ == "__main__":
    args = parser.parse_args()

    ModelCatalog.register_custom_model(
        "cc_model", CentralizedCriticModel)

    action_space = Discrete(2)
    observer_space = Dict({
        "own_obs": Discrete(6),
        # These two fields are filled in by the CentralCriticObserver, and are
        # not used for inference, only for training.
        "opponent_obs": Discrete(6),
        "opponent_action": Discrete(2),
    })

    config = {
        "env": TwoStepGame,
        "batch_mode": "complete_episodes",
        "callbacks": FillInActions,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "multiagent": {
            "policies": {
                "pol1": (None, observer_space, action_space, {}),
                "pol2": (None, observer_space, action_space, {}),
            },
            "policy_mapping_fn": (
                lambda aid, **kwargs: "pol1" if aid == 0 else "pol2"),
            "observation_fn": central_critic_observer,
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

    results = tune.run("PPO", config=config, stop=stop, verbose=1)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)