import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch()

class RNNModel(TorchRNN, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs,
                 model_config,
                 name, **customized_model_kwargs):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = customized_model_kwargs.pop("fc_size", 64)
        self.use_initial_fc = customized_model_kwargs.pop("use_initial_fc", True)
        self.lstm_state_size = customized_model_kwargs.pop("lstm_state_size", 256)
        self.mlp_config = model_config["fcnet_hiddens"]
        mlp = [self.lstm_state_size]
        for i in range(len(self.mlp_config)):
            mlp.append(self.mlp_config[i])

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        if self.use_initial_fc:
            self.fc_action = nn.Linear(self.obs_size, self.fc_size)
            self.fc_value = nn.Linear(self.obs_size, self.fc_size)
            self.lstm_action = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)
            self.lstm_value = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)
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

        # Holds the current "base" output (before logits layer).
        self._input = None

    @override(ModelV2)
    def get_initial_state(self):
        h = [torch.zeros(self.lstm_state_size), torch.zeros(self.lstm_state_size)]
        return h

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

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        #print(seq_lens)

        self._input = [inputs, state]
        x = inputs
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


class SharedRNNModel(TorchRNN, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **customized_model_kwargs):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = customized_model_kwargs.pop("shared_fc_size", 64)
        self.use_initial_fc = customized_model_kwargs.pop("use_initial_fc", True)
        self.lstm_state_size = customized_model_kwargs.pop("lstm_state_size", 128)
        self.mlp_config = model_config["fcnet_hiddens"]
        if len(self.mlp_config) > 0:
            mlp = [self.lstm_state_size]
            for i in range(len(self.mlp_config)):
                mlp.append(self.mlp_config[i])

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        if self.use_initial_fc:
            self.fc1 = nn.Linear(self.obs_size, self.fc_size)
            self.lstm = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)
        else:
            self.lstm = nn.LSTM(self.obs_size, self.lstm_state_size, batch_first=True)

        if len(self.mlp_config) > 0:
            self.linears_action = nn.ModuleList([nn.Linear(mlp[i], mlp[i + 1]) for i in range(len(mlp) - 1)])
            self.linears_value = nn.ModuleList([nn.Linear(mlp[i], mlp[i + 1]) for i in range(len(mlp) - 1)])
            self.final_action = nn.Linear(mlp[len(mlp) - 1], num_outputs)
            self.final_value = nn.Linear(mlp[len(mlp) - 1], 1)
        else:
            self.final_action = nn.Linear(self.lstm_state_size, num_outputs)
            self.final_value = nn.Linear(self.lstm_state_size, 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        h = [torch.zeros(self.lstm_state_size), torch.zeros(self.lstm_state_size)]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"

        xx = self._features
        if len(self.mlp_config) > 0:
            for l in self.linears_value:
                xx = nn.functional.relu(l(xx))

        value_out = self.final_value(xx)

        return torch.reshape(value_out, [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = inputs
        if self.use_initial_fc:
            x = nn.functional.relu(self.fc1(x))

        self._features, [h, c] = self.lstm(x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])

        xx = self._features
        if len(self.mlp_config) > 0:
            for l in self.linears_action:
                xx = nn.functional.relu(l(xx))

        action_out = self.final_action(xx)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]