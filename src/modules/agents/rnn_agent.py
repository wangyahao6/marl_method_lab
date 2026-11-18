import torch
import torch.nn as nn
from modules.layers.trhr import TRHRReLU
from modules.layers.act_layer import ActivateLayer


LEGACY_FC1_KEYS = ('fc1.weight', 'fc1.bias')


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        # Preserve the seeded initial values of every layer used by forward.
        self._initial_rng = torch.random.get_rng_state()
        nn.Linear(input_shape, args.rnn_hidden_dim)
        self.net = nn.Sequential(nn.Linear(input_shape, args.rnn_hidden_dim),
                                 nn.ReLU(inplace=True),
                                 ActivateLayer(args.rnn_hidden_dim, 'A'),
                                 nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                                 nn.ReLU(inplace=True),)
        if getattr(args, "trhr", None):
            self.net[4] = TRHRReLU(args.rnn_hidden_dim, args.n_agents, args.trhr)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.net[0].weight.new_zeros(1, self.args.rnn_hidden_dim)

    def legacy_initial_state(self, initial_state):
        """Reconstruct the unused tensors solely for comparing archived initializer hashes."""
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(self._initial_rng)
            legacy = nn.Linear(self.net[0].in_features, self.net[0].out_features, device='cpu')
        return dict(zip(LEGACY_FC1_KEYS, legacy.state_dict().values()), **initial_state)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        keys = [prefix + key for key in LEGACY_FC1_KEYS]
        present = [key in state_dict for key in keys]
        self.loaded_legacy_fc1 = all(present)
        if any(present):
            expected = (self.net[0].weight.shape, self.net[0].bias.shape)
            if not self.loaded_legacy_fc1 or any(
                not torch.is_tensor(state_dict[key]) or state_dict[key].shape != shape
                for key, shape in zip(keys, expected)
            ):
                error_msgs.append(f'{prefix}legacy fc1 requires the original weight and bias shapes')
            else:
                for key in keys:
                    state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, inputs, hidden_state):
        x = self.net(inputs)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
