import torch.nn as nn

from modules.layers.act_layer import ActivateLayer
from modules.layers.trhr import TRHRReLU


class SingleFFRNNAgent(nn.Module):
    """Shared recurrent agent with one hidden feedforward layer before the GRU."""

    def __init__(self, input_shape, args):
        super().__init__()
        self.hidden_dim = args.rnn_hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_shape, self.hidden_dim),
            nn.ReLU(inplace=True),
            ActivateLayer(self.hidden_dim, "A"),
        )
        if getattr(args, "trhr", None):
            self.net[1] = TRHRReLU(self.hidden_dim, args.n_agents, args.trhr)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.net[0].weight.new_zeros(1, self.hidden_dim)

    def forward(self, inputs, hidden_state):
        features = self.net(inputs)
        hidden = self.rnn(features, hidden_state.reshape(-1, self.hidden_dim))
        return self.fc2(hidden), hidden
