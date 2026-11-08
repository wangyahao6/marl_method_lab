REGISTRY = {}

from .central_rnn_agent import CentralRNNAgent
from .rnn_agent import RNNAgent
from .single_ff_rnn_agent import SingleFFRNNAgent
from .rnn_agent_rmix import RmixAgent
from .iqn_rnn_agent import IQNRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["single_ff_rnn"] = SingleFFRNNAgent
REGISTRY["iqn_rnn"] = IQNRNNAgent
REGISTRY["rnn_agent_rmix"] = RmixAgent
REGISTRY["central_rnn"] = CentralRNNAgent
