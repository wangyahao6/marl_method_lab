import copy
import unittest
from types import SimpleNamespace

import torch as th
from torch import nn

from components.optimizer_checkpoint import load_optimizer_state
from drivers.continual.common.checkpoints import hash_model_states, parent_hash_fields
from modules.agents.rnn_agent import RNNAgent


class OriginalAgent(RNNAgent):
    def __init__(self, input_shape, args):
        nn.Module.__init__(self)
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.net = nn.Sequential(nn.Linear(input_shape, args.rnn_hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Identity(),
                                 nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                                 nn.ReLU(inplace=True))
        from modules.layers.act_layer import ActivateLayer
        self.net[2] = ActivateLayer(args.rnn_hidden_dim, 'A')
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)


class RNNAgentCheckpointTest(unittest.TestCase):
    def setUp(self):
        args = SimpleNamespace(rnn_hidden_dim=8, n_actions=5)
        with th.random.fork_rng(devices=[]):
            th.manual_seed(4)
            self.original = OriginalAgent(6, args)
            self.original_rng = th.random.get_rng_state()
            th.manual_seed(4)
            self.current = RNNAgent(6, args)
            self.current_rng = th.random.get_rng_state()

    def test_layout_initialization_and_forward(self):
        self.assertFalse(hasattr(self.current, 'fc1'))
        self.assertTrue(th.equal(self.original_rng, self.current_rng))
        initial = self.current.legacy_initial_state(self.current.state_dict())
        for key, tensor in self.original.state_dict().items():
            th.testing.assert_close(tensor, initial[key], rtol=0, atol=0)
        hidden = self.current.double().init_hidden()
        self.assertEqual(hidden.shape, (1, 8))
        self.assertEqual(hidden.dtype, th.float64)
        self.assertEqual(hidden.device, self.current.net[0].weight.device)
        self.assertEqual(hidden.count_nonzero().item(), 0)
        x = th.randn(4, 6)
        h = th.randn(4, 8)
        for old, new in zip(self.original(x, h), self.current.float()(x, h)):
            th.testing.assert_close(new, old, rtol=0, atol=0)

    def test_strict_legacy_load_and_hash(self):
        state = self.original.state_dict()
        self.current.load_state_dict(state)
        self.assertIn('fc1.weight', state)
        self.assertNotIn('fc1.weight', self.current.state_dict())
        for invalid in (dict(state, surprise=state['fc1.bias']),
                        {key: value for key, value in state.items() if key != 'fc2.bias'},
                        {key: value for key, value in state.items() if key != 'fc1.bias'}):
            with self.assertRaises(RuntimeError):
                self.current.load_state_dict(invalid)
        archived = hash_model_states((state, state))
        fields = parent_hash_fields({'model_hash': archived}, (state, state))
        self.assertEqual(fields['archived_parent_model_hash'], archived)
        expected = hash_model_states((self.current.state_dict(), self.current.state_dict()))
        self.assertEqual(fields['parent_model_hash'], expected)
        self.assertEqual(parent_hash_fields({'model_hash': 'wrong'}, (state, state)),
                         {'parent_model_hash': 'wrong'})

    def test_rmsprop_state_migration_and_order_validation(self):
        old_mixer, new_mixer = nn.Linear(8, 1), nn.Linear(8, 1)
        old = th.optim.RMSprop([*self.original.parameters(), *old_mixer.parameters()], momentum=.9)
        new = th.optim.RMSprop([*self.current.parameters(), *new_mixer.parameters()])
        for i, parameter in enumerate(old.param_groups[0]['params']):
            if parameter.requires_grad and i >= 2:
                parameter.grad = th.full_like(parameter, i + 1.)
        old.step()
        state = old.state_dict()
        self.current.load_state_dict(self.original.state_dict())
        load_optimizer_state(new, state, self.current)
        self.assertEqual(new.param_groups[0]['momentum'], .9)
        for index, new_parameter in enumerate(new.param_groups[0]['params'], 2):
            for name, value in state['state'].get(index, {}).items():
                th.testing.assert_close(new.state[new_parameter][name], value)
        self.assertEqual(len(new.state), len(old.state))
        for malformed in ('order', 'groups', 'shape'):
            bad = copy.deepcopy(state)
            if malformed == 'order':
                ids = bad['param_groups'][0]['params']
                ids[2], ids[3] = ids[3], ids[2]
            elif malformed == 'groups':
                bad['param_groups'].append({'params': []})
            else:
                bad['state'][2]['square_avg'] = th.zeros(1)
            with self.subTest(malformed=malformed), self.assertRaisesRegex(ValueError, 'legacy fc1'):
                load_optimizer_state(new, bad, self.current)
