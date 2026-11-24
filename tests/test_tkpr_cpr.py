"""CPR decision rules, replay masking and persistent-reference checks (no environment)."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'drivers/continual'))

from drivers.continual.memory import History, PreferenceHistory
from drivers.continual.memory.cpr import initialize_pool, select_reference, update_pool, restore_history
from drivers.continual.action_set_protocol import CPR_DEFAULTS, validate
from learners.action_set_preference import batch_episodes, decision_masks, unroll
from learners.ranking_preference import RankingPreference, topk_preference


SCHEME = dict(obs=dict(vshape=1, group='agents'),
              actions_onehot=dict(vshape=(3,), group='agents'),
              avail_actions=dict(vshape=(3,), group='agents', dtype=th.int),
              terminated=dict(vshape=(1,), dtype=th.uint8), filled=dict(vshape=(1,), dtype=th.long))
GROUPS = dict(agents=2)


def episode(length):
    data = dict(obs=th.ones(length + 1, 2, 1), actions_onehot=th.zeros(length + 1, 2, 3),
                avail_actions=th.ones(length + 1, 2, 3, dtype=th.int),
                terminated=th.zeros(length + 1, 1, dtype=th.uint8),
                filled=th.ones(length + 1, 1, dtype=th.long),
                old_q_values=th.tensor([.05, 0., -1.]).expand(length + 1, 2, 3).clone())
    data['actions_onehot'][..., 0] = 1
    data['terminated'][length - 1] = 1
    return dict(length=length, data=data)


def pool():
    return initialize_pool(dict(task='A', phase=0, reference_checkpoint='models/phase0000/agent.th',
                                episodes=[episode(2), episode(3)]))


class MAC:
    args = SimpleNamespace(device='cpu')

    def __init__(self):
        self.agent = th.nn.Linear(1, 3, bias=False)
        with th.no_grad():
            self.agent.weight[:, 0] = th.tensor([2., 0., -1.])
        self.hidden_states = th.ones(1, 2, 1)
        self.calls = []

    def init_hidden(self, batch_size):
        self.hidden_states = th.zeros(batch_size, 2, 1)

    def forward(self, batch, t):
        self.calls.append((t, th.is_grad_enabled()))
        self.hidden_states = self.hidden_states + 1
        return self.agent(batch['obs'][:, t])


def q_at_confidence(confidence, reverse=False):
    # Three legal actions [x,0,-1]: solve gap/std for a desired confidence.
    low, high = 0., 100.
    for _ in range(80):
        x = (low + high) / 2
        if x / np.std([x, 0., -1.]) < confidence:
            low = x
        else:
            high = x
    q = [(low + high) / 2, 0., -1.]
    return [q[1], q[0], q[2]] if reverse else q


class CPRTest(unittest.TestCase):
    def test_state_selection_update_keep_conflict_and_reactivation(self):
        cases = [ # old conf, new conf, top differs, old mask, expected
            (.8, 1.3, True, True, 'update'),
            (1.3, .8, True, True, 'keep'),
            (.8, .85, True, True, 'uncertain'),
            (.2, .3, False, True, 'uncertain'),
            (.8, .85, False, True, 'keep'),
            (.8, 1.0, False, True, 'update'),
            (1.3, .6, True, False, 'update'),
            (.8, .2, True, False, 'uncertain'),
            (.8, .2, True, True, 'keep'),
            (.51, .49, True, True, 'uncertain'),
            (.49, .55, False, True, 'keep'),
            (.49, .55, True, True, 'uncertain'),
        ]
        old = th.tensor([q_at_confidence(x[0]) for x in cases], dtype=th.float64)
        new = th.tensor([q_at_confidence(x[1], x[2]) for x in cases], dtype=th.float64)
        valid = th.tensor([x[3] for x in cases])
        legal = th.ones_like(old, dtype=th.bool)
        decisions = th.ones(len(cases), dtype=th.bool)
        masks = select_reference(old, new, legal, decisions, valid, CPR_DEFAULTS)
        for index, case in enumerate(cases):
            self.assertEqual([name for name, mask in zip(('update', 'keep', 'uncertain'), masks[:3])
                              if mask[index]], [case[4]])
        # Padding/nondecisions never mutate a reference; one-legal-action/dead rows excluded.
        decisions[:] = False
        self.assertFalse(any(mask.any() for mask in select_reference(
            old, new, legal, decisions, valid, CPR_DEFAULTS)[:3]))
        legal[:, 1:] = False
        decisions[:] = True
        self.assertFalse(any(mask.any() for mask in select_reference(
            old, new, legal, decisions, valid, CPR_DEFAULTS)[:3]))

    def test_chunked_update_gate_failure_and_save_restore_preserve_accepted_targets(self):
        memory = pool()
        original = copy.deepcopy(memory)
        mac = MAC()
        hidden = mac.hidden_states
        parameters = copy.deepcopy(mac.agent.state_dict())
        update = update_pool(mac, memory, SCHEME, GROUPS, CPR_DEFAULTS, 1, .8, .8,
                             'models/phase0001/agent.th', chunk_size=1)
        self.assertEqual(update['reevaluated_episodes'], 2)
        self.assertEqual(update['updated_states'], 10)
        self.assertEqual(update['decision_states'], 10)
        self.assertEqual([t for t, _ in mac.calls], [0, 1, 0, 1, 2])
        self.assertTrue(all(not enabled for _, enabled in mac.calls))
        self.assertIs(mac.hidden_states, hidden)
        for before, after in zip(original['episodes'], memory['episodes']):
            th.testing.assert_close(before['original_q_values'], after['original_q_values'])
            for key in before['data']:
                if key != 'old_q_values':
                    self.assertTrue(th.equal(before['data'][key], after['data'][key]))
            th.testing.assert_close(after['data']['old_q_values'][-1], before['data']['old_q_values'][-1])
            self.assertTrue((after['reference_source_phase'][:-1] == 1).all())
            self.assertTrue((after['reference_source_phase'][-1] == 0).all())
        for key, value in parameters.items():
            th.testing.assert_close(value, mac.agent.state_dict()[key])
        self.assertTrue(all(p.grad is None for p in mac.agent.parameters()))
        memory['episodes'][0]['reference_valid_mask'][0, 0] = False
        memory['episodes'][0]['reference_source_phase'][0, 0] = -1
        accepted = copy.deepcopy(memory)
        with patch.object(mac, 'forward', side_effect=AssertionError('failed gate evaluated MAC')):
            failed = update_pool(mac, memory, SCHEME, GROUPS, CPR_DEFAULTS, 2, .4, .8,
                                 'models/phase0002/agent.th', chunk_size=1)
        self.assertFalse(failed['task_gate_passed'])
        self.assertEqual(failed['reevaluated_episodes'], 0)
        for a, b in zip(accepted['episodes'], memory['episodes']):
            for key in ('original_q_values', 'reference_valid_mask', 'reference_source_phase'):
                self.assertTrue(th.equal(a[key], b[key]))
            self.assertTrue(th.equal(a['data']['old_q_values'], b['data']['old_q_values']))
        with tempfile.TemporaryDirectory() as tmp:
            parent, child = Path(tmp)/'parent', Path(tmp)/'child'
            path = PreferenceHistory.phase_path(parent, 0)/'success.th'
            History.save(path, memory)
            restored = PreferenceHistory()
            restore_history(restored, parent, [0], child)
            ep = restored[0]['episodes'][0]
            self.assertEqual(ep['reference_valid_mask'].shape, (3, 2))
            self.assertEqual(ep['original_q_values'].shape, (3, 2, 3))
            self.assertTrue(th.equal(ep['data']['old_q_values'], memory['episodes'][0]['data']['old_q_values']))
            self.assertTrue(th.equal(ep['reference_source_phase'], memory['episodes'][0]['reference_source_phase']))
            self.assertTrue(th.equal(ep['reference_valid_mask'], memory['episodes'][0]['reference_valid_mask']))
            self.assertNotEqual(ep['data']['old_q_values'].data_ptr(), ep['original_q_values'].data_ptr())
            self.assertTrue((PreferenceHistory.phase_path(child, 0)/'success.th').exists())
            self.assertEqual(restored[0]['reference_checkpoints'][1], str(parent/'models/phase0001/agent.th'))
        invalid = pool()
        invalid['episodes'][0]['reference_valid_mask'] = th.ones(2, 2, dtype=th.bool)
        with self.assertRaisesRegex(ValueError, 'CPR metadata'):
            initialize_pool(invalid)

    def test_loss_mask_uses_exact_k1_loss_and_sampling_remains_total_eight(self):
        memory = pool()
        mac = MAC()
        cpr = RankingPreference('tkpr_cpr', dict(CPR_DEFAULTS, batch_episodes=8),
                                np.random.default_rng(3), SCHEME, GROUPS, None)
        # Low-confidence old references produce valid min_gap pairs; disable one complete episode.
        memory['episodes'][0]['reference_valid_mask'][:] = False
        memory['episodes'][0]['reference_source_phase'][:] = -1
        loss, count, _ = cpr.task_loss(mac, memory, memory['episodes'])
        batch, _ = batch_episodes(memory['episodes'], SCHEME, GROUPS, 'cpu', include_good=False, include_old_q=True)
        legal, decisions = decision_masks(batch)
        valid = th.nn.utils.rnn.pad_sequence([ep['reference_valid_mask'] for ep in memory['episodes']], batch_first=True)
        expected = topk_preference(unroll(mac, batch, batch.max_seq_length - 1),
                                  batch['old_q_values'][:, :-1], legal, decisions & valid[:, :-1], CPR_DEFAULTS)
        th.testing.assert_close(loss, expected[0])
        self.assertEqual(count.item(), expected[1].item())
        self.assertEqual(count.item(), 6)
        actual_grad = th.autograd.grad(loss, tuple(mac.agent.parameters()), retain_graph=True)
        expected_grad = th.autograd.grad(expected[0], tuple(mac.agent.parameters()))
        th.testing.assert_close(actual_grad, expected_grad)
        for i, task in enumerate('ABCD'):
            cpr.history[i] = dict(memory, task=task, phase=i)
        cpr.task = 'E'
        self.assertEqual(sum(len(ids) for _, ids in cpr.sample()), 8)

    def test_conflict_disables_reference_and_later_reactivates_without_losing_original(self):
        memory = pool()
        mac = MAC()
        for ep in memory['episodes']:
            ep['data']['old_q_values'][:] = th.tensor(q_at_confidence(.8))
        # Start from a newly collected pool so original and actual references agree.
        for ep in memory['episodes']:
            for field in ('original_q_values', 'reference_valid_mask', 'reference_source_phase'):
                del ep[field]
        initialize_pool(memory)
        original = copy.deepcopy(memory)
        with th.no_grad():
            mac.agent.weight[:, 0] = th.tensor(q_at_confidence(.85))
        kept = update_pool(mac, memory, SCHEME, GROUPS, CPR_DEFAULTS, 1, .8, .8, 'phase1')
        self.assertEqual(kept['retained_states'], 10)
        self.assertTrue(all((ep['reference_source_phase'] == 0).all() for ep in memory['episodes']))
        with th.no_grad():
            mac.agent.weight[:, 0] = th.tensor(q_at_confidence(.85, reverse=True))
        conflict = update_pool(mac, memory, SCHEME, GROUPS, CPR_DEFAULTS, 2, .8, .8, 'phase2')
        self.assertEqual(conflict['uncertain_states'], 10)
        for ep in memory['episodes']:
            self.assertFalse(ep['reference_valid_mask'][:-1].any())
            self.assertTrue((ep['reference_source_phase'][:-1] == -1).all())
        reactivated = update_pool(mac, memory, SCHEME, GROUPS, CPR_DEFAULTS, 3, .8, .8, 'phase3')
        self.assertEqual(reactivated['updated_states'], 10)
        for before, after in zip(original['episodes'], memory['episodes']):
            self.assertTrue(after['reference_valid_mask'][:-1].all())
            self.assertTrue((after['reference_source_phase'][:-1] == 3).all())
            self.assertTrue(th.equal(before['original_q_values'], after['original_q_values']))

    def test_ts5_configuration_matches_k1_except_cpr_options(self):
        root = Path(__file__).resolve().parents[1]
        base = json.loads((root/'configs/continual/action_preference/ts5/tkpr_k1.json').read_text())
        raw = json.loads((root/'configs/continual/action_preference/ts5/tkpr_cpr.json').read_text())
        effective = validate(raw, 'vdn_1ff', 'tkpr_cpr')
        self.assertEqual(effective['tkpr_cpr'], CPR_DEFAULTS)
        self.assertEqual(effective['action_set']['batch_episodes'], 8)
        for forbidden in ('batch_episodes_per_task', 'refresh_success_episodes'):
            self.assertNotIn(forbidden, effective['action_set'])
        base.pop('id'); raw.pop('id')
        old = base.pop('topk_preference'); new = raw.pop('tkpr_cpr')
        self.assertEqual(base, raw)
        self.assertTrue(all(new[key] == value for key, value in old.items()))
        raw['tkpr_cpr'] = dict(new, k=2)
        with self.assertRaisesRegex(ValueError, 'k must be 1'):
            validate(raw, 'vdn_1ff', 'tkpr_cpr')


if __name__ == '__main__':
    unittest.main()
