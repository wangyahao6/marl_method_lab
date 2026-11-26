"""Targeted offline tests for original-Top-1 anchored reference replay."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch as th

from test_tkpr_cpr import MAC, SCHEME, GROUPS, pool, q_at_confidence
from drivers.continual.action_set_protocol import CPR_DEFAULTS, validate
from drivers.continual.memory import History, PreferenceHistory
from drivers.continual.memory.cpr import restore_history
from drivers.continual.memory.cpr_anchor import select_reference, update_pool
from learners.ranking_preference import RankingPreference


class CPRAnchorTest(unittest.TestCase):
    def test_original_top1_anchor_threshold_margin_and_legal_actions(self):
        # Original and current reference disagree: compare to ORIGINAL, not current reference.
        cases = [ # old confidence, new confidence, new differs from original, valid, accepts
            (.8, 1.3, True, True, False),
            (.8, 1.3, False, True, True),
            (.8, .85, False, True, False),
            (.2, .6, False, False, False), # min_gap=.7 dominates confidence threshold=.5
            (.8, .85, False, False, False), # invalid does not bypass switch margin
        ]
        original = th.tensor([q_at_confidence(.8) for _ in cases], dtype=th.float64)
        old = th.tensor([q_at_confidence(c[0], reverse=True) for c in cases], dtype=th.float64)
        new = th.tensor([q_at_confidence(c[1], reverse=c[2]) for c in cases], dtype=th.float64)
        legal = th.ones_like(old, dtype=th.bool)
        decisions = th.ones(len(cases), dtype=th.bool)
        valid = th.tensor([c[3] for c in cases])
        masks = select_reference(old, new, legal, decisions, valid,
                                 dict(CPR_DEFAULTS, min_gap=.7), original_q=original)
        self.assertEqual(masks[0].tolist(), [c[4] for c in cases])
        self.assertTrue(th.equal(masks[1], ~masks[0]))
        self.assertFalse(masks[2].any())
        # Illegal high-Q action cannot become the original anchor.
        original = th.cat((original, th.full((len(cases), 1), 1000.)), dim=-1)
        old = th.cat((old, th.zeros(len(cases), 1)), dim=-1)
        new = th.cat((new, th.zeros(len(cases), 1)), dim=-1)
        legal = th.cat((legal, th.zeros(len(cases), 1, dtype=th.bool)), dim=-1)
        anchored = select_reference(old, new, legal, decisions, valid,
                                    CPR_DEFAULTS, original_q=original)
        self.assertTrue(anchored[0][1])
        decisions[:] = False
        self.assertFalse(any(m.any() for m in select_reference(
            old, new, legal, decisions, valid, CPR_DEFAULTS, original_q=original)[:3]))
        decisions[:] = True
        legal[:, 1:] = False
        self.assertFalse(any(m.any() for m in select_reference(
            old, new, legal, decisions, valid, CPR_DEFAULTS, original_q=original)[:3]))

    def test_accepted_update_preserves_original_and_interactions_and_roundtrips(self):
        memory, mac = pool(), MAC()
        before = copy.deepcopy(memory)
        hidden = mac.hidden_states
        update = update_pool(mac, memory, SCHEME, GROUPS, CPR_DEFAULTS, 2, .8, .8, 'phase2', chunk_size=1)
        self.assertEqual(update['updated_states'], 10)
        self.assertEqual(update['uncertain_states'], 0)
        self.assertIs(mac.hidden_states, hidden)
        self.assertTrue(all(not enabled for _, enabled in mac.calls))
        self.assertTrue(all(p.grad is None for p in mac.agent.parameters()))
        for a, b in zip(before['episodes'], memory['episodes']):
            self.assertTrue(th.equal(a['original_q_values'], b['original_q_values']))
            self.assertFalse(th.equal(a['data']['old_q_values'][:-1], b['data']['old_q_values'][:-1]))
            self.assertTrue((b['reference_source_phase'][:-1] == 2).all())
            self.assertTrue(th.equal(a['reference_source_phase'][-1], b['reference_source_phase'][-1]))
            for key in a['data']:
                if key != 'old_q_values':
                    self.assertTrue(th.equal(a['data'][key], b['data'][key]))
        with tempfile.TemporaryDirectory() as tmp:
            parent, child = Path(tmp)/'parent', Path(tmp)/'child'
            History.save(PreferenceHistory.phase_path(parent, 0)/'success.th', memory)
            restored = PreferenceHistory()
            restore_history(restored, parent, [0], child)
            self.assert_references_equal(memory, restored[0])

    def assert_references_equal(self, before, after):
        for a, b in zip(before['episodes'], after['episodes']):
            for field in ('original_q_values', 'reference_valid_mask', 'reference_source_phase'):
                self.assertTrue(th.equal(a[field], b[field]), field)
            self.assertTrue(th.equal(a['data']['old_q_values'], b['data']['old_q_values']))

    def test_delayed_update_keeps_existing_masks_sources_and_q(self):
        memory, mac = pool(), MAC()
        for ep in memory['episodes']:
            ep['data']['old_q_values'][:] = th.tensor(q_at_confidence(.8))
            ep['reference_valid_mask'][0, 0] = False
            ep['reference_source_phase'][0, 0] = -1
        before = copy.deepcopy(memory)
        # Conflict, too-small improvement and low-confidence candidates all preserve masks.
        for confidence, reverse in ((1.3, True), (.85, False), (.3, False)):
            with th.no_grad():
                mac.agent.weight[:, 0] = th.tensor(q_at_confidence(confidence, reverse))
            result = update_pool(mac, memory, SCHEME, GROUPS, CPR_DEFAULTS, 3, .8, .8, 'phase3')
            self.assertEqual(result['updated_states'], 0)
            self.assertEqual(result['retained_states'], 10)
            self.assertEqual(result['uncertain_states'], 0)
            self.assert_references_equal(before, memory)

    def test_task_gate_failure_preserves_previously_accepted_reference(self):
        memory, mac = pool(), MAC()
        update_pool(mac, memory, SCHEME, GROUPS, CPR_DEFAULTS, 1, .8, .8, 'phase1')
        before = copy.deepcopy(memory)
        with patch.object(mac, 'forward', side_effect=AssertionError('gate must skip online MAC')):
            result = update_pool(mac, memory, SCHEME, GROUPS, CPR_DEFAULTS, 2, .4, .8, 'phase2')
        self.assertFalse(result['task_gate_passed'])
        self.assertEqual(result['reevaluated_episodes'], 0)
        self.assert_references_equal(before, memory)
        self.assertEqual(before['reference_checkpoints'], memory['reference_checkpoints'])

    def test_configs_and_loss_and_sampling_match_original_cpr(self):
        root = Path(__file__).resolve().parents[1]
        for suffix in ('', '_1m'):
            folder = root/'configs/continual/action_preference/ts5'
            old = json.loads((folder/f'tkpr_cpr{suffix}.json').read_text())
            anchor = json.loads((folder/f'tkpr_cpr_anchor{suffix}.json').read_text())
            effective = validate(anchor, 'vdn_1ff', 'tkpr_cpr_anchor')
            self.assertEqual(effective['tkpr_cpr_anchor'], CPR_DEFAULTS)
            self.assertEqual(effective['action_set']['batch_episodes'], 8)
            self.assertNotIn('batch_episodes_per_task', effective['action_set'])
            self.assertNotIn('refresh_success_episodes', effective['action_set'])
            old.pop('id'); anchor.pop('id')
            self.assertEqual(old.pop('tkpr_cpr'), anchor.pop('tkpr_cpr_anchor'))
            self.assertEqual(old, anchor)
        memory, mac = pool(), MAC()
        memory['episodes'][0]['reference_valid_mask'][0] = False
        preferences = [RankingPreference(name, dict(CPR_DEFAULTS, batch_episodes=8),
                                        np.random.default_rng(3), SCHEME, GROUPS, None)
                       for name in ('tkpr_cpr', 'tkpr_cpr_anchor')]
        outputs = [p.task_loss(mac, memory, memory['episodes']) for p in preferences]
        th.testing.assert_close(outputs[0][0], outputs[1][0])
        self.assertEqual(outputs[0][1].item(), outputs[1][1].item())
        gradients = [th.autograd.grad(output[0], tuple(mac.agent.parameters())) for output in outputs]
        th.testing.assert_close(gradients[0], gradients[1])
        samples = []
        for preference in preferences:
            for i, task in enumerate('ABCD'):
                preference.history[i] = dict(memory, task=task, phase=i)
            preference.task = 'E'
            sample = preference.sample()
            self.assertEqual(sum(len(ids) for _, ids in sample), 8)
            samples.append([(phase, list(ids)) for phase, ids in sample])
        self.assertEqual(samples[0], samples[1])


if __name__ == '__main__':
    unittest.main()
