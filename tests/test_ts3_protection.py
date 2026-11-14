"""Offline checks for CHI protection updates and recovery."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'drivers/continual'))
from ts3_protection import TS3Protection


def protector(mode):
    learner = SimpleNamespace(args=SimpleNamespace(rnn_hidden_dim=4),
                              optimiser=SimpleNamespace(step=Mock()))
    return TS3Protection(learner, 'A', mode, Mock(), recovery_steps=100)


def selection(ids, **fields):
    return {'A': dict(sets={'overall': ids}, **fields)}


class TS3ProtectionTest(unittest.TestCase):
    def test_chi_exception_preserves_last_selection_and_recovery(self):
        protection = protector('chi')
        protection.update(selection([0, 1], defined=True), 10)
        protection.update(selection([1], defined=True), 20)
        recovering = deepcopy(protection.schedule.recovering)
        for step, expected_scale in ((70, .55), (130, 1.)):
            with self.subTest(step=step):
                row = protection.update(selection(None, defined=False, skip_reason='detection_error'), step)
                self.assertEqual(protection.schedule.protected['A'], {1})
                self.assertEqual(protection.schedule.recovering, recovering)
                self.assertFalse(row['detection_defined'])
                self.assertFalse(row['protection_updated'])
                self.assertEqual(row['skip_reason'], 'detection_error')
                self.assertEqual(row['protected_before'], [1])
                self.assertEqual(row['protected_after'], [1])
                self.assertAlmostEqual(row['unit_scales']['A'][0], expected_scale)

    def test_successful_empty_chi_selection_releases_neurons(self):
        protection = protector('chi')
        protection.update(selection([1], defined=True), 10)
        row = protection.update(selection([], defined=True), 20)
        self.assertTrue(row['detection_defined'])
        self.assertTrue(row['protection_updated'])
        self.assertIsNone(row['skip_reason'])
        self.assertEqual(row['protected_before'], [1])
        self.assertEqual(row['protected_after'], [])
        self.assertEqual(protection.schedule.recovering['A'][1]['start'], 20)
        self.assertAlmostEqual(protection.schedule.values(70)['A'][1], .55)
        self.assertEqual(protection.schedule.values(120)['A'][1], 1.)

    def test_chi_union_failure_preserves_historical_list(self):
        protection = protector('chi_union')
        protection.update(selection([0], defined=True), 10)
        protection.update(selection([2], defined=True), 20)
        # An explicitly undefined result takes precedence over any supplied list.
        row = protection.update(selection([3], defined=False), 30)
        self.assertFalse(row['protection_updated'])
        self.assertEqual(row['skip_reason'], 'detection_undefined')
        self.assertEqual(row['protected_before'], [0, 2])
        self.assertEqual(row['protected_after'], [0, 2])
        self.assertEqual(protection.schedule.recovering['A'], {})
        self.assertTrue(protection.update(selection([], defined=True), 40)['protection_updated'])
        self.assertEqual(protection.schedule.protected['A'], {0, 2})

    def test_legacy_non_chi_modes_ignore_defined_flag(self):
        for mode, expected in [('linear', {2}), ('union', {0, 2})]:
            with self.subTest(mode=mode):
                protection = protector(mode)
                protection.update(selection([0]), 10)
                row = protection.update(selection([2], defined=False), 20)
                self.assertTrue(row['detection_defined'])
                self.assertNotIn('protection_updated', row)
                protection.update(selection(None), 30)
                self.assertEqual(protection.schedule.protected['A'], expected)


if __name__ == '__main__':
    unittest.main()
