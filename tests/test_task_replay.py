"""Replay policy checks using complete padded episodes and CPU tensors."""
from collections import Counter
from io import BytesIO
from types import SimpleNamespace
import unittest

import torch as th

from components.episode_buffer import EpisodeBatch, ReplayBuffer
from components.task_replay import make_replay_buffer, restore_replay_buffer


SCHEME = {
    'state': {'vshape': (2,)},
    'actions': {'vshape': (1,), 'dtype': th.long},
    'terminated': {'vshape': (1,), 'dtype': th.uint8},
    'episode_id': {'vshape': (1,), 'dtype': th.long, 'episode_const': True},
}


def episode(task_id, number, length=5):
    batch = EpisodeBatch(SCHEME, {}, 1, length)
    for step in range(length):
        batch.update({'state': [[[task_id, step]]], 'actions': [[[step]]], 'terminated': [[[int(step == length - 2)]]]}, ts=step)
    batch.update({'episode_id': [[number]]}, mark_filled=False)
    return batch


def make(mode, order=('A', 'B', 'C', 'D'), capacity=24, batch_size=8):
    args = SimpleNamespace(buffer_size=capacity, batch_size=batch_size)
    if mode is not None:
        args.replay_on_switch = mode
    return make_replay_buffer(args, order, SCHEME, {}, 5, {})


class TaskReplayTest(unittest.TestCase):
    def assert_batch_intact(self, sample):
        for row in range(sample.batch_size):
            length = int(sample['filled'][row].sum())
            self.assertEqual(sample['state'][row, :length, 1].tolist(), list(range(length)))
            self.assertEqual(sample['actions'][row, :length, 0].tolist(), list(range(length)))
            self.assertEqual(sample['terminated'][row, :length, 0].nonzero().flatten().tolist(),
                             [length - 2])
            self.assertFalse(sample['filled'][row, length:].any())

    def test_default_uses_original_buffer(self):
        self.assertIs(type(make(None)), ReplayBuffer)
        self.assertIs(type(make('retain')), ReplayBuffer)

    def test_ft_switch_clears_old_episodes_and_warmup(self):
        buffer = make('ft', capacity=8)
        buffer.start_task('A')
        for number in range(11):
            buffer.insert_episode_batch(episode(1, number))
        self.assertEqual(sorted(buffer.sample(8)['episode_id'].flatten().tolist()), list(range(3, 11)))
        buffer.start_task('B')
        self.assertEqual(buffer.counts(), {'B': 0})
        for number in range(7):
            buffer.insert_episode_batch(episode(2, number))
        self.assertFalse(buffer.can_sample(8))
        buffer.insert_episode_batch(episode(2, 7, length=3))
        sample = buffer.sample(8)
        self.assertEqual(sample['state'][:, 0, 0].tolist(), [2] * 8)
        self.assert_batch_intact(sample)
        buffer.start_task('A')
        self.assertEqual(buffer.counts(), {'A': 0})

    def test_er_balances_history_with_capacity_revisit_and_short_episodes(self):
        buffer = make('er')
        for task_id, task, size in ((1, 'A', 30), (2, 'B', 1), (3, 'C', 12)):
            buffer.start_task(task)
            for number in range(size):
                buffer.insert_episode_batch(episode(task_id, number))
        self.assertEqual(buffer.counts(), {'A': 8, 'B': 1, 'C': 8})
        buffer.start_task('D')
        self.assertFalse(buffer.can_sample(8))
        for number in range(4):
            buffer.insert_episode_batch(episode(4, number, length=3))
        for _ in range(12):
            sample = buffer.sample(8)
            counts = Counter(sample['state'][:, 0, 0].int().tolist())
            self.assertEqual(counts[4], 4)
            self.assertEqual(sorted(counts[task_id] for task_id in (1, 2, 3)), [1, 1, 2])
            self.assert_batch_intact(sample)
        buffer.start_task('A')
        sample = buffer.sample(8)
        counts = Counter(sample['state'][:, 0, 0].int().tolist())
        self.assertEqual(counts[1], 4)
        self.assertTrue(set(sample['episode_id'][:4].flatten().tolist()).issubset(range(24, 30)))
        for number in range(30, 40):
            buffer.insert_episode_batch(episode(1, number))
        self.assertEqual(buffer.counts(), {'A': 6, 'B': 1, 'C': 6, 'D': 4})
        occupied = [slot for slots in buffer.slots.values() for slot in slots]
        self.assertEqual(sorted(occupied + list(buffer.free_slots)), list(range(24)))
        self.assertEqual(len(occupied), buffer.episodes_in_buffer)

    def test_restore_legacy_single_task_and_round_trip_tagged_snapshot(self):
        buffer = make('er', order=('A', 'B', 'C'), capacity=9, batch_size=6)
        old = make(None, capacity=9, batch_size=6)
        for number in range(14):
            old.insert_episode_batch(episode(1, number))
        with self.assertRaisesRegex(ValueError, 'single-task parent'):
            restore_replay_buffer(buffer, old, ['A', 'B'])
        buffer = restore_replay_buffer(buffer, old, ['A'])
        self.assertEqual(buffer.counts(), {'A': 9})
        stream = BytesIO()
        th.save(buffer, stream)
        stream.seek(0)
        restored = restore_replay_buffer(
            make('er', order=('A', 'B', 'C'), capacity=9, batch_size=6),
            th.load(stream, map_location='cpu'), ['A'],
        )
        restored.start_task('B')
        self.assertEqual(restored.counts(), {'A': 5, 'B': 0})
        for number in range(3):
            restored.insert_episode_batch(episode(2, number))
        sample = restored.sample(6)
        self.assertEqual(sample['state'][:, 0, 0].tolist(), [2] * 3 + [1] * 3)
        self.assertTrue(set(sample['episode_id'][3:].flatten().tolist()).issubset(range(9, 14)))

    def test_er_first_task_uses_full_batch_and_rejects_impossible_capacity(self):
        buffer = make('er')
        buffer.start_task('A')
        for number in range(7):
            buffer.insert_episode_batch(episode(1, number))
        self.assertFalse(buffer.can_sample(8))
        buffer.insert_episode_batch(episode(1, 7))
        self.assert_batch_intact(buffer.sample(8))
        self.assertEqual(buffer.last_sample_counts, {'A': 8})
        with self.assertRaisesRegex(ValueError, 'even batch size'):
            make('er', batch_size=7)
        with self.assertRaisesRegex(ValueError, 'room for each task'):
            make('er', capacity=12)


if __name__ == '__main__':
    unittest.main()
