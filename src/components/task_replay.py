"""Task-aware episodic replay for continual VDN baselines."""
from collections import deque

import numpy as np

from components.episode_buffer import ReplayBuffer


class TaskReplayBuffer(ReplayBuffer):
    """Use the existing tensor storage with task-specific slot ownership."""

    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess, mode, order):
        super().__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess)
        self.mode = mode
        self.order = tuple(dict.fromkeys(order))
        self.task = None
        self.slots = {}
        self.free_slots = deque(range(buffer_size)) if mode == 'er' else None
        self.last_sample_counts = {}

    def start_task(self, task):
        self.last_sample_counts = {}
        if task == self.task:
            return
        self.task = task
        if self.mode == 'ft':
            self.buffer_index = 0
            self.episodes_in_buffer = 0
            return
        self.slots.setdefault(task, deque())
        for owner, quota in self._quotas().items():
            slots = self.slots[owner]
            while len(slots) > quota:
                self.free_slots.append(slots.popleft())
                self.episodes_in_buffer -= 1

    def _quotas(self):
        owners = [task for task in self.order if task == self.task or self.slots.get(task)]
        base, extra = divmod(self.buffer_size, len(owners))
        return {task: base + (index < extra) for index, task in enumerate(owners)}

    def counts(self):
        if self.mode == 'ft':
            return {self.task: self.episodes_in_buffer}
        return {task: len(slots) for task, slots in self.slots.items()}

    def insert_episode_batch(self, batch):
        if self.mode == 'ft' and batch.max_seq_length == self.max_seq_length:
            super().insert_episode_batch(batch)
            return
        if self.mode == 'er':
            slots = self.slots[self.task]
            quota = self._quotas()[self.task]
        for row in range(batch.batch_size):
            if self.mode == 'ft':
                slot = self.buffer_index
                self.buffer_index = (slot + 1) % self.buffer_size
                self.episodes_in_buffer = min(self.buffer_size, self.episodes_in_buffer + 1)
            else:
                if len(slots) == quota:
                    slot = slots.popleft()
                else:
                    slot = self.free_slots.popleft()
                    self.episodes_in_buffer += 1
                slots.append(slot)
            length = batch.max_seq_length
            for key, target in self.data.transition_data.items():
                target[slot, :length].copy_(batch[key][row])
                target[slot, length:].zero_()
            for key, target in self.data.episode_data.items():
                target[slot].copy_(batch[key][row])

    def _historical(self):
        return [task for task, slots in self.slots.items() if task != self.task and slots]

    def can_sample(self, batch_size):
        if self.mode == 'ft':
            return super().can_sample(batch_size)
        required = batch_size // 2 if self._historical() else batch_size
        return len(self.slots[self.task]) >= required

    def sample(self, batch_size):
        if self.mode == 'ft':
            batch = super().sample(batch_size)
            self.last_sample_counts = {self.task: batch_size}
            return batch
        if not self.can_sample(batch_size):
            raise ValueError('Not enough current-task episodes to sample a replay batch')
        historical = self._historical()
        count = batch_size // 2 if historical else batch_size
        counts = {self.task: count}
        if historical:
            share, extra = divmod(batch_size - count, len(historical))
            for index, task in enumerate(np.random.permutation(historical).tolist()):
                quota = share + (index < extra)
                if quota:
                    counts[task] = quota
        indices = []
        for task, count in counts.items():
            slots = self.slots[task]
            # Fill a small task's quota without discarding unique episodes.
            unique = min(count, len(slots))
            chosen = (list(range(unique)) if count == len(slots) else
                      np.random.choice(len(slots), unique, replace=False).tolist())
            if count > unique:
                chosen += np.random.choice(len(slots), count - unique).tolist()
            indices.extend(slots[index] for index in chosen)
        self.last_sample_counts = counts
        return self[indices]


def make_replay_buffer(args, order, scheme, groups, max_seq_length, preprocess):
    mode = getattr(args, 'replay_on_switch', 'retain')
    if mode == 'retain':
        return ReplayBuffer(scheme, groups, args.buffer_size, max_seq_length, preprocess=preprocess)
    if mode not in ('ft', 'er'):
        raise ValueError('replay_on_switch must be retain, ft or er')
    if args.buffer_size < args.batch_size:
        raise ValueError('Replay capacity must hold a full batch')
    if mode == 'er':
        tasks = len(set(order))
        if (args.batch_size < 2 or args.batch_size % 2
                or args.buffer_size // tasks < args.batch_size // 2):
            raise ValueError('VDN-ER needs an even batch size and room for each task to supply half a batch')
    return TaskReplayBuffer(scheme, groups, args.buffer_size, max_seq_length, preprocess, mode, order)


def restore_replay_buffer(buffer, saved, previous_tasks):
    """Recover task ownership from new snapshots or a known single-task legacy prefix."""
    if isinstance(saved, TaskReplayBuffer):
        if not isinstance(buffer, TaskReplayBuffer) or saved.mode != buffer.mode:
            raise ValueError('Use the same replay mode as the parent snapshot')
        if saved.buffer_size != buffer.buffer_size:
            raise ValueError('Use the same replay capacity as the parent snapshot')
        if saved.scheme != buffer.scheme or saved.max_seq_length != buffer.max_seq_length:
            raise ValueError('Parent replay schema differs from the current environment')
        if saved.mode == 'er' and not set(saved.slots).issubset(buffer.order):
            raise ValueError('Parent replay contains tasks missing from the current sequence')
        saved.order = buffer.order
        return saved
    if not isinstance(buffer, TaskReplayBuffer):
        return saved
    tasks = set(previous_tasks)
    if len(tasks) != 1:
        if buffer.mode == 'er':
            raise ValueError('VDN-ER needs task-tagged replay or a known single-task parent')
        return buffer
    if saved.buffer_size != buffer.buffer_size:
        raise ValueError('Use the same replay capacity as the parent snapshot')
    if saved.scheme != buffer.scheme or saved.max_seq_length != buffer.max_seq_length:
        raise ValueError('Parent replay schema differs from the current environment')
    task, = tasks
    buffer.start_task(task)
    buffer.data = saved.data
    buffer.episodes_in_buffer = saved.episodes_in_buffer
    buffer.buffer_index = saved.buffer_index
    if buffer.mode == 'er':
        count = saved.episodes_in_buffer
        indices = range(count)
        if count == saved.buffer_size:
            indices = (*range(saved.buffer_index, count), *range(saved.buffer_index))
        buffer.slots[task].extend(indices)
        buffer.free_slots = deque(range(count, buffer.buffer_size))
    return buffer


def start_replay_task(buffer, task):
    if isinstance(buffer, TaskReplayBuffer):
        buffer.start_task(task)
        return {'replay_mode': buffer.mode, 'replay_task_episodes': buffer.counts()}
    return {}


def log_first_replay_batch(buffer, append, out, phase, task, t_env):
    append(out / 'replay_samples.jsonl', dict(
        phase=phase, task=task, t_env=t_env, mode=buffer.mode,
        sampled_task_episodes=buffer.last_sample_counts))
