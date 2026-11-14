"""Episode layout and phase bookkeeping shared by continual drivers."""
import os
import random
import time

import numpy as np
import torch as th
from components.transforms import OneHot

from .records import append, write


def episode_spec(info):
    scheme = dict(state=dict(vshape=info['state_shape']),
                  obs=dict(vshape=info['obs_shape'], group='agents'),
                  actions=dict(vshape=(1,), group='agents', dtype=th.long),
                  avail_actions=dict(vshape=(info['n_actions'],), group='agents', dtype=th.int),
                  reward=dict(vshape=(1,)), terminated=dict(vshape=(1,), dtype=th.uint8))
    groups = dict(agents=info['n_agents'])
    preprocess = dict(actions=('actions_onehot', [OneHot(out_dim=info['n_actions'])]))
    return scheme, groups, preprocess


def rng_state():
    return (random.getstate(), np.random.get_state(), th.get_rng_state(),
            th.cuda.get_rng_state_all() if th.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    th.set_rng_state(state[2].cpu())
    if state[3]:
        th.cuda.set_rng_state_all([value.cpu() for value in state[3]])


def log_task_epsilon(runner, out, phase, task, event):
    selector = runner.mac.action_selector
    mode = runner.args.epsilon_on_switch
    if event == 'start' and mode == 'per_task':
        selector.reset_exploration(runner.t_env)
    append(out / 'task_epsilon.jsonl', dict(
        event=event, phase=phase, task=task, t_env=runner.t_env,
        exploration_steps=selector.exploration_steps(runner.t_env),
        epsilon=selector.training_epsilon(runner.t_env), schedule=mode,
    ))


def begin_environment_audit(runner, out, phase, *, all_resets=False):
    first = True

    def audit(row):
        nonlocal first
        if first or all_resets:
            append(out / 'environment_audit.jsonl', dict(phase=phase, t_env=runner.t_env, **row))
            first = False

    runner.env.audit = audit


def log_phase_end(out, phase, task, begin, end, actual_end):
    append(out / 'phase_end.jsonl', dict(
        phase=phase, task=task, actual_start=begin, actual_end=actual_end,
        nominal_end=end, overshoot=actual_end-end))


def td_sample(buffer, batch_size, device):
    sample = buffer.sample(batch_size)
    sample = sample[:, :int(sample.max_t_filled())]
    sample.to(device)
    return sample



def write_sequence_status(out, state, t_env, phase, task, updates, episode, started, **extra):
    write(out / 'status.json', dict(
        status=state, pid=os.getpid(), t_env=t_env, phase=phase, task=task,
        updates=updates, episode=episode, elapsed_seconds=time.time()-started, **extra))
