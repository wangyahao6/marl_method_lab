"""Capture and compare isolated, paired learner updates."""

from copy import deepcopy
from time import perf_counter
from types import SimpleNamespace

import torch as th

from components.episode_buffer import EpisodeBatch
from .checkpoints import cpu_snapshot, restore_networks_and_optimizer


def capture_update(out, learner, batch, preference, phase, task, part, t_env, episode, update):
    directory = out / "timing_snapshots"
    directory.mkdir(exist_ok=True)
    path = directory / f"phase{phase:04d}_quarter{part}.th"
    state = {
        name: cpu_snapshot(network)
        for name, network in (("agent", learner.mac.agent), ("target_agent", learner.target_mac.agent),
                              ("mixer", learner.mixer), ("target_mixer", learner.target_mixer))
    }
    data = {name: {key: value.detach().cpu().clone() for key, value in tensors.items()}
            for name, tensors in vars(batch.data).items()}
    th.save(dict(state, optimizer=learner.optimiser.state_dict(), data=data,
                 batch_size=batch.batch_size, max_seq_length=batch.max_seq_length,
                 last_target_update_episode=learner.last_target_update_episode,
                 preference_rng=preference.rng.bit_generator.state,
                 phase=phase, task=task, quarter=part, t_env=t_env, episode=episode, update=update), path)


def paired_updates(learner, snapshot, preference, scheme, groups, warmup, repeats):
    device = th.device(learner.args.device)
    data = SimpleNamespace(**{name: {key: value.to(device) for key, value in tensors.items()}
                              for name, tensors in snapshot["data"].items()})
    batch = EpisodeBatch(scheme, groups, snapshot["batch_size"], snapshot["max_seq_length"],
                         data=data, device=device)
    preference.phase, preference.task = snapshot["phase"], snapshot["task"]

    def synchronize():
        if device.type == "cuda":
            th.cuda.synchronize(device)

    def update(with_preference):
        # Restore before both arms and outside the timed region. On CPU, a
        # loaded optimizer tensor can share storage with its state dictionary.
        state = dict(snapshot, optimizer=deepcopy(snapshot["optimizer"]))
        restore_networks_and_optimizer(learner, state)
        learner.last_target_update_episode = snapshot["last_target_update_episode"]
        learner.log_stats_t = snapshot["t_env"]
        preference.rng.bit_generator.state = snapshot["preference_rng"]
        synchronize()
        start = perf_counter()
        learner.train(batch, snapshot["t_env"], snapshot["episode"],
                      preference=preference if with_preference else None)
        synchronize()
        return (perf_counter() - start) * 1000

    for index in range(warmup + repeats):
        order = (False, True) if index % 2 == 0 else (True, False)
        td, rank = sorted(((mode, update(mode)) for mode in order), key=lambda arm: arm[0])
        if index >= warmup:
            yield dict(repeat=index - warmup, first_arm="TD-only" if not order[0] else "TD+TKPR-K1",
                       td_ms=td[1], td_tkpr_ms=rank[1], extra_ms=rank[1] - td[1])
