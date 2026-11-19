"""Compact complete-episode snapshots for method history and diagnostics."""
from types import SimpleNamespace

import torch as th


def cpu_episode(batch):
    length = int(batch.max_t_filled().item())
    data = SimpleNamespace(
        transition_data={k: v[:, :length].detach().cpu().clone()
                         for k, v in batch.data.transition_data.items()},
        episode_data={k: v.detach().cpu().clone() for k, v in batch.data.episode_data.items()})
    from components.episode_buffer import EpisodeBatch
    return EpisodeBatch(batch.scheme, batch.groups, 1, length, data=data, device='cpu')


def combine(episodes):
    from components.episode_buffer import EpisodeBatch
    length = max(x.max_seq_length for x in episodes)
    first = episodes[0]
    transition = {}
    for k, v in first.data.transition_data.items():
        out = th.zeros((len(episodes), length, *v.shape[2:]), dtype=v.dtype)
        for i, ep in enumerate(episodes):
            out[i, :ep.max_seq_length] = ep[k][0]
        transition[k] = out
    data = SimpleNamespace(transition_data=transition,
        episode_data={k: th.cat([ep[k] for ep in episodes]) for k in first.data.episode_data})
    return EpisodeBatch(first.scheme, first.groups, len(episodes), length, data=data, device='cpu')


def batch_on_device(batch, device):
    """Copy batch containers and only transfer tensors when the device changes."""
    from components.episode_buffer import EpisodeBatch
    data = SimpleNamespace(
        transition_data={key: value.to(device) for key, value in batch.data.transition_data.items()},
        episode_data={key: value.to(device) for key, value in batch.data.episode_data.items()})
    return EpisodeBatch(batch.scheme, batch.groups, batch.batch_size, batch.max_seq_length,
                        data=data, preprocess=batch.preprocess, device=device)


def pack_episode(batch, length, **metadata):
    return dict(metadata, length=length, data={
        key: value[0, :length + 1].detach().cpu().clone()
        for key, value in batch.data.transition_data.items()
    })


def preference_episode_id(task, episode):
    """Hash exact FRPR inputs, including saved reference Q, independently of list order."""
    import hashlib
    import json

    length = episode["length"]
    identity = [task, episode.get("seed"), episode.get("index"), length]
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode())
    for key in ("obs", "avail_actions", "actions_onehot", "old_q_values", "terminated", "filled"):
        # Bootstrap actions/Q/termination are unused, but the last filled flag is used.
        stop = length + 1 if key == "filled" else length
        values = episode["data"][key][:stop].detach().cpu().contiguous()
        digest.update(json.dumps([key, str(values.dtype), list(values.shape)]).encode())
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()
