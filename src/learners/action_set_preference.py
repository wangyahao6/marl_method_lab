"""Recurrent action-set replay for a shared SMAC VDN agent."""
from collections import Counter
from time import perf_counter
from types import SimpleNamespace

import torch as th
from torch.nn.utils.rnn import pad_sequence

from components.episode_buffer import EpisodeBatch


def unroll(mac, batch, steps):
    previous_hidden = mac.hidden_states
    mac.init_hidden(batch.batch_size)
    try:
        return th.stack([mac.forward(batch, t=t) for t in range(steps)], dim=1)
    finally:
        mac.hidden_states = previous_hidden


def decision_masks(batch):
    # The runner fills one final bootstrap observation even for a timeout.
    filled = batch["filled"].bool()
    terminal = batch["terminated"][:, :-1].bool()
    after_terminal = th.cat((th.zeros_like(terminal[:, :1]), terminal[:, :-1]), dim=1)
    decision = filled[:, :-1] & filled[:, 1:] & (after_terminal.cumsum(dim=1) == 0)
    legal = batch["avail_actions"][:, :-1].bool()
    # In SMACv2, a dead agent has only action 0 (no-op) available.
    alive = legal[..., 1:].any(dim=-1)
    return legal, decision & alive


def reference_gaps(batch, reference_q):
    legal, _ = decision_masks(batch)
    best = reference_q.masked_fill(~legal, -th.inf).amax(dim=-1, keepdim=True)
    return (best - reference_q).masked_fill(~legal, th.inf)


def good_mask_from_gaps(batch, gaps, delta):
    legal, decision = decision_masks(batch)
    good = legal & decision.unsqueeze(-1) & (gaps <= delta)
    # Keep the serialized time axis aligned with the full EpisodeBatch.
    final = th.zeros_like(batch["avail_actions"][:, -1:], dtype=th.bool)
    return th.cat((good, final), dim=1)


def make_good_mask(batch, reference_q, delta):
    gaps = reference_gaps(batch, reference_q)
    return good_mask_from_gaps(batch, gaps, delta), gaps


def set_counts(legal, decisions, good):
    sizes = good.sum(dim=-1)
    return th.stack((decisions.sum(), (sizes * decisions).sum(),
                     (decisions & (sizes == 1)).sum(),
                     (decisions & (sizes == legal.sum(dim=-1))).sum()))


def set_statistics(count, size, singleton, all_actions):
    return dict(decision_states=int(count),
                average_good_set_size=size / count if count else None,
                singleton_ratio=singleton / count if count else None,
                all_action_ratio=all_actions / count if count else None)


def preference_loss(mac, batch, good, margin):
    legal, decision = decision_masks(batch)
    good = good[:, :-1] & legal
    bad = legal & ~good
    valid = decision & good.any(dim=-1) & bad.any(dim=-1)
    q = unroll(mac, batch, batch.max_seq_length - 1)
    q_good = q.masked_fill(~good, -th.inf).amax(dim=-1).masked_fill(~valid, 0)
    q_bad = q.masked_fill(~bad, -th.inf).amax(dim=-1).masked_fill(~valid, 0)
    slack = q_bad - q_good + margin
    loss_sum = slack.relu().masked_fill(~valid, 0).sum()
    count = valid.sum()
    denominator = count.clamp_min(1)
    greedy = q.detach().masked_fill(~legal, -th.inf).argmax(dim=-1, keepdim=True)
    outside = valid & ~good.gather(-1, greedy).squeeze(-1)
    stats = th.stack((loss_sum.detach(), q_good.detach().sum(), q_bad.detach().sum(),
                      (valid & (slack > 0)).sum(), outside.sum())) / denominator
    return loss_sum / denominator, count, th.cat((stats, count.reshape(1),
                                                   set_counts(legal, decision, good)))


def preference_statistics(summary):
    loss, q_good, q_bad, violation, outside, count, *set_stats = summary
    return dict(loss=loss if count else None, q_good_max=q_good if count else None,
                q_bad_max=q_bad if count else None, violation_ratio=violation if count else None,
                greedy_outside_ratio=outside if count else None,
                valid_states=int(count), **set_statistics(*set_stats))


def pack_episode(batch, length, **metadata):
    return dict(metadata, length=length, data={
        key: value[0, :length + 1].detach().cpu().clone()
        for key, value in batch.data.transition_data.items()
    })


def batch_episodes(episodes, scheme, groups, device, *, include_good=True):
    fields = ("obs", "actions_onehot", "avail_actions", "terminated", "filled")
    data = {key: pad_sequence([ep["data"][key] for ep in episodes], batch_first=True).to(device)
            for key in fields}
    steps = data["obs"].shape[1]
    batch = EpisodeBatch({key: scheme[key] for key in fields}, groups, len(episodes), steps,
                         data=SimpleNamespace(transition_data=data, episode_data={}), device=device)
    good = None
    if include_good:
        good = pad_sequence([ep["good_mask"] for ep in episodes], batch_first=True).to(device)
    return batch, good


class ActionSetPreference:
    statistics = staticmethod(preference_statistics)

    def __init__(self, settings, rng, scheme, groups, log):
        self.settings = settings
        self.rng = rng
        self.scheme = scheme
        self.groups = groups
        self.log = log
        self.history = {}
        self.phase = None
        self.task = None
        self.report = None
        self.forward_events = None

    def begin_update(self):
        self.update_started = perf_counter()

    def sample(self):
        # Keep every stage snapshot. Use the latest available nonempty pool per old task.
        pools = {pool["task"]: pool for pool in self.history.values()
                 if pool["task"] != self.task and pool["episodes"]}
        if not pools:
            return []
        task_ids = list(pools)
        slots = self.rng.permutation(len(task_ids)).tolist()
        allocation = Counter(slots[i % len(slots)] for i in range(self.settings["batch_episodes"]))
        result = []
        for index, count in allocation.items():
            pool = pools[task_ids[index]]
            size = len(pool["episodes"])
            episode_ids = self.rng.choice(size, count, replace=count > size).tolist()
            result.append((pool, episode_ids))
        return result

    def task_loss(self, mac, pool, episodes):
        batch, good = batch_episodes(episodes, self.scheme, self.groups, mac.args.device)
        return preference_loss(mac, batch, good, self.settings["margin"])

    def __call__(self, mac):
        start = perf_counter()
        rows, summaries, losses, valid_counts = [], [], [], []
        sampled = self.sample()
        events = None
        if sampled and th.device(mac.args.device).type == "cuda":
            if self.forward_events is None:
                self.forward_events = tuple(th.cuda.Event(enable_timing=True) for _ in range(2))
            events = self.forward_events
            events[0].record()
        for pool, episode_ids in sampled:
            episodes = [pool["episodes"][i] for i in episode_ids]
            loss, count, summary = self.task_loss(mac, pool, episodes)
            rows.append(dict(task=pool["task"], source_phase=pool["phase"],
                             pool_size=len(pool["episodes"]), episode_ids=episode_ids,
                             sampled_episodes=len(episode_ids)))
            summaries.append(summary)
            losses.append(loss)
            valid_counts.append(count)
        result = None
        if losses:
            included = th.stack(valid_counts) > 0
            result = (th.stack(losses) * included).sum() / included.sum().clamp_min(1)
        if events is not None:
            events[1].record()
        self.report = (rows, summaries, perf_counter() - start, events)
        return result

    def record(self, td_loss, total_loss, pref_loss, t_env):
        rows, summaries, elapsed, events = self.report
        header = th.stack((td_loss.detach(), total_loss.detach(),
                           td_loss.new_zeros(()) if pref_loss is None else pref_loss.detach()))
        values = th.cat((header, *summaries)).cpu().tolist()
        update_seconds = perf_counter() - self.update_started
        forward_cuda_seconds = events[0].elapsed_time(events[1]) / 1000 if events else None
        offset = len(header)
        for row, summary in zip(rows, summaries):
            end = offset + summary.numel()
            row.update(self.statistics(values[offset:end]))
            offset = end
        self.log(dict(phase=self.phase, task=self.task, t_env=t_env, td_loss=values[0],
                      total_loss=values[1], preference_loss=values[2], historical_tasks=rows,
                      historical_episode_count=sum(row["sampled_episodes"] for row in rows),
                      preference_prepare_seconds=elapsed, preference_forward_cuda_seconds=forward_cuda_seconds,
                      learner_update_seconds=update_seconds))
        self.report = None
