"""Pairwise decision preference losses for recurrent VDN replay."""
import copy

import torch as th
from torch.nn.functional import softplus

from learners.action_set_preference import ActionSetPreference, batch_episodes, decision_masks, unroll


@th.no_grad()
def reference_order(reference_q, legal, q_scale_floor):
    count = legal.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = reference_q.masked_fill(~legal, 0).sum(dim=-1, keepdim=True) / count
    centered = (reference_q - mean).masked_fill(~legal, 0)
    scale = (centered.square().sum(dim=-1, keepdim=True) / count).sqrt().clamp_min(q_scale_floor)
    best_two = reference_q.masked_fill(~legal, -th.inf).topk(2, dim=-1).indices
    best, second = reference_q.gather(-1, best_two).unbind(dim=-1)
    confidence = ((best - second) / scale.squeeze(-1)).masked_fill(count.squeeze(-1) < 2, 0)
    return centered / scale, scale, best_two[..., :1], confidence


def full_rank_preference(current_q, reference_q, legal, decisions, settings):
    old, scale, best, confidence = reference_order(reference_q, legal, settings["q_scale_floor"])
    old_gap = old.unsqueeze(-1) - old.unsqueeze(-2)
    current_gap = (current_q.unsqueeze(-1) - current_q.unsqueeze(-2)) / scale.unsqueeze(-1)
    action_count = legal.size(-1)
    unique_pairs = th.ones((action_count, action_count), device=legal.device, dtype=th.bool).triu(1)
    pairs = (unique_pairs & legal.unsqueeze(-1) & legal.unsqueeze(-2) & decisions[..., None, None]
             & (old_gap != 0) & (old_gap.abs() >= settings["min_gap"]))
    penalties = softplus(-old_gap.sign() * current_gap / settings["temperature"])
    pair_counts = pairs.sum(dim=(-2, -1))
    state_loss = penalties.masked_fill(~pairs, 0).sum(dim=(-2, -1)) / pair_counts.clamp_min(1)
    return summarize(state_loss, pair_counts, current_q, legal, decisions, best, confidence)


def top1_preference(current_q, reference_q, legal, decisions, settings):
    _, scale, best, confidence = reference_order(reference_q, legal, settings["q_scale_floor"])
    confident = decisions & (confidence > 0) & (confidence >= settings["min_gap"])
    pairs = legal.scatter(-1, best, False) & confident.unsqueeze(-1)
    current_gap = (current_q.gather(-1, best) - current_q) / scale
    penalties = softplus(-current_gap / settings["temperature"])
    pair_counts = pairs.sum(dim=-1)
    mean_penalty = penalties.masked_fill(~pairs, 0).sum(dim=-1) / pair_counts.clamp_min(1)
    state_loss = mean_penalty * confidence.masked_fill(~confident, 0)
    return summarize(state_loss, pair_counts, current_q, legal, decisions, best, confidence)


def summarize(state_loss, pair_counts, current_q, legal, decisions, best, confidence):
    valid_count = (pair_counts > 0).sum()
    loss = state_loss.sum() / valid_count.clamp_min(1)
    unique_top1 = decisions & (confidence > 0)
    greedy = current_q.detach().masked_fill(~legal, -th.inf).argmax(dim=-1)
    flips = (unique_top1 & (greedy != best.squeeze(-1))).sum()
    return loss, valid_count, th.stack((loss.detach(), valid_count, pair_counts.sum(),
                                       decisions.sum(), flips, unique_top1.sum()))


def ranking_statistics(summary):
    loss, valid, pairs, decisions, flips, top1_states = summary
    return dict(loss=loss if valid else None, valid_states=int(valid), valid_pairs=int(pairs),
                decision_states=int(decisions), coverage=valid / decisions if decisions else None,
                top1_flips=int(flips), top1_states=int(top1_states),
                top1_flip_rate=flips / top1_states if top1_states else None)


class RankingPreference(ActionSetPreference):
    statistics = staticmethod(ranking_statistics)

    def __init__(self, method, run_dir, settings, rng, scheme, groups, log):
        super().__init__(settings, rng, scheme, groups, log)
        self.loss = {"full_rank_preference": full_rank_preference, "top1_preference": top1_preference}[method]
        self.run_dir = run_dir
        self.references = {}

    def reference_mac(self, mac, pool):
        checkpoint = self.run_dir / pool["reference_checkpoint"]
        if checkpoint not in self.references:
            reference = copy.copy(mac)
            reference.agent = copy.deepcopy(mac.agent)
            reference.hidden_states = None
            reference.agent.load_state_dict(th.load(checkpoint, map_location="cpu"))
            reference.agent.eval().requires_grad_(False)
            self.references[checkpoint] = reference
        return self.references[checkpoint]

    def task_loss(self, mac, pool, episodes):
        batch, _ = batch_episodes(episodes, self.scheme, self.groups, mac.args.device, include_good=False)
        legal, decisions = decision_masks(batch)
        decisions = decisions & (legal.sum(dim=-1) >= 2)
        steps = batch.max_seq_length - 1
        with th.no_grad():
            reference_q = unroll(self.reference_mac(mac, pool), batch, steps)
        current_q = unroll(mac, batch, steps)
        return self.loss(current_q, reference_q, legal, decisions, self.settings)
