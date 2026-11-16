"""Decision preference losses for recurrent VDN replay."""
from functools import partial

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


def full_rank_preference(current_q, reference_q, legal, decisions, settings, *, pair_audit=False):
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings, pair_audit=pair_audit)


def full_rank_preference_unfiltered(current_q, reference_q, legal, decisions, settings):
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings,
                           unfiltered=True, pair_audit=True)


def full_rank_preference_relu(current_q, reference_q, legal, decisions, settings):
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings, th.relu)


def topk_preference(current_q, reference_q, legal, decisions, settings):
    decisions = decisions & (legal.sum(dim=-1) >= 2)
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings, k=settings["k"])


@th.no_grad()
def _topk_actions(reference_q, legal, effective_k):
    # Stable descending sort breaks ties by ascending action index.
    order = th.sort(reference_q.masked_fill(~legal, -th.inf), dim=-1, descending=True, stable=True).indices
    sorted_legal = legal.gather(-1, order)
    selected = sorted_legal & (sorted_legal.cumsum(dim=-1) <= effective_k.unsqueeze(-1))
    return th.zeros_like(legal).scatter(-1, order, selected)


def _full_rank_loss(current_q, reference_q, legal, decisions, settings, penalty=None, *,
                    k=None, unfiltered=False, pair_audit=False):
    old, scale, best, confidence = reference_order(reference_q, legal, settings["q_scale_floor"])
    old_gap = old.unsqueeze(-1) - old.unsqueeze(-2)
    current_gap = (current_q.unsqueeze(-1) - current_q.unsqueeze(-2)) / scale.unsqueeze(-1)
    action_count = legal.size(-1)
    unique_pairs = th.ones((action_count, action_count), device=legal.device, dtype=th.bool).triu(1)
    candidates = unique_pairs & legal.unsqueeze(-1) & legal.unsqueeze(-2) & decisions[..., None, None]
    eligible = candidates & (old_gap != 0)
    pairs = eligible if unfiltered else eligible & (old_gap.abs() >= settings["min_gap"])
    if k is not None:
        max_k = (legal.sum(dim=-1) - 1).clamp_min(0)
        effective_k = max_k.clamp_max(k)
        if k < action_count - 1:
            top = _topk_actions(reference_q, legal, effective_k)
            pairs = pairs & (top.unsqueeze(-1) | top.unsqueeze(-2))

    inverted_gap = -old_gap.sign() * current_gap
    penalties = softplus(inverted_gap / settings["temperature"]) if penalty is None else penalty(inverted_gap)
    pair_counts = pairs.sum(dim=(-2, -1))
    state_loss = penalties.masked_fill(~pairs, 0).sum(dim=(-2, -1)) / pair_counts.clamp_min(1)
    loss, count, summary = summarize(state_loss, pair_counts, current_q, legal, decisions, best, confidence)
    if pair_audit:
        summary = th.cat((summary, th.stack((candidates.sum(), eligible.sum()))))
    if k is not None:
        violations = (pairs & (inverted_gap.detach() >= 0)).sum()
        effective_k_sum = effective_k.masked_fill(~decisions, 0).sum()
        clipped = (decisions & (effective_k < k)).sum()
        histogram = effective_k.new_zeros(min(k, action_count - 1) + 1)
        histogram.scatter_add_(0, effective_k.reshape(-1), (pair_counts > 0).long().reshape(-1))
        summary = th.cat((summary, th.stack((violations, effective_k_sum, clipped)), histogram[1:]))
    return loss, count, summary


def top1_preference(current_q, reference_q, legal, decisions, settings):
    _, scale, best, confidence = reference_order(reference_q, legal, settings["q_scale_floor"])
    confident = decisions & (confidence > 0) & (confidence >= settings["min_gap"])
    return _top1_softplus(current_q, legal, decisions, settings["temperature"],
                          scale, best, confidence, confident)


def top1_unfiltered_preference(current_q, reference_q, legal, decisions, settings):
    _, scale, best, confidence = reference_order(reference_q, legal, settings["q_scale_floor"])
    decisions = decisions & (legal.sum(dim=-1) >= 2)
    confident = decisions & (confidence > 0)
    loss, count, summary = _top1_softplus(current_q, legal, decisions, settings["temperature"],
                                          scale, best, confidence, confident)
    mean_confidence = confidence.masked_fill(~confident, 0).sum() / count.clamp_min(1)
    return loss, count, th.cat((summary, mean_confidence.reshape(1)))


def _top1_softplus(current_q, legal, decisions, temperature, scale, best, confidence, confident):
    pairs = legal.scatter(-1, best, False) & confident.unsqueeze(-1)
    current_gap = (current_q.gather(-1, best) - current_q) / scale
    penalties = softplus(-current_gap / temperature)
    pair_counts = pairs.sum(dim=-1)
    mean_penalty = penalties.masked_fill(~pairs, 0).sum(dim=-1) / pair_counts.clamp_min(1)
    state_loss = mean_penalty * confidence.masked_fill(~confident, 0)
    return summarize(state_loss, pair_counts, current_q, legal, decisions, best, confidence)



def top1_max_preference(current_q, reference_q, legal, decisions, settings):
    _, scale, best, confidence = reference_order(reference_q, legal, settings["q_scale_floor"])
    decisions = decisions & (legal.sum(dim=-1) >= 2)
    unique_top1 = decisions & (confidence > 0)
    valid = unique_top1 & (confidence >= settings["min_gap"])

    competitors = legal.scatter(-1, best, False)
    strongest = current_q.masked_fill(~competitors, -th.inf).max(dim=-1).values
    preferred = current_q.gather(-1, best).squeeze(-1)
    gap = (strongest - preferred).masked_fill(~valid, 0) / scale.squeeze(-1)
    slack = (gap + settings["margin"]).masked_fill(~valid, 0)

    valid_count = valid.sum()
    loss = slack.relu().square().sum() / valid_count.clamp_min(1)
    greedy = current_q.detach().masked_fill(~legal, -th.inf).argmax(dim=-1)
    flipped = greedy != best.squeeze(-1)
    summary = th.stack((loss.detach(), valid_count, decisions.sum(),
                        (unique_top1 & flipped).sum(), unique_top1.sum(),
                        (valid & flipped).sum(), (slack.detach() > 0).sum()))
    return loss, valid_count, summary


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



def pair_filter_statistics(candidate_pairs, eligible_pairs, valid_pairs):
    filtered_pairs = eligible_pairs - valid_pairs
    return dict(candidate_pairs=candidate_pairs, tie_pairs=candidate_pairs - eligible_pairs,
                eligible_pairs=eligible_pairs, valid_pairs=valid_pairs, filtered_pairs=filtered_pairs,
                pair_filter_rate=filtered_pairs / eligible_pairs if eligible_pairs else 0.0,
                pair_keep_rate=valid_pairs / eligible_pairs if eligible_pairs else None)


def aggregate_pair_filter_statistics(historical_tasks):
    totals = {key: sum(task[key] for task in historical_tasks)
              for key in ("candidate_pairs", "eligible_pairs", "valid_pairs")}
    return pair_filter_statistics(**totals)


def full_rank_statistics(summary):
    stats = ranking_statistics(summary[:6])
    candidates, eligible = map(int, summary[6:])
    stats.update(pair_filter_statistics(candidates, eligible, stats["valid_pairs"]))
    return stats


def topk_statistics(summary):
    stats = ranking_statistics(summary[:6])
    violations, effective_k_sum, clipped = summary[6:9]
    stats["effective_k_histogram"] = {str(k): int(count) for k, count in enumerate(summary[9:], 1)}
    pairs, decisions = stats["valid_pairs"], stats["decision_states"]
    stats.update(pair_violation_rate=violations / pairs if pairs else None,
                 mean_effective_k=effective_k_sum / decisions if decisions else None,
                 k_clip_rate=clipped / decisions if decisions else None)
    return stats


def top1_unfiltered_statistics(summary):
    stats = ranking_statistics(summary[:-1])
    stats["mean_constrained_confidence"] = summary[-1] if stats["valid_states"] else None
    return stats


def top1_max_statistics(summary):
    loss, valid, decisions, flips, top1_states, constrained_flips, violations = summary
    return dict(loss=loss if valid else None, valid_states=int(valid),
                decision_states=int(decisions), coverage=valid / decisions if decisions else None,
                top1_flips=int(flips), top1_states=int(top1_states),
                top1_flip_rate=flips / top1_states if top1_states else None,
                constrained_top1_flips=int(constrained_flips),
                constrained_top1_flip_rate=constrained_flips / valid if valid else None,
                margin_violations=int(violations),
                margin_violation_rate=violations / valid if valid else None)


class RankingPreference(ActionSetPreference):
    statistics = staticmethod(ranking_statistics)

    def __init__(self, method, settings, rng, scheme, groups, log):
        super().__init__(settings, rng, scheme, groups, log)
        self.loss = {
            "full_rank_preference": partial(full_rank_preference, pair_audit=True),
            "full_rank_preference_unfiltered": full_rank_preference_unfiltered,
            "full_rank_preference_relu": full_rank_preference_relu,
            "topk_preference": topk_preference,
            "top1_preference": top1_preference,
            "top1_max_preference": top1_max_preference,
            "top1_unfiltered_preference": top1_unfiltered_preference,
        }[method]
        if method in ("full_rank_preference", "full_rank_preference_unfiltered"):
            self.statistics = full_rank_statistics
        elif method == "top1_max_preference":
            self.statistics = top1_max_statistics
        elif method == "top1_unfiltered_preference":
            self.statistics = top1_unfiltered_statistics
        elif method == "topk_preference":
            self.statistics = topk_statistics

    def task_loss(self, mac, pool, episodes):
        batch, _ = batch_episodes(episodes, self.scheme, self.groups, mac.args.device,
                                  include_good=False, include_old_q=True)
        legal, decisions = decision_masks(batch)
        decisions = decisions & (legal.sum(dim=-1) >= 2)
        current_q = unroll(mac, batch, batch.max_seq_length - 1)
        return self.loss(current_q, batch["old_q_values"][:, :-1], legal, decisions, self.settings)
