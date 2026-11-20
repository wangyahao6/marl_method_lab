"""Decision preference losses for recurrent VDN replay."""
from functools import partial

import torch as th
from torch.nn.functional import softplus

from learners.action_set_preference import (
    ActionSetPreference, batch_episodes, decision_masks, set_counts, set_statistics, unroll,
)


WEIGHT_BINS = ("zero", "(0,0.5)", "[0.5,1)", "[1,2)", "[2,+inf)")


@th.no_grad()
def reference_scale(reference_q, legal, q_scale_floor):
    count = legal.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = reference_q.masked_fill(~legal, 0).sum(dim=-1, keepdim=True) / count
    centered = (reference_q - mean).masked_fill(~legal, 0)
    scale = (centered.square().sum(dim=-1, keepdim=True) / count).sqrt().clamp_min(q_scale_floor)
    return mean, centered, scale


@th.no_grad()
def reference_order(reference_q, legal, q_scale_floor):
    _, centered, scale = reference_scale(reference_q, legal, q_scale_floor)
    best_two = reference_q.masked_fill(~legal, -th.inf).topk(2, dim=-1).indices
    best, second = reference_q.gather(-1, best_two).unbind(dim=-1)
    confidence = ((best - second) / scale.squeeze(-1)).masked_fill(legal.sum(dim=-1) < 2, 0)
    return centered / scale, scale, best_two[..., :1], confidence


def full_rank_preference(current_q, reference_q, legal, decisions, settings, *, pair_audit=False,
                         pair_selection=None):
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings,
                           pair_audit=pair_audit, pair_selection=pair_selection)


def aspr_preference(current_q, reference_q, legal, decisions, settings, *, good):
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings,
                           pair_audit=True, good=good & legal)


def full_rank_preference_raw(current_q, reference_q, legal, decisions, settings):
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings,
                           raw_reference=True, pair_audit=True)


def full_rank_preference_raw_loss(current_q, reference_q, legal, decisions, settings):
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings,
                           raw_loss=True, pair_audit=True)


def full_rank_preference_unfiltered(current_q, reference_q, legal, decisions, settings):
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings,
                           unfiltered=True, pair_audit=True)


def full_rank_preference_relu(current_q, reference_q, legal, decisions, settings):
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings, th.relu)


def topk_preference(current_q, reference_q, legal, decisions, settings):
    decisions = decisions & (legal.sum(dim=-1) >= 2)
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings, k=settings["k"])


def topk_preference_weighted(current_q, reference_q, legal, decisions, settings):
    decisions = decisions & (legal.sum(dim=-1) >= 2)
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings,
                           k=1, confidence_weighted=True)


def topk_preference_high_regret(current_q, reference_q, legal, decisions, settings):
    decisions = decisions & (legal.sum(dim=-1) >= 2)
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings,
                           k=1, high_regret=True)


@th.no_grad()
def _topk_actions(reference_q, legal, effective_k):
    # Stable descending sort breaks ties by ascending action index.
    order = th.sort(reference_q.masked_fill(~legal, -th.inf), dim=-1, descending=True, stable=True).indices
    sorted_legal = legal.gather(-1, order)
    selected = sorted_legal & (sorted_legal.cumsum(dim=-1) <= effective_k.unsqueeze(-1))
    return th.zeros_like(legal).scatter(-1, order, selected)


def pair_state_loss(inverted_gap, pairs, temperature, pair_dims, penalty=None):
    penalties = softplus(inverted_gap / temperature) if penalty is None else penalty(inverted_gap)
    pair_counts = pairs.sum(dim=pair_dims)
    state_loss = penalties.masked_fill(~pairs, 0).sum(dim=pair_dims) / pair_counts.clamp_min(1)
    return state_loss, pair_counts


def full_rank_pairs(old, legal, decisions, settings, *, raw_reference=False, unfiltered=False, good=None):
    """Pair mask shared by FRPR training and offline analysis; old is already standardized."""
    old_gap = old.unsqueeze(-1) - old.unsqueeze(-2)
    action_count = legal.size(-1)
    unique_pairs = th.ones((action_count, action_count), device=legal.device, dtype=th.bool).triu(1)
    candidates = unique_pairs & legal.unsqueeze(-1) & legal.unsqueeze(-2) & decisions[..., None, None]
    if good is not None:
        candidates = candidates & (good.unsqueeze(-1) ^ good.unsqueeze(-2))
    eligible = candidates & (old_gap != 0)
    gap_key = "min_gap_raw" if raw_reference else "min_gap"
    pairs = eligible if unfiltered else eligible & (old_gap.abs() >= settings[gap_key])
    return old_gap, candidates, eligible, pairs


def _full_rank_loss(current_q, reference_q, legal, decisions, settings, penalty=None, *,
                    k=None, unfiltered=False, pair_audit=False, raw_reference=False, raw_loss=False,
                    confidence_weighted=False, high_regret=False, good=None, pair_selection=None, anchor=None):
    current_gap = current_q.unsqueeze(-1) - current_q.unsqueeze(-2)
    if raw_reference:
        old = reference_q.detach()
        best_two = old.masked_fill(~legal, -th.inf).topk(2, dim=-1).indices
        first, second = old.gather(-1, best_two).unbind(dim=-1)
        best = best_two[..., :1]
        confidence = (first - second).masked_fill(legal.sum(dim=-1) < 2, 0)
    else:
        old, scale, best, confidence = reference_order(reference_q, legal, settings["q_scale_floor"])
        if not raw_loss:
            current_gap = current_gap / scale.unsqueeze(-1)
    old_gap, candidates, eligible, pairs = full_rank_pairs(
        old, legal, decisions, settings, raw_reference=raw_reference, unfiltered=unfiltered, good=good)
    action_count = legal.size(-1)
    if pair_selection is not None:
        pairs = pairs & pair_selection
    if k is not None:
        max_k = (legal.sum(dim=-1) - 1).clamp_min(0)
        effective_k = max_k.clamp_max(k)
        if k < action_count - 1:
            top = _topk_actions(reference_q, legal, effective_k)
            pairs = pairs & (top.unsqueeze(-1) | top.unsqueeze(-2))

    inverted_gap = -old_gap.sign() * current_gap
    state_loss, pair_counts = pair_state_loss(
        inverted_gap, pairs, settings.get("temperature"), (-2, -1), penalty)
    if confidence_weighted:
        valid = pair_counts > 0
        weights = confidence.masked_fill(~valid, 0)
        confidence_sum = weights.sum()
        mean_confidence = confidence_sum / valid.sum().clamp_min(1)
        denominator = th.where(confidence_sum > 0, mean_confidence, th.ones_like(mean_confidence))
        weights = th.where(confidence_sum > 0, weights / denominator, valid.to(weights.dtype))
        state_loss = state_loss * weights

    loss, count, summary = summarize(state_loss, pair_counts, current_q, legal, decisions, best, confidence)
    if pair_audit:
        summary = th.cat((summary, th.stack((candidates.sum(), eligible.sum()))))
    if good is not None:
        violations = (pairs & (inverted_gap.detach() >= 0)).sum()
        summary = th.cat((summary, violations.reshape(1), set_counts(legal, decisions, good)))
    if k is not None:
        violations = (pairs & (inverted_gap.detach() >= 0)).sum()
        effective_k_sum = effective_k.masked_fill(~decisions, 0).sum()
        clipped = (decisions & (effective_k < k)).sum()
        histogram = effective_k.new_zeros(min(k, action_count - 1) + 1)
        histogram.scatter_add_(0, effective_k.reshape(-1), (pair_counts > 0).long().reshape(-1))
        summary = th.cat((summary, th.stack((violations, effective_k_sum, clipped)), histogram[1:]))
    if confidence_weighted:
        weight_bins = th.bucketize(weights, weights.new_tensor([0.5, 1., 2.]), right=True) + 1
        weight_bins.masked_fill_(weights == 0, 0)
        weight_histogram = pair_counts.new_zeros(len(WEIGHT_BINS))
        weight_histogram.scatter_add_(0, weight_bins.reshape(-1), valid.long().reshape(-1))
        summary = th.cat((summary, th.stack((mean_confidence, (valid & (confidence == 0)).sum())),
                          weight_histogram))
    if high_regret:
        hr_loss, active, regret = high_regret_penalty(current_q, reference_q, legal, decisions, scale, settings)
        count = ((pair_counts > 0) | active).sum()
        base_loss = loss
        extra_loss = settings["hr_weight"] * hr_loss
        loss = base_loss + extra_loss
        nonzero = (state_loss.detach() != 0) | (active & (extra_loss.detach() != 0))
        summary = th.cat((loss.detach().reshape(1), summary[1:6],
                          th.stack((nonzero.sum(),)), summary[7:],
                          th.stack((active.sum(), hr_loss.detach(), regret,
                                    base_loss.detach(), extra_loss.detach()))))
    if anchor is not None:
        summary = th.cat((summary, anchor_statistics_tensor(
            anchor, current_q, legal, decisions, candidates, eligible, pairs, inverted_gap)))
    return loss, count, summary


@th.no_grad()
def rank_anchor(reference_q, legal, rank):
    order = th.sort(reference_q.masked_fill(~legal, -th.inf), dim=-1, descending=True, stable=True).indices
    index = order[..., rank - 1:rank]
    return th.zeros_like(legal).scatter(-1, index, True) & legal


def anchor_pairs(anchor):
    return anchor.unsqueeze(-1) | anchor.unsqueeze(-2)


def rank_position_preference(current_q, reference_q, legal, decisions, settings, *, rank):
    decisions = decisions & (legal.sum(dim=-1) >= 2)
    anchor = rank_anchor(reference_q, legal, rank)
    if rank == 1:
        # Audited K1 uses the existing K1 loss path verbatim.
        return _full_rank_loss(current_q, reference_q, legal, decisions, settings, k=1, anchor=anchor)
    return _full_rank_loss(current_q, reference_q, legal, decisions, settings,
                           pair_selection=anchor_pairs(anchor), anchor=anchor)


@th.no_grad()
def anchor_statistics_tensor(anchor, current_q, legal, decisions, candidates, eligible, pairs, inverted_gap):
    endpoints = anchor_pairs(anchor)
    valid = pairs.any(dim=-1).any(dim=-1)
    order = th.sort(current_q.masked_fill(~legal, -th.inf), dim=-1, descending=True, stable=True).indices
    rank = ((order.argsort(dim=-1) + 1) * anchor).sum(dim=-1)
    histogram = rank.new_zeros(legal.size(-1) + 1)
    histogram.scatter_add_(0, rank.flatten(), valid.long().flatten())
    totals = th.stack(((candidates & endpoints).sum(), (eligible & endpoints).sum(),
                       (pairs & (inverted_gap >= 0)).sum(), (anchor.any(dim=-1) & decisions).sum()))
    return th.cat((totals, histogram[1:]))


def rank_position_statistics(summary, *, rank, action_count):
    base = summary[:-(action_count + 4)]
    stats = topk_statistics(base) if rank == 1 else ranking_statistics(base)
    candidates, eligible, violations, available, *histogram = map(int, summary[len(base):])
    pairs, states = stats["valid_pairs"], stats["valid_states"]
    stats.update(pair_filter_statistics(candidates, eligible, pairs))
    stats.update(anchor_old_rank=rank, anchor_available_states=available, pair_violations=violations,
                 pair_violation_rate=violations / pairs if pairs else None,
                 mean_valid_pairs=pairs / states if states else None,
                 anchor_new_rank_histogram={str(i): n for i, n in enumerate(histogram, 1)},
                 anchor_mean_new_rank=sum(i * n for i, n in enumerate(histogram, 1)) / states if states else None,
                 anchor_rank_retained_rate=histogram[rank - 1] / states if states else None)
    return stats


class RankPositionPreference(ActionSetPreference):
    def __init__(self, rank, settings, rng, scheme, groups, log):
        super().__init__(settings, rng, scheme, groups, log)
        self.rank = rank
        self.statistics = partial(rank_position_statistics, rank=rank,
                                  action_count=scheme["avail_actions"]["vshape"][0])

    def task_loss(self, mac, pool, episodes):
        batch, _ = batch_episodes(episodes, self.scheme, self.groups, mac.args.device,
                                  include_good=False, include_old_q=True)
        legal, decisions = decision_masks(batch)
        current_q = unroll(mac, batch, batch.max_seq_length - 1)
        return rank_position_preference(current_q, batch["old_q_values"][:, :-1], legal, decisions,
                                        self.settings, rank=self.rank)


def high_regret_penalty(current_q, reference_q, legal, decisions, scale, settings):
    with th.no_grad():
        old_best = reference_q.masked_fill(~legal, -th.inf).argmax(dim=-1, keepdim=True)
        new_best = current_q.detach().masked_fill(~legal, -th.inf).argmax(dim=-1, keepdim=True)
        old_gap = reference_q.gather(-1, old_best) - reference_q.gather(-1, new_best)
        regret = (old_gap / scale).squeeze(-1)
        active = (decisions & (legal.sum(dim=-1) >= 2) & (old_best != new_best).squeeze(-1)
                  & (regret >= settings["hr_threshold"]))
    gap = current_q.gather(-1, new_best) - current_q.gather(-1, old_best)
    logits = (gap / scale / settings["hr_temperature"]).squeeze(-1)
    count = active.sum().clamp_min(1)
    loss = softplus(logits).masked_fill(~active, 0).sum() / count
    mean_regret = regret.masked_fill(~active, 0).sum() / count
    return loss, active, mean_regret


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
                                       decisions.sum(), flips, unique_top1.sum(), (state_loss.detach() != 0).sum()))


def ranking_statistics(summary):
    loss, valid, pairs, decisions, flips, top1_states, nonzero = summary
    return dict(loss=loss if valid else None, valid_states=int(valid), valid_pairs=int(pairs),
                decision_states=int(decisions), coverage=valid / decisions if decisions else None,
                nonzero_loss_states=int(nonzero),
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
    stats = ranking_statistics(summary[:7])
    candidates, eligible = map(int, summary[7:])
    stats.update(pair_filter_statistics(candidates, eligible, stats["valid_pairs"]))
    return stats


def aspr_statistics(summary):
    stats = full_rank_statistics(summary[:9])
    violations = int(summary[9])
    stats.update(pair_violations=violations,
                 pair_violation_rate=violations / stats["valid_pairs"] if stats["valid_pairs"] else None,
                 **set_statistics(*summary[10:]))
    return stats


def topk_statistics(summary):
    stats = ranking_statistics(summary[:7])
    violations, effective_k_sum, clipped = summary[7:10]
    stats["effective_k_histogram"] = {str(k): int(count) for k, count in enumerate(summary[10:], 1)}
    pairs, decisions = stats["valid_pairs"], stats["decision_states"]
    stats.update(pair_violation_rate=violations / pairs if pairs else None,
                 mean_effective_k=effective_k_sum / decisions if decisions else None,
                 k_clip_rate=clipped / decisions if decisions else None)
    return stats


def topk_weighted_statistics(summary):
    extra_fields = len(WEIGHT_BINS) + 2
    stats = topk_statistics(summary[:-extra_fields])
    confidence, zero, *counts = summary[-extra_fields:]
    histogram = dict(zip(WEIGHT_BINS, map(int, counts)))
    nonzero = sum(map(int, counts[1:]))
    valid, decisions = stats["valid_states"], stats["decision_states"]
    stats.update(mean_state_confidence=confidence if valid else None,
                 zero_confidence_states=int(zero),
                 weight_histogram=histogram,
                 nonzero_weight_states=int(nonzero),
                 nonzero_weight_coverage=nonzero / decisions if decisions else None,
                 nonzero_weight_fraction_of_valid=nonzero / valid if valid else None)
    return stats


def topk_high_regret_statistics(summary):
    stats = topk_statistics(summary[:-5])
    count, loss, regret, base_loss, extra_loss = summary[-5:]
    decisions = stats["decision_states"]
    stats.update(high_regret_states=int(count),
                 high_regret_fraction=count / decisions if decisions else None,
                 high_regret_loss=loss if count else None,
                 base_preference_loss=base_loss,
                 weighted_high_regret_loss=extra_loss,
                 high_regret_loss_share=extra_loss / summary[0] if summary[0] else None,
                 mean_high_regret=regret if count else None)
    if count:
        stats["loss"] = summary[0]
    return stats


def top1_unfiltered_statistics(summary):
    stats = ranking_statistics(summary[:-1])
    stats["mean_constrained_confidence"] = summary[-1] if stats["valid_states"] else None
    return stats


def top1_max_statistics(summary):
    loss, valid, decisions, flips, top1_states, constrained_flips, violations = summary
    return dict(loss=loss if valid else None, valid_states=int(valid),
                decision_states=int(decisions), coverage=valid / decisions if decisions else None,
                valid_pairs=int(valid), nonzero_loss_states=int(violations),
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
        self.use_good_mask = method == "aspr_preference"
        self.loss = {
            "aspr_preference": aspr_preference,
            "full_rank_preference": partial(full_rank_preference, pair_audit=True),
            "full_rank_preference_unfiltered": full_rank_preference_unfiltered,
            "full_rank_preference_raw": full_rank_preference_raw,
            "full_rank_preference_raw_loss": full_rank_preference_raw_loss,
            "full_rank_preference_relu": full_rank_preference_relu,
            "topk_preference": topk_preference,
            "topk_preference_weighted": topk_preference_weighted,
            "topk_preference_high_regret": topk_preference_high_regret,
            "top1_preference": top1_preference,
            "top1_max_preference": top1_max_preference,
            "top1_unfiltered_preference": top1_unfiltered_preference,
        }[method]
        if self.use_good_mask:
            self.statistics = aspr_statistics
        elif method in ("full_rank_preference", "full_rank_preference_unfiltered",
                      "full_rank_preference_raw", "full_rank_preference_raw_loss"):
            self.statistics = full_rank_statistics
        elif method == "top1_max_preference":
            self.statistics = top1_max_statistics
        elif method == "top1_unfiltered_preference":
            self.statistics = top1_unfiltered_statistics
        elif method == "topk_preference":
            self.statistics = topk_statistics
        elif method == "topk_preference_weighted":
            self.statistics = topk_weighted_statistics
        elif method == "topk_preference_high_regret":
            self.statistics = topk_high_regret_statistics

    def task_loss(self, mac, pool, episodes):
        batch, good = batch_episodes(episodes, self.scheme, self.groups, mac.args.device,
                                     include_good=self.use_good_mask, include_old_q=True)
        legal, decisions = decision_masks(batch)
        current_q = unroll(mac, batch, batch.max_seq_length - 1)
        if self.use_good_mask:
            return self.loss(current_q, batch["old_q_values"][:, :-1], legal, decisions,
                             self.settings, good=good[:, :-1])
        decisions = decisions & (legal.sum(dim=-1) >= 2)
        return self.loss(current_q, batch["old_q_values"][:, :-1], legal, decisions, self.settings)
