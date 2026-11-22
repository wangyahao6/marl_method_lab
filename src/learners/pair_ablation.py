"""Fixed action-pair selections for the two SZC ranking ablations."""
import hashlib

import torch as th
from torch.nn.utils.rnn import pad_sequence

from learners.action_set_preference import ActionSetPreference, batch_episodes, decision_masks, unroll
from learners.ranking_preference import pair_state_loss, ranking_statistics, reference_order, summarize


STRUCTURES = ("k1", "adjacent", "k1_adjacent", "frpr", "random_n_minus_1",
              "random_2n_minus_3", "random_2n_minus_3_matched")


def ranked_subset(candidates, scores, count):
    order = scores.masked_fill(~candidates, th.inf).argsort(dim=-1)
    return candidates & (order.argsort(dim=-1) < count[..., None])


def ratio_counts(budget, percent, generator, prior_budget):
    """Apportion the cumulative rounded target in a randomized state order."""
    permutation = th.randperm(budget.numel(), generator=generator, device=budget.device)
    carry = (50 + prior_budget * percent) % 100
    cumulative = th.div(budget.flatten()[permutation].cumsum(0) * percent + carry,
                        100, rounding_mode="floor")
    counts = th.empty_like(cumulative)
    counts[permutation] = cumulative.diff(prepend=cumulative.new_zeros(1))
    return counts.reshape_as(budget)


@th.no_grad()
def select_pairs(old_q, legal, decisions, selection, seed, *, prior_budget=0,
                 min_gap=0, q_scale_floor=0.1):
    """Return masks over upper-triangular pairs indexed by original action ID."""
    left, right = th.triu_indices(legal.size(-1), legal.size(-1), offset=1, device=legal.device)
    legal_pairs = legal[..., left] & legal[..., right]
    distinct = old_q[..., left] != old_q[..., right]
    eligible = legal_pairs & distinct
    # Use the same population standard deviation and floor as reference_order.
    standardized, _, _, _ = reference_order(old_q, legal, q_scale_floor)
    gap = (standardized[..., left] - standardized[..., right]).abs()
    filtered = eligible & (gap >= min_gap)
    order = old_q.masked_fill(~legal, -th.inf).argsort(dim=-1, descending=True)
    top = order[..., :1]
    top_pairs = (left == top) | (right == top)
    legal_count = legal.sum(dim=-1)
    budget = (legal_count - 1).clamp_min(0)
    valid = decisions & (legal_count >= 4)
    generator = th.Generator(device=old_q.device).manual_seed(seed)

    if selection in STRUCTURES:
        valid &= ~(legal_pairs & ~distinct).any(dim=-1)
        if min_gap:
            # The largest random control needs 2N-3 eligible edges. Applying this
            # condition to every structure keeps their decision states identical.
            valid &= filtered.sum(dim=-1) >= 2 * legal_count - 3
        ranks = order.argsort(dim=-1)
        all_adjacent = eligible & ((ranks[..., left] - ranks[..., right]).abs() == 1)
        adjacent = filtered & all_adjacent
        if selection == "k1":
            selected = filtered & top_pairs
        elif selection == "adjacent":
            selected = adjacent
        elif selection == "k1_adjacent":
            selected = (filtered & top_pairs) | adjacent
        elif selection == "frpr":
            selected = filtered
        elif selection == "random_2n_minus_3_matched":
            scores = th.rand(eligible.shape, generator=generator, device=old_q.device)
            # Match the filtered non-Top-1 count of K1+Adjacent state by state.
            non_top_budget = (adjacent & ~top_pairs).sum(dim=-1)
            non_top = ranked_subset(filtered & ~top_pairs, scores, non_top_budget)
            selected = (filtered & top_pairs) | non_top
        else:
            scores = th.rand(eligible.shape, generator=generator, device=old_q.device)
            pair_budget = 2 * legal_count - 3 if selection == "random_2n_minus_3" else budget
            selected = ranked_subset(filtered, scores, pair_budget)
    else:
        non_top = eligible & ~top_pairs
        unique_top = (eligible & top_pairs).sum(dim=-1) == budget
        valid &= unique_top & (non_top.sum(dim=-1) >= budget)
        top_count = ratio_counts(budget.masked_fill(~valid, 0), selection, generator, prior_budget)
        scores = th.rand(eligible.shape, generator=generator, device=old_q.device)
        selected = (ranked_subset(eligible & top_pairs, scores, top_count)
                    | ranked_subset(non_top, scores, budget - top_count))
    selected &= valid[..., None]
    counts = th.stack((legal_pairs.sum(dim=-1) * valid, eligible.sum(dim=-1) * valid,
                       (selected & top_pairs).sum(dim=-1)), dim=-1)
    diagnostics = {}
    if selection in STRUCTURES:
        # A strictly oriented pair set has full closure iff it contains every adjacent edge.
        diagnostics["complete_order_states"] = int((valid & ~(all_adjacent & ~selected).any(dim=-1)).sum())
        diagnostics["ordering_edges"] = int((all_adjacent & valid[..., None]).sum())
        diagnostics["selected_ordering_edges"] = int((all_adjacent & selected).sum())
        diagnostics["gap_eligible_pairs"] = int((filtered & valid[..., None]).sum())
        diagnostics["gap_filtered_pairs"] = int(((eligible & ~filtered) & valid[..., None]).sum())
    else:
        diagnostics["budget_mismatch_states"] = int((valid & (selected.sum(dim=-1) != budget)).sum())
    return selected, valid, counts, diagnostics


def pair_statistics(decisions, valid, pairs, candidates, eligible, top_pairs):
    return dict(decision_states=int(decisions), valid_states=int(valid),
                coverage=valid / decisions if decisions else None, candidate_pairs=int(candidates),
                eligible_pairs=int(eligible), valid_pairs=int(pairs), top1_pairs=int(top_pairs),
                non_top1_pairs=int(pairs - top_pairs), top1_ratio=top_pairs / pairs if pairs else None,
                mean_pairs_per_state=pairs / valid if valid else None,
                pair_compression_rate=1 - pairs / eligible if eligible else None)


def ablation_statistics(summary):
    stats = ranking_statistics(summary[:7])
    stats.update(pair_statistics(summary[3], summary[1], summary[2], *summary[7:10]))
    if len(summary) >= 13:
        stats.update(kendall_tau_a=summary[10] if summary[1] else None,
                     mean_normalized_regret=summary[11] if summary[1] else None,
                     selected_top1_retention=summary[12] if summary[1] else None)
    return stats


def aggregate_ablation_statistics(rows):
    return pair_statistics(*(sum(row[key] for row in rows) for key in (
        "decision_states", "valid_states", "valid_pairs", "candidate_pairs", "eligible_pairs", "top1_pairs")))


def ablation_loss(current_q, old_q, legal, decisions, settings, selected, counts, *, pair_audit=False):
    _, scale, best, confidence = reference_order(old_q, legal, settings["q_scale_floor"])
    left, right = th.triu_indices(legal.size(-1), legal.size(-1), offset=1, device=legal.device)
    current_gap = (current_q[..., left] - current_q[..., right]) / scale
    old_q = old_q.detach()
    direction = (old_q[..., left] - old_q[..., right]).sign()
    state_loss, pair_counts = pair_state_loss(-direction * current_gap, selected, settings["temperature"], (-1,))
    loss, count, summary = summarize(state_loss, pair_counts, current_q, legal, decisions, best, confidence)
    with th.no_grad():
        active = selected.any(dim=-1)
        all_pairs = legal[..., left] & legal[..., right] & (direction != 0)
        agreement = direction * current_gap.detach().sign()
        tau = agreement.masked_fill(~all_pairs, 0).sum(dim=-1) / all_pairs.sum(dim=-1).clamp_min(1)
        greedy = current_q.detach().masked_fill(~legal, -th.inf).argmax(dim=-1)
        old_best = old_q.gather(-1, best).squeeze(-1)
        regret = (old_best - old_q.gather(-1, greedy.unsqueeze(-1)).squeeze(-1)) / scale.squeeze(-1)
        active_count = active.sum().clamp_min(1)
        diagnostics = th.stack((tau.masked_fill(~active, 0).sum() / active_count,
                                regret.masked_fill(~active, 0).sum() / active_count,
                                ((greedy == best.squeeze(-1)) & active).sum() / active_count))
    summary = th.cat((summary, counts.sum(dim=tuple(range(counts.ndim - 1))), diagnostics))
    if pair_audit:
        violations = (selected & (-direction * current_gap.detach() >= 0)).sum()
        summary = th.cat((summary, violations.reshape(1)))
    return loss, count, summary


class PairAblationPreference(ActionSetPreference):
    statistics = staticmethod(ablation_statistics)

    def __init__(self, selection, settings, rng, scheme, groups, log, *, selector=None,
                 pair_audit=False):
        super().__init__(settings, rng, scheme, groups, log)
        self.selection = selection
        self.selector = selector
        self.pair_audit = pair_audit

    def prepare_pool(self, pool, *, prior_budget):
        episodes = pool["episodes"]
        if not episodes:
            audit_key = "complete_order_states" if self.selection in STRUCTURES or self.selector is not None else "budget_mismatch_states"
            empty_hash = hashlib.sha256().hexdigest()
            return [], dict(pair_statistics(0, 0, 0, 0, 0, 0), **{audit_key: 0}, episodes=0,
                            valid_state_sha256=empty_hash, pair_mask_sha256=empty_hash)
        batch, _ = batch_episodes(episodes, self.scheme, self.groups, "cpu",
                                  include_good=False, include_old_q=True)
        legal, decisions = decision_masks(batch)
        selector = self.selector if self.selector is not None else select_pairs
        selected, valid, counts, audit = selector(
            batch["old_q_values"][:, :-1], legal, decisions, self.selection, self.settings["pair_seed"],
            prior_budget=prior_budget, min_gap=self.settings["min_gap"],
            q_scale_floor=self.settings["q_scale_floor"])
        prepared = []
        valid_hash = hashlib.sha256()
        mask_hash = hashlib.sha256()
        for episode, mask, state_mask, pair_counts in zip(episodes, selected, valid, counts):
            length = episode["length"]
            mask = mask[:length].clone()
            valid_hash.update(state_mask[:length].numpy().tobytes())
            mask_hash.update(mask.numpy().tobytes())
            prepared.append(dict(episode, pair_mask=mask, pair_counts=pair_counts[:length].clone()))
        stats = pair_statistics(int(decisions.sum()), int(valid.sum()), int(selected.sum()),
                                *counts.flatten(0, -2).sum(dim=0).tolist())
        if self.selection in STRUCTURES or self.selector is not None:
            audit["ordering_edge_coverage"] = (audit["selected_ordering_edges"] / audit["ordering_edges"]
                                                if audit["ordering_edges"] else None)
            audit["complete_order_coverage"] = (audit["complete_order_states"] / stats["valid_states"]
                                                 if stats["valid_states"] else None)
            audit["gap_filter_rate"] = (audit["gap_filtered_pairs"] / stats["eligible_pairs"]
                                        if stats["eligible_pairs"] else None)
            audit["pair_coverage"] = (stats["valid_pairs"] / stats["eligible_pairs"]
                                      if stats["eligible_pairs"] else None)
        return prepared, dict(stats, **audit, episodes=len(episodes),
                              valid_state_sha256=valid_hash.hexdigest(), pair_mask_sha256=mask_hash.hexdigest())

    def task_loss(self, mac, pool, episodes):
        batch, _ = batch_episodes(episodes, self.scheme, self.groups, mac.args.device,
                                  include_good=False, include_old_q=True)
        legal, decisions = decision_masks(batch)
        selected = pad_sequence([ep["pair_mask"] for ep in episodes], batch_first=True).to(mac.args.device)
        counts = pad_sequence([ep["pair_counts"] for ep in episodes], batch_first=True).to(mac.args.device)
        current_q = unroll(mac, batch, batch.max_seq_length - 1)
        return ablation_loss(current_q, batch["old_q_values"][:, :-1], legal, decisions,
                             self.settings, selected, counts, pair_audit=self.pair_audit)
