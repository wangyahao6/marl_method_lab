"""Top-1-free action pairs with the closest feasible old-Q gap distribution."""
from statistics import mean

import torch as th

from learners.pair_ablation import PairAblationPreference
from learners.pair_distance import distance_statistics
from learners.ranking_preference import reference_order


def optimal_gap_subset(targets, candidates):
    """Minimum sorted L1 matching; ties choose the smallest original pair IDs."""
    target = sorted(float(gap) for gap in targets)
    ordered = sorted(((float(gap), int(pair_id)) for gap, pair_id in candidates))
    budget = len(target)
    if len(ordered) < budget:
        raise ValueError("insufficient non-Top-1 pairs for the matching budget")
    best = [(0.0, ())] + [None] * budget
    for position, (gap, pair_id) in enumerate(ordered, 1):
        for count in range(min(position, budget), 0, -1):
            previous = best[count - 1]
            if previous is None:
                continue
            cost = previous[0] + abs(gap - target[count - 1])
            incumbent = best[count]
            if incumbent is not None and cost > incumbent[0]:
                continue
            pair_ids = tuple(sorted((*previous[1], pair_id)))
            if incumbent is None or cost < incumbent[0] or pair_ids < incumbent[1]:
                best[count] = cost, pair_ids
    return best[budget][1], best[budget][0] / budget


@th.no_grad()
def select_gap_matched_pairs(old_q, legal, decisions, selection, seed, *, prior_budget=0,
                             min_gap=0, q_scale_floor=0.1):
    """Return a fixed mask over original-index upper-triangle pairs."""
    if selection != "gap_matched_non_top1" or min_gap != 0:
        raise ValueError("gap-matched non-Top-1 selection requires min_gap=0")
    size = legal.size(-1)
    left, right = th.triu_indices(size, size, 1, device=legal.device)
    legal_pairs = legal[..., left] & legal[..., right]
    distinct = old_q[..., left] != old_q[..., right]
    eligible = legal_pairs & distinct
    standardized, _, _, _ = reference_order(old_q, legal, q_scale_floor)
    gap = (standardized[..., left] - standardized[..., right]).abs()
    order = th.sort(old_q.masked_fill(~legal, -th.inf), dim=-1,
                    descending=True, stable=True).indices
    ranks = order.argsort(dim=-1)
    top_pairs = (left == order[..., :1]) | (right == order[..., :1])
    non_top = eligible & ~top_pairs
    count = legal.sum(dim=-1)
    budget = (count - 1).clamp_min(0)
    # This is the intersection of the archived structure-K1 and Ratio-0 rules.
    valid = (decisions & (count >= 4) & ~(legal_pairs & ~distinct).any(dim=-1)
             & (non_top.sum(dim=-1) >= budget))
    selected = th.zeros_like(eligible)
    flat_gap = gap.reshape(-1, gap.size(-1))
    flat_top = (eligible & top_pairs).reshape_as(flat_gap)
    flat_non_top = non_top.reshape_as(flat_gap)
    flat_selected = selected.reshape_as(flat_gap)
    flat_budget = budget.reshape(-1)
    w1, relative_error = [], []
    for row in valid.reshape(-1).nonzero().flatten().tolist():
        target = flat_gap[row, flat_top[row]].tolist()
        candidates = [(flat_gap[row, pair].item(), pair)
                      for pair in flat_non_top[row].nonzero().flatten().tolist()]
        pair_ids, error = optimal_gap_subset(target, candidates)
        flat_selected[row, list(pair_ids)] = True
        w1.append(error)
        relative_error.append(error / mean(target))
    adjacent = eligible & ((ranks[..., left] - ranks[..., right]).abs() == 1)
    counts = th.stack((legal_pairs.sum(dim=-1) * valid,
                       eligible.sum(dim=-1) * valid,
                       (selected & top_pairs).sum(dim=-1)), dim=-1)
    diagnostics = dict(
        complete_order_states=int((valid & ~(adjacent & ~selected).any(dim=-1)).sum()),
        ordering_edges=int((adjacent & valid[..., None]).sum()),
        selected_ordering_edges=int((adjacent & selected).sum()),
        gap_eligible_pairs=int((eligible & valid[..., None]).sum()),
        gap_filtered_pairs=0,
        budget_mismatch_states=int((valid & (selected.sum(dim=-1) != budget)).sum()),
        matching_w1_mean=mean(w1) if w1 else None,
        matching_w1_p90=float(th.tensor(w1).quantile(.9)) if w1 else None,
        matching_relative_error_mean=mean(relative_error) if relative_error else None,
        matching_relative_error_p90=float(th.tensor(relative_error).quantile(.9)) if w1 else None,
    )
    return selected, valid, counts, diagnostics


class PairGapMatchedPreference(PairAblationPreference):
    statistics = staticmethod(distance_statistics)

    def __init__(self, selection, settings, rng, scheme, groups, log):
        super().__init__(selection, settings, rng, scheme, groups, log,
                         selector=select_gap_matched_pairs, pair_audit=True)
