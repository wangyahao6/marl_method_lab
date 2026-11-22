"""Fixed Top-1 plus rank-distance pair masks for the SZC comparison."""
from functools import partial

import torch as th

from learners.pair_ablation import PairAblationPreference, ablation_statistics
from learners.ranking_preference import reference_order


DISTANCE_SELECTIONS = ("k1", "near", "mid", "far", "far_mid", "random_matched", "maxgap",
                       "far_mid_unfiltered", "far_mid_filtered")


def stable_subset(candidates, scores, count):
    """Take the lowest scores, using original pair index to resolve ties."""
    indices = th.sort(scores.masked_fill(~candidates, th.inf), dim=-1, stable=True).indices
    return candidates & (indices.argsort(dim=-1) < count[..., None])


@th.no_grad()
def select_distance_pairs(old_q, legal, decisions, selection, seed, *, prior_budget=0,
                          min_gap=0, q_scale_floor=0.1, extra_min_gap=0):
    """Return fixed upper-triangle pair masks; only optional extra edges are gap-filtered."""
    if selection not in DISTANCE_SELECTIONS or min_gap != 0:
        raise ValueError("distance ablation requires a registered selection and min_gap=0")
    size = legal.size(-1)
    left, right = th.triu_indices(size, size, offset=1, device=legal.device)
    legal_pairs = legal[..., left] & legal[..., right]
    distinct = old_q[..., left] != old_q[..., right]
    eligible = legal_pairs & distinct
    order = th.sort(old_q.masked_fill(~legal, -th.inf), dim=-1,
                    descending=True, stable=True).indices
    ranks = order.argsort(dim=-1)
    distance = (ranks[..., left] - ranks[..., right]).abs()
    top = order[..., :1]
    top_pairs = (left == top) | (right == top)
    non_top = eligible & ~top_pairs
    near, mid, far = (non_top & (distance == 1), non_top & (distance == 2),
                      non_top & (distance >= 3))
    count = legal.sum(dim=-1)
    budget = th.minimum(th.minimum(count - 2, count - 3),
                        th.div((count - 4) * (count - 3), 2, rounding_mode="floor")).clamp_min(0)
    budget = budget.masked_fill(count < 5, 0)
    far_quota = th.div(2 * budget + 1, 3, rounding_mode="floor")
    mid_quota = budget - far_quota
    base_valid = decisions & (count >= 4) & ~(legal_pairs & ~distinct).any(dim=-1)
    _, scale, _, _ = reference_order(old_q, legal, q_scale_floor)
    gap = (old_q[..., left] - old_q[..., right]).abs() / scale
    filtered_far = far & (gap >= extra_min_gap)
    filtered_mid = mid & (gap >= extra_min_gap)
    gap_pair = selection in ("far_mid_unfiltered", "far_mid_filtered")
    valid = base_valid
    if gap_pair:
        # Both arms use the stricter arm's feasible states.
        common_far = far & (gap >= 0.5)
        common_mid = mid & (gap >= 0.5)
        valid = valid & (common_far.sum(dim=-1) >= far_quota)
        valid = valid & (common_mid.sum(dim=-1) >= mid_quota)

    generator = th.Generator(device=old_q.device).manual_seed(seed)
    random_scores = th.rand(eligible.shape, generator=generator, device=old_q.device)
    zero = th.zeros_like(budget)
    if selection == "k1":
        extra = th.zeros_like(eligible)
    elif selection == "near":
        extra = stable_subset(near, random_scores, budget)
    elif selection == "mid":
        extra = stable_subset(mid, random_scores, budget)
    elif selection == "far":
        extra = stable_subset(far, random_scores, budget)
    elif selection in ("far_mid", "far_mid_unfiltered", "far_mid_filtered"):
        chosen_far = filtered_far if gap_pair else far
        chosen_mid = filtered_mid if gap_pair else mid
        extra = (stable_subset(chosen_far, random_scores, far_quota)
                 | stable_subset(chosen_mid, random_scores, mid_quota))
    elif selection == "random_matched":
        extra = stable_subset(non_top, random_scores, budget)
    else:
        extra = stable_subset(non_top, -gap, budget)
    selected = (eligible & top_pairs | extra) & valid[..., None]
    counts = th.stack((legal_pairs.sum(dim=-1) * valid,
                       eligible.sum(dim=-1) * valid,
                       (selected & top_pairs).sum(dim=-1)), dim=-1)
    adjacent = eligible & (distance == 1)
    rejected = ((far | mid) & (gap < extra_min_gap)) if gap_pair else th.zeros_like(eligible)
    diagnostics = dict(
        complete_order_states=int((valid & ~(adjacent & ~selected).any(dim=-1)).sum()),
        ordering_edges=int((adjacent & valid[..., None]).sum()),
        selected_ordering_edges=int((adjacent & selected).sum()),
        gap_eligible_pairs=int(((eligible & ~rejected) & valid[..., None]).sum()),
        gap_filtered_pairs=int((rejected & valid[..., None]).sum()),
        excluded_by_extra_gap_states=int((base_valid & ~valid).sum()),
        extra_budget_pairs=int((budget * valid).sum()),
        extra_selected_pairs=int((selected & ~top_pairs).sum()),
        extra_near_pairs=int((selected & near).sum()),
        extra_mid_pairs=int((selected & mid).sum()),
        extra_far_pairs=int((selected & far).sum()),
        target_far_pairs=int((far_quota * valid).sum()) if selection.startswith("far_mid") else int(zero.sum()),
        target_mid_pairs=int((mid_quota * valid).sum()) if selection.startswith("far_mid") else int(zero.sum()),
    )
    return selected, valid, counts, diagnostics


def distance_statistics(summary):
    stats = ablation_statistics(summary[:13])
    violations = int(summary[13])
    pairs = stats["valid_pairs"]
    stats.update(pair_violations=violations,
                 pair_violation_rate=violations / pairs if pairs else None)
    return stats


class PairDistancePreference(PairAblationPreference):
    statistics = staticmethod(distance_statistics)

    def __init__(self, selection, settings, rng, scheme, groups, log):
        selector = partial(select_distance_pairs, extra_min_gap=settings["extra_min_gap"])
        super().__init__(selection, settings, rng, scheme, groups, log,
                         selector=selector, pair_audit=True)
