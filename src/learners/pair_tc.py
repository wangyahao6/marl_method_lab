"""Offline closure-gain selection and frozen-mask replay for TC coverage ablation."""
import hashlib
from pathlib import Path

import numpy as np
import torch as th

from learners.pair_ablation import PairAblationPreference
from learners.pair_distance import distance_statistics
from learners.ranking_preference import reference_order, full_rank_pairs


LEVELS = ('low', 'medium', 'high')


def digest_masks(masks):
    digest = hashlib.sha256()
    for mask in masks:
        digest.update(mask.contiguous().numpy().tobytes())
    return digest.hexdigest()


def quotas(n):
    """N-2 extras, balanced distance quotas, respecting each bucket's capacity."""
    a, b = np.triu_indices(n - 1, 1)
    bucket = np.minimum(b - a - 1, 2)
    caps = np.bincount(bucket, minlength=3)
    result = np.zeros(3, dtype=np.int64)
    for _ in range(n - 2):
        k = min((k for k in range(3) if result[k] < caps[k]), key=lambda k: (result[k], k))
        result[k] += 1
    return result


def ranked_graphs(q, level, permitted=None):
    """Vectorized exact incremental closure gains; q is descending, strictly ordered.

    Gain is primary. Distance-bucket mean Q-gap is the secondary tie-break target,
    shared by all arms. Action-rank pair ID is the final deterministic tie break.
    Medium uses the median of distinct feasible gain values at each addition.
    """
    rows, n = q.shape
    left, right = np.triu_indices(n, 1)
    extra_ids = np.flatnonzero(left > 0)
    a, b = left[extra_ids], right[extra_ids]
    bucket = np.minimum(b - a - 1, 2)
    extra_permitted = np.ones((rows, len(a)), dtype=bool) if permitted is None else permitted[:, extra_ids]
    remaining = np.broadcast_to(quotas(n), (rows, 3)).copy()
    chosen = np.zeros((rows, len(a)), dtype=bool)
    reach = np.zeros((rows, n), dtype=np.int32)
    if permitted is None:
        reach[:, 0] = (1 << n) - 2
    else:
        reach[:, 0] = (permitted[:, left == 0] * (1 << right[left == 0])).sum(axis=1)
    gap = q[:, a] - q[:, b]
    target_gap = np.zeros_like(gap)
    for k in range(3):
        use = bucket == k
        if use.any():
            target_gap[:, use] = (gap[:, use].mean(axis=1, keepdims=True) if permitted is None else
                (gap[:, use] * extra_permitted[:, use]).sum(axis=1, keepdims=True) / extra_permitted[:, use].sum(axis=1, keepdims=True).clip(1))
    gap_error = np.abs(gap - target_gap)
    pop = np.array([bin(i).count('1') for i in range(1 << n)])
    node_bits = 1 << np.arange(n)
    ri = np.arange(rows)
    for _ in range(n - 2):
        descendants = reach[:, b] | (1 << b)
        ancestors = ((reach[:, None, :] & (1 << a)[None, :, None]) != 0) | (np.arange(n)[None, None, :] == a[None, :, None])
        gains = (pop[descendants[:, :, None] & ~reach[:, None, :]] * ancestors).sum(axis=-1)
        feasible = extra_permitted & ~chosen & (remaining[:, bucket] > 0)
        if level == 'low':
            target = np.where(feasible, gains, n*n).min(axis=1)
        elif level == 'high':
            target = np.where(feasible, gains, -1).max(axis=1)
        elif level == 'medium':
            present = np.stack([(feasible & (gains == g)).any(axis=1) for g in range(n*n)], axis=1)
            pos = (present.sum(axis=1) - 1) // 2
            target = ((present.cumsum(axis=1) > pos[:, None]) & present).argmax(axis=1)
        else:
            raise ValueError(level)
        allowed = feasible & (gains == target[:, None])
        pick = np.where(allowed, gap_error, np.inf).argmin(axis=1)
        chosen[ri, pick] = True
        remaining[ri, bucket[pick]] -= 1
        desc = descendants[ri, pick]
        anc = ancestors[ri, pick]
        reach |= np.where(anc, desc[:, None], 0)
    selected = np.zeros((rows, len(left)), dtype=bool)
    selected[:, left == 0] = True if permitted is None else permitted[:, left == 0]
    selected[:, extra_ids] = chosen
    return selected, pop[reach].sum(axis=1)


@th.no_grad()
def select_tc_pairs(old_q, legal, decisions, selection, seed, *, prior_budget=0,
                    min_gap=0, q_scale_floor=.1):
    if min_gap not in (0, .5):
        raise ValueError('TC coverage supports min_gap 0 or 0.5')
    size = legal.size(-1)
    left, right = th.triu_indices(size, size, 1)
    legal_pairs = legal[..., left] & legal[..., right]
    strict = legal_pairs & (old_q[..., left] != old_q[..., right])
    n = legal.sum(dim=-1)
    valid = decisions & (n >= 4) & ~(legal_pairs & ~strict).any(dim=-1)
    order = th.sort(old_q.masked_fill(~legal, -th.inf), descending=True, stable=True).indices
    normalized, _, _, _ = reference_order(old_q, legal, q_scale_floor)
    _, _, _, full_filtered = full_rank_pairs(normalized, legal, decisions, dict(min_gap=min_gap))
    filtered = full_filtered[..., left, right]
    top = (left == order[..., :1]) | (right == order[..., :1])
    base = filtered & top
    distance = (order.argsort(-1)[..., left] - order.argsort(-1)[..., right]).abs()
    extra = filtered & ~top
    if min_gap:
        valid &= base.any(-1)
        for count in range(4, size+1):
            for k, quota in enumerate(quotas(count)):
                capacity = (extra & ((distance-1).clamp(0,2) == k)).sum(-1)
                valid &= (n != count) | (capacity >= int(quota))
    selected = (base & valid[..., None]).clone() if selection == 'k1' else th.zeros_like(legal_pairs)
    counts = th.stack((legal_pairs.sum(-1)*valid, strict.sum(-1)*valid, (base.sum(-1))*valid), dim=-1)
    flat_out = selected.reshape(-1, selected.size(-1))
    flat_q = old_q.reshape(-1, size)
    flat_order = order.reshape(-1, size)
    flat_valid, flat_n = valid.flatten(), n.flatten()
    flat_filtered = filtered.reshape(-1, filtered.size(-1))
    pair_id = th.full((size, size), -1, dtype=th.long)
    pair_id[left, right] = th.arange(len(left))
    closure_total = complete = ordering = selected_ordering = 0
    for count in range(4, size + 1):
        indices = (flat_valid & (flat_n == count)).nonzero().flatten()
        a, b = np.triu_indices(count, 1)
        for rows in indices.split(512):
            if not len(rows):
                continue
            ranks = flat_order[rows, :count]
            q = flat_q[rows].gather(-1, ranks).numpy()
            original_a, original_b = ranks[:, a], ranks[:, b]
            ids = pair_id[th.minimum(original_a, original_b), th.maximum(original_a, original_b)]
            permitted = flat_filtered[rows[:, None], ids].numpy() if min_gap else None
            if selection == 'k1':
                mask = np.zeros((len(rows), len(a)), dtype=bool)
                mask[:, a == 0] = True if permitted is None else permitted[:, a == 0]
                reached = mask.sum(axis=1)
            else:
                mask, reached = ranked_graphs(q, selection, permitted)
            flat_out[rows[:, None], ids] = th.from_numpy(mask)
            closure_total += int(reached.sum())
            complete += int((reached == count*(count-1)//2).sum())
            ordering += len(rows)*(count-1)
            selected_ordering += int(mask[:, b-a == 1].sum())
    audit = dict(complete_order_states=complete, ordering_edges=ordering,
                 selected_ordering_edges=selected_ordering, closure_pairs=closure_total,
                 gap_eligible_pairs=int((filtered & valid[..., None]).sum()),
                 gap_filtered_pairs=int((strict & ~filtered & valid[..., None]).sum()))
    return selected, valid, counts, audit


class TCPPreference(PairAblationPreference):
    statistics = staticmethod(distance_statistics)

    def __init__(self, selection, settings, rng, scheme, groups, log):
        super().__init__(selection, settings, rng, scheme, groups, log,
                         selector=select_tc_pairs, pair_audit=True)
        self.mask_root = None

    def prepare_pool(self, pool, *, prior_budget):
        if self.mask_root is None:
            return super().prepare_pool(pool, prior_budget=prior_budget)
        # Training only consumes frozen offline products. It never selects new edges.
        path = Path(self.mask_root) / self.selection / ('phase%04d.th' % pool['phase'])
        saved = th.load(path, map_location='cpu')
        masks, counts = saved['masks'], saved['counts']
        if len(masks) != len(pool['episodes']) or digest_masks(masks) != saved['audit']['pair_mask_sha256']:
            raise ValueError('invalid frozen TC mask artifact: ' + str(path))
        prepared = [dict(ep, pair_mask=mask, pair_counts=count)
                    for ep, mask, count in zip(pool['episodes'], masks, counts)]
        return prepared, saved['audit']
