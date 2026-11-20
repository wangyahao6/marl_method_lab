"""Permutation estimates and Discovery-only transfer for Shapley-FRPR."""
from collections import defaultdict
from time import perf_counter

import numpy as np
from scipy.stats import rankdata


def ranks(values):
    # Suppress projection round-off on mathematically tied margins.
    return rankdata(-np.round(values, 12), method="average")


def rank_agreement(a, b):
    first, second = ranks(a), ranks(b)
    if not np.ptp(first) or not np.ptp(second):
        return None
    return float(np.corrcoef(first, second)[0, 1])


def ordered_top(values, pairs, k):
    return sorted(range(len(values)), key=lambda i: (-round(float(values[i]), 12), pairs[i]))[:k]


def compare(a, b, pairs, k):
    left = set(ordered_top(a, pairs, k))
    right = set(ordered_top(b, pairs, k))
    return dict(spearman=rank_agreement(a, b), top_fraction=len(left & right) / len(left),
                top_jaccard=len(left & right) / len(left | right),
                mae=float(np.abs(a - b).mean()), max_error=float(np.abs(a - b).max()))


def batched_shapley(value, pairs, k, seed, settings):
    """Use uniform full permutations. Each permutation telescopes from Empty to Full."""
    m = len(pairs)
    batch = settings["batch"]
    minimum = settings["minimum"]
    maximum = settings["maximum"]
    if not (0 < batch <= minimum <= maximum and minimum % batch == maximum % batch == 0):
        raise ValueError("Permutation limits must be positive multiples of the batch size")
    rng = np.random.default_rng(seed)
    start = perf_counter()
    target = value((1 << m) - 1) - value(0)
    total = np.zeros((m, len(target)))
    squares = np.zeros_like(total)
    checks = []
    previous = None
    stable_checks = 0
    max_efficiency = 0.
    for n in range(batch, maximum + 1, batch):
        marginal = np.empty((batch, m, len(target)))
        for offset in range(batch):
            mask = 0
            base = value(mask)
            for index in rng.permutation(m).tolist():
                mask |= 1 << index
                updated = value(mask)
                marginal[offset, index] = updated - base
                base = updated
        max_efficiency = max(max_efficiency, float(np.abs(marginal.sum(axis=1) - target).max()))
        if max_efficiency > 1e-9:
            raise ValueError(f"Permutation efficiency mismatch: {max_efficiency}")
        total += marginal.sum(axis=0)
        squares += (marginal ** 2).sum(axis=0)
        mean = total / n
        if previous is not None:
            batch_mean = marginal.mean(axis=0)
            batch_check = [compare(previous[:, i], batch_mean[:, i], pairs, k) for i in range(len(target))]
            step_check = [compare(previous[:, i], mean[:, i], pairs, k) for i in range(len(target))]
            # Discovery selection uses return. Report win stability independently.
            tol = settings["abs_tol"] + settings["rel_tol"] * float(np.abs(mean[:, 0]).max())
            stable = (step_check[0]["max_error"] <= tol
                      and batch_check[0]["top_fraction"] >= settings["min_top_overlap"])
            stable_checks = stable_checks + 1 if stable else 0
            for i, metric in enumerate(("return", "win")):
                checks.append(dict(permutations=n, metric=metric, stable_return_batches=stable_checks,
                                   **{f"batch_{name}": val for name, val in batch_check[i].items()},
                                   **{f"cumulative_{name}": val for name, val in step_check[i].items()}))
        if n >= minimum and stable_checks >= settings["stable_batches"] and settings["stop_early"]:
            break
        previous = mean
    stderr = (np.sqrt(np.maximum((squares - total ** 2 / n) / (n - 1), 0) / n)
              if n > 1 else np.zeros_like(total))
    return mean, stderr, checks, dict(permutations=n, converged=stable_checks >= settings["stable_batches"],
                                      efficiency_error=max_efficiency, mc_seconds=perf_counter() - start)


def fit_transfer(observations, shrinkage):
    """Fit case-global and matched-stratum percentile ranks on Discovery alone."""
    strata = defaultdict(list)
    overall = defaultdict(list)
    for stratum, pairs, values in observations:
        m = len(pairs)
        percentiles = 1 - (ranks(values) - 1) / (m - 1) if m > 1 else np.full(m, .5)
        for pair, percentile in zip(pairs, percentiles):
            overall[pair].append(float(percentile))
            strata[(*stratum, *pair)].append(float(percentile))
    prior = {pair: (float(np.mean(values)), len(values)) for pair, values in overall.items()}
    cells = {}
    for key, values in strata.items():
        # Pool toward the case-global evidence for this same directed action pair.
        baseline, _ = prior[key[-2:]]
        cells[key] = ((sum(values) + shrinkage * baseline) / (len(values) + shrinkage), len(values))
    return prior, cells


def transferred_scores(stratum, pairs, fit):
    global_scores, local = fit
    seen = [pair in global_scores for pair in pairs]
    scores = [local.get((*stratum, *pair), global_scores.get(pair, (.5, 0)))[0] for pair in pairs]
    return scores, seen
