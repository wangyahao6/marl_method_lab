"""Thin single-agent adapter for the repository's ranking losses."""

import torch

from learners.ranking_preference import (
    full_rank_preference, topk_preference, topk_preference_weighted,
    topk_statistics, topk_weighted_statistics,
)


METHODS = {
    "DQN-TKPR-K1": (topk_preference, 1),
    "DQN-TKPR-K1-W": (topk_preference_weighted, 1),
    "DQN-TKPR-K2": (topk_preference, 2),
}


def preference_result(method, current_q, old_q_values, settings):
    function, k = METHODS[method]
    options = dict(settings, k=k)
    legal = torch.ones_like(old_q_values, dtype=torch.bool)
    decisions = torch.ones(old_q_values.shape[0], dtype=torch.bool, device=old_q_values.device)
    return function(current_q, old_q_values.detach(), legal, decisions, options)


def preference_loss(method, current_q, old_q_values, settings):
    if method in ("DQN-FT", "DQN-ER"):
        return current_q.sum() * 0
    return preference_result(method, current_q, old_q_values, settings)[0]


def preference_statistics(method, summary):
    values = summary.detach().cpu().tolist()
    return (topk_weighted_statistics(values) if method == "DQN-TKPR-K1-W"
            else topk_statistics(values))


def frpr_loss(current_q, old_q_values, settings):
    """Reference used to check K2 equivalence with three legal actions."""
    legal = torch.ones_like(old_q_values, dtype=torch.bool)
    decisions = torch.ones(old_q_values.shape[0], dtype=torch.bool, device=old_q_values.device)
    return full_rank_preference(current_q, old_q_values.detach(), legal, decisions, settings)[0]
