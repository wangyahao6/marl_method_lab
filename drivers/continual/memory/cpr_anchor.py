"""CPR-Anchor: accept stronger references only along the original Top-1 anchor."""
import torch as th

from learners.ranking_preference import reference_order
from .cpr import update_pool as update_cpr_pool


@th.no_grad()
def select_reference(old_q, new_q, legal, decisions, valid, settings, *, original_q,
                     with_confidence=False):
    """Rejecting an update leaves the existing Q, validity and provenance untouched."""
    eligible = decisions & (legal.sum(dim=-1) >= 2)
    _, _, original_top, _ = reference_order(original_q, legal, settings["q_scale_floor"])
    _, _, old_top, old_conf = reference_order(old_q, legal, settings["q_scale_floor"])
    _, _, new_top, new_conf = reference_order(new_q, legal, settings["q_scale_floor"])
    threshold = max(settings["reference_confidence_threshold"], settings["min_gap"])
    update = (eligible & (new_top == original_top).squeeze(-1) & (new_conf >= threshold)
              & (new_conf > old_conf + settings["reference_switch_margin"]))
    keep = eligible & ~update
    uncertain = th.zeros_like(eligible)
    changed = (old_top != new_top).squeeze(-1)
    result = (update, keep, uncertain, eligible, changed)
    return (*result, old_conf, new_conf) if with_confidence else result


def update_pool(*args, **kwargs):
    """Reuse CPR's gate, no-grad online RNN unroll, chunks, persistence and diagnostics."""
    return update_cpr_pool(*args, **kwargs, reference_selector=select_reference)
