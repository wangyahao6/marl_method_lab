"""CPR reference updates; trajectories and original targets are never replaced."""
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch as th

from learners.action_set_preference import batch_episodes, decision_masks, unroll
from learners.ranking_preference import reference_order
from . import History, PreferenceHistory


def initialize_pool(pool):
    """Initialize only legacy/static records; preserve already accepted CPR references."""
    fields = ("original_q_values", "reference_valid_mask", "reference_source_phase")
    for ep in pool["episodes"]:
        q = ep["data"]["old_q_values"]
        shape = ep["data"]["avail_actions"].shape
        if q.shape != shape:
            raise ValueError("CPR old_q_values must align with avail_actions")
        present = [field in ep for field in fields]
        if not any(present):
            ep["original_q_values"] = q.clone()
            ep["reference_valid_mask"] = th.ones(shape[:-1], dtype=th.bool)
            ep["reference_source_phase"] = th.full(shape[:-1], pool["phase"], dtype=th.long)
        elif (not all(present) or ep["original_q_values"].shape != shape
              or ep["reference_valid_mask"].shape != shape[:-1]
              or ep["reference_source_phase"].shape != shape[:-1]
              or ep["reference_valid_mask"].dtype != th.bool
              or ep["reference_source_phase"].dtype != th.long):
            raise ValueError("CPR metadata must align with the full episode time/agent axes")
    pool["reference_checkpoint_role"] = "original_collection_only"
    pool.setdefault("reference_checkpoints", {pool["phase"]: pool["reference_checkpoint"]})
    return pool


@th.no_grad()
def select_reference(old_q, new_q, legal, decisions, valid, settings, *, with_confidence=False):
    """Return disjoint update/keep/uncertain masks over real legal decisions."""
    eligible = decisions & (legal.sum(dim=-1) >= 2)
    _, _, old_top, old_conf = reference_order(old_q, legal, settings["q_scale_floor"])
    _, _, new_top, new_conf = reference_order(new_q, legal, settings["q_scale_floor"])
    threshold, margin = settings["reference_confidence_threshold"], settings["reference_switch_margin"]
    new_good = new_conf >= threshold
    both_low = (old_conf < threshold) & ~new_good
    same = old_top.squeeze(-1) == new_top.squeeze(-1)
    update = eligible & new_good & (~valid | (new_conf > old_conf + margin))
    keep = eligible & ~update & valid & ~both_low & (same | (old_conf > new_conf + margin))
    uncertain = eligible & ~update & ~keep
    result = (update, keep, uncertain, eligible, ~same)
    return (*result, old_conf, new_conf) if with_confidence else result


def confidence_summary(parts):
    """Diagnostic only: quantiles cannot alter reference selection or halt a run."""
    try:
        values = th.cat(parts).double() if parts else th.empty(0, dtype=th.float64)
        finite = values[th.isfinite(values)]
        if not finite.numel():
            return dict(n=0, nonfinite=int(values.numel()), mean=None, median=None, p10=None, p90=None)
        quantiles = th.quantile(finite, th.tensor([.1, .5, .9], dtype=th.float64)).tolist()
        return dict(n=int(finite.numel()), nonfinite=int(values.numel() - finite.numel()),
                    mean=finite.mean().item(), median=quantiles[1], p10=quantiles[0], p90=quantiles[2])
    except Exception as error:
        return dict(error=str(error))


@th.no_grad()
def update_pool(mac, pool, scheme, groups, settings, phase, current_win_rate,
                historical_best_win_rate, checkpoint, *, chunk_size=8):
    """Re-evaluate complete episodes using the online MAC, only after the task gate."""
    start = perf_counter()
    gate = current_win_rate >= historical_best_win_rate - settings["task_gate_tolerance"]
    totals = Counter()
    sources = Counter()
    old_confidences, new_confidences = [], []
    for offset in range(0, len(pool["episodes"]), chunk_size):
        episodes = pool["episodes"][offset:offset + chunk_size]
        device = mac.args.device if gate else "cpu"
        batch, _ = batch_episodes(episodes, scheme, groups, device,
                                  include_good=False, include_old_q=True)
        legal, decisions = decision_masks(batch)
        decisions = decisions & (legal.sum(dim=-1) >= 2)
        current = unroll(mac, batch, batch.max_seq_length - 1) if gate else None
        for index, ep in enumerate(episodes):
            steps = ep["length"]
            real = decisions[index, :steps]
            totals["decision_states"] += int(real.sum())
            old_valid = ep["reference_valid_mask"][:steps].to(device)
            if gate:
                update, keep, uncertain, eligible, changed, old_conf, new_conf = select_reference(
                    batch["old_q_values"][index, :steps], current[index, :steps],
                    legal[index, :steps], real, old_valid, settings, with_confidence=True)
                old_confidences.append(old_conf[real].cpu())
                new_confidences.append(new_conf[real].cpu())
                update, keep, uncertain = (mask.cpu() for mask in (update, keep, uncertain))
                ep["data"]["old_q_values"][:steps][update] = current[index, :steps].cpu()[update]
                ep["reference_valid_mask"][:steps][update] = True
                ep["reference_valid_mask"][:steps][uncertain] = False
                ep["reference_source_phase"][:steps][update] = phase
                ep["reference_source_phase"][:steps][uncertain] = -1
                totals["updated_states"] += int(update.sum())
                totals["retained_states"] += int(keep.sum())
                totals["uncertain_states"] += int(uncertain.sum())
                totals["reference_top1_changed_states"] += int((update.to(device) & changed).sum())
            else:
                # Gate failure leaves Q, mask and provenance entirely untouched.
                _, _, _, old_conf = reference_order(batch["old_q_values"][index, :steps],
                                                    legal[index, :steps], settings["q_scale_floor"])
                old_confidences.append(old_conf[real].cpu())
                totals["retained_states"] += int((real & old_valid).sum())
                totals["uncertain_states"] += int((real & ~old_valid).sum())
            active = ep["reference_valid_mask"][:steps] & real.cpu()
            totals["valid_reference_states"] += int(active.sum())
            sources.update(ep["reference_source_phase"][:steps][real.cpu()].tolist())
    if gate:
        pool["reference_checkpoints"][phase] = checkpoint
    count = totals["decision_states"]
    return dict(task=pool["task"], pool_phase=pool["phase"], phase=phase,
                current_win_rate=current_win_rate, historical_best_win_rate=historical_best_win_rate,
                task_gate_passed=gate, reevaluated_episodes=len(pool["episodes"]) if gate else 0,
                decision_states=count, updated_states=totals["updated_states"],
                retained_states=totals["retained_states"], uncertain_states=totals["uncertain_states"],
                reference_top1_changed_states=totals["reference_top1_changed_states"],
                reference_top1_change_rate=totals["reference_top1_changed_states"] / count if count else None,
                valid_reference_states=totals["valid_reference_states"],
                valid_preference_coverage=totals["valid_reference_states"] / count if count else None,
                reference_source_counts=dict(sources),
                old_reference_confidence=confidence_summary(old_confidences),
                new_reference_confidence=confidence_summary(new_confidences) if gate else None,
                reevaluation_seconds=perf_counter() - start)


def restore_history(history, parent, phases, out):
    """Reuse success.th restore, accounting for revisits with no newly created pool."""
    parent = Path(parent)
    for phase in phases:
        history.restore_success(parent, [phase], require_old_q=True)
        pool = initialize_pool(history[phase])
        pool["reference_checkpoints"] = {
            source: str(parent / path) for source, path in pool["reference_checkpoints"].items()}
        # A branch owns copies; updates must never overwrite parent artifacts.
        History.save(PreferenceHistory.phase_path(out, phase) / "success.th", pool)
