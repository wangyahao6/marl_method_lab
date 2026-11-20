"""Fit Shapley transfer scores from Discovery and evaluate frozen coalitions on new SC2 traces."""
import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from analyze_shapley_frpr import (
    KEYS, PAIR_FIELDS, ROOT, collect_action_branches, csv_output, read, run, sha, write,
)
from drivers.continual.analysis.shapley_frpr import coalition_rows, exact_shapley
from drivers.continual.analysis.shapley_mc import batched_shapley, compare, fit_transfer, ordered_top, transferred_scores
from learners.selected_frpr import choose_pairs


def key(row):
    return str(row["task"]), str(row["episode_id"]), int(row["timestep"]), int(row["agent_id"])


def stratum(row):
    return row["battle_phase"], row["regret_stratum"]


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sample_across_episodes(items, quota, seed, *, min_pairs=1):
    """Give each different winning trajectory a draw before a second draw in a stratum."""
    groups = defaultdict(lambda: defaultdict(list))
    for item in items:
        row = item[0]
        if row["M"] >= min_pairs:
            groups[stratum(row)][row["episode_id"]].append(item)
    rng = np.random.default_rng(seed)
    for cell, episodes in sorted(groups.items()):
        episode_ids = sorted(episodes)
        count = quota if isinstance(quota, int) else quota.get("/".join(cell), 0)
        schedules = {ep: rng.permutation(len(episodes[ep])).tolist() for ep in episode_ids}
        while count and episode_ids:
            live = []
            for i in rng.permutation(len(episode_ids)):
                ep = episode_ids[int(i)]
                yield episodes[ep][schedules[ep].pop()]
                count -= 1
                if schedules[ep]:
                    live.append(ep)
                if not count:
                    break
            episode_ids = sorted(live)


def case_pairs(row, pairs):
    identity = dict(zip(KEYS, key(row)))
    return [dict(identity, participant=i, preferred_action=p, dispreferred_action=d)
            for i, (p, d) in enumerate(pairs)]


def collect(item, spec, cfg, writer, audit):
    row, _, pairs, q = item
    actions = json.loads(row["legal_actions"])
    row["base_q_legal"] = json.dumps([q[action] for action in actions])
    outcomes, raw = collect_action_branches(item, spec, cfg, audit)
    row["status"] = "complete" if len(outcomes) == cfg["rollout_repeats"] else "invalid_branch"
    return case_pairs(row, pairs), raw, [], []


def discovery_rows(folder, source, permitted_seeds):
    audit = read(folder / "audit.json")
    if not audit["passed"]:
        raise ValueError(f"Discovery audit failed: {folder}")
    records, branches = defaultdict(list), defaultdict(list)
    for row in load_csv(folder / "frpr_pairs.csv"):
        records[key(row)].append(row)
    for row in load_csv(folder / "action_returns.csv"):
        branches[key(row)].append(row)
    for row in load_csv(folder / "contexts.csv"):
        if row["status"] != "complete" or int(row["env_seed"]) not in permitted_seeds:
            continue
        identity = key(row)
        actions = json.loads(row["legal_actions"])
        ordered = sorted(records[identity], key=lambda row: int(row["participant"]))
        pairs = [(int(r["preferred_action"]), int(r["dispreferred_action"])) for r in ordered]
        q = np.zeros(len(json.loads(row["ft_q"])))
        q[actions] = json.loads(row["base_q_legal"])
        outcome_rows = branches[identity]
        repeats = sorted({int(r["repeat"]) for r in outcome_rows})
        observed = {(int(r["repeat"]), int(r["action"])): (float(r["post_intervention_return"]), r["won"] == "True")
                    for r in outcome_rows if r["valid"] == "True"}
        if (len(pairs) != int(row["M"]) or len(set(pairs)) != len(pairs) or not repeats
                or len(observed) != len(outcome_rows) or len(observed) != len(actions) * len(repeats)):
            raise ValueError(f"Incomplete or duplicated Discovery evidence for {identity}")
        outcomes = np.array([[observed[(repeat, a)] for a in actions] for repeat in repeats])
        yield row, pairs, q, actions, outcomes, source


def solve_discovery(data, cfg):
    row, pairs, q, actions, outcomes, source = data
    m = len(pairs)
    k = min(m, cfg["top_k"])
    identity = dict(zip(KEYS, key(row)))

    @lru_cache(maxsize=None)
    def value(mask):
        return next(coalition_rows(q, actions, pairs, outcomes, cfg["projection_margin"],
                                   cfg["softmax_temperature"], masks=[mask]))[3].mean(axis=0)

    exact, exact_seconds = None, None
    if m <= cfg["max_exact_pairs"]:
        started = perf_counter()
        values = np.array([value(mask) for mask in range(1 << m)])
        exact, _ = exact_shapley(values, m)
        exact_seconds = perf_counter() - started
    # A separate cache makes exact-versus-MC wall times comparable.
    value.cache_clear()
    seed = np.random.SeedSequence([cfg["sampling_seed"], int(row["env_seed"]), *key(row)[2:]])
    mc, se, convergence, diagnostics = batched_shapley(value, pairs, k, seed, cfg["monte_carlo"])
    used = exact if exact is not None else mc
    errors = []
    for i, metric in enumerate(("return", "win")):
        if exact is not None:
            errors.append(dict(identity, metric=metric, M=m, **compare(exact[:, i], mc[:, i], pairs, k)))
    summary = dict(identity, env_seed=int(row["env_seed"]), battle_phase=row["battle_phase"],
                   regret_stratum=row["regret_stratum"], M=m, source=source,
                   method="exact" if exact is not None else "permutation", exact_seconds=exact_seconds,
                   mc_coalitions=value.cache_info().currsize, **diagnostics)
    pair_rows = [dict(pair, phi_return=float(used[i, 0]), phi_win=float(used[i, 1]),
                      mc_return=float(mc[i, 0]), mc_win=float(mc[i, 1]),
                      mc_se_return=float(se[i, 0]), mc_se_win=float(se[i, 1]))
                 for i, pair in enumerate(case_pairs(row, pairs))]
    return (stratum(row), pairs, used[:, 0]), summary, pair_rows, errors, [dict(identity, **x) for x in convergence]


def validate_context(item, spec, cfg, audit, fit, selected_rows, outcomes_rows):
    row, _, pairs, q = item
    actions = json.loads(row["legal_actions"])
    identity = dict(zip(KEYS, key(row)))
    scores, seen = transferred_scores(stratum(row), pairs, fit)
    k = min(len(pairs), cfg["top_k"])
    selection = ordered_top(scores, pairs, k)
    indices = {pair: i for i, pair in enumerate(pairs)}
    arms = [("empty", "", []), ("full", "", list(range(len(pairs)))), ("shapley_topk", "", selection)]
    for seed in cfg["random_seeds"]:
        chosen = choose_pairs(key(row), dict(zip(pairs, scores)),
                              dict(selection_k=k, selection_seed=seed), random_control=True)
        arms.append(("random_topk", seed, [indices[pair] for pair in chosen]))
    by_mask = defaultdict(list)
    for arm, seed, chosen in arms:
        mask = sum(1 << i for i in chosen)
        by_mask[mask].append((arm, seed))
        selected_rows.append(dict(identity, arm=arm, random_seed=seed, mask=mask,
                                  selected_pairs=json.dumps([pairs[i] for i in chosen])))
    row.update(transfer_observed_pairs=sum(seen), transfer_observed_selected=sum(seen[i] for i in selection),
               selected_count=k, base_q_legal=json.dumps([q[a] for a in actions]))
    outcomes, raw = collect_action_branches(item, spec, cfg, audit)
    if len(outcomes) != cfg["rollout_repeats"]:
        row["status"] = "invalid_branch"
        return case_pairs(row, pairs), raw, [], []
    for mask, _, _, values, diag in coalition_rows(q, actions, pairs, outcomes, cfg["projection_margin"],
                                                  cfg["softmax_temperature"], masks=by_mask):
        audit["coalitions"] += 1
        for name in ("projection_residual", "stationarity_residual"):
            audit[name] = max(audit[name], diag[name])
        for arm, seed in by_mask[mask]:
            outcomes_rows.append(dict(identity, arm=arm, random_seed=seed, mask=mask,
                                      outcome_return=float(values[:, 0].mean()), outcome_win=float(values[:, 1].mean())))
    row["status"] = "complete"
    return case_pairs(row, pairs), raw, [], []


def run_case(config, out):
    cfg = read(config)
    if cfg["max_exact_pairs"] != 12:
        raise ValueError("Use the standard exact cap of 12")
    if len(set(cfg["random_seeds"])) != len(cfg["random_seeds"]) or len(cfg["random_seeds"]) < 2:
        raise ValueError("Use multiple distinct Random-TopK seeds")
    out.mkdir(parents=True, exist_ok=False)
    case_config = dict(cfg["environment"], **{k: cfg[k] for k in (
        "projection_margin", "softmax_temperature", "max_exact_pairs", "top_k")})
    case_config.pop("inventory_pool", None)
    pilot = ROOT / cfg["pilot"]
    prior = read(pilot / "manifest.json")
    for name in ("old_checkpoint", "ft_checkpoint", "task", "projection_margin", "softmax_temperature", "frpr"):
        if prior["config"][name] != case_config[name]:
            raise ValueError(f"Pilot provenance mismatch for {name}")
    for name in ("old_checkpoint", "ft_checkpoint"):
        path = ROOT / case_config[name]
        if not any(Path(record["path"]) == path and sha(path) == record["sha256"] for record in prior["inputs"]):
            raise ValueError(f"Pilot checkpoint hash mismatch for {name}")
    pilot_seeds = {int(pair[0]) for pair in prior["config"]["seeds"]}
    discovery_seeds = {int(pair[0]) for pair in cfg["discovery_seeds"]}
    validation_seeds = {int(pair[0]) for pair in cfg["validation_seeds"]}
    validation_sc2 = {int(pair[1]) for pair in cfg["validation_seeds"]}
    discovery_pairs = [*cfg["discovery_seeds"], *(pair for pair in prior["config"]["seeds"]
                                                if pair[0] in cfg["pilot_discovery_env_seeds"])]
    if (not set(cfg["pilot_discovery_env_seeds"]) <= pilot_seeds
            or any(pair[0] in validation_seeds or pair[1] in validation_sc2 for pair in discovery_pairs)):
        raise ValueError("Discovery and Validation seeds must be disjoint")
    snapshot = ("scripts/validate_shapley_frpr.py",)
    discovery = out / "discovery"
    validation = out / "validation"
    for label, quota in (("discovery", cfg["discovery_per_stratum"]), ("validation", cfg["validation_per_stratum"])):
        write(out / f"{label}_config.json", dict(case_config, rollout_repeats=1, per_stratum=quota, seeds=cfg[f"{label}_seeds"]))
    discovery_status = run(out / "discovery_config.json", discovery, evaluator=collect, snapshot_paths=snapshot,
                           sampler=lambda rows, q, s: sample_across_episodes(rows, q, s, min_pairs=13))
    if discovery_status:
        raise RuntimeError("No fully verified high-M Discovery data; see discovery/audit.json")
    data = [*discovery_rows(pilot, "pilot_exact", set(cfg["pilot_discovery_env_seeds"])),
            *discovery_rows(discovery, "new_high_M", discovery_seeds)]
    if len({key(row) for row, *_ in data}) != len(data):
        raise ValueError("Repeated Discovery context")
    if not any(len(pairs) <= 12 for _, pairs, *_ in data):
        raise ValueError("No exact comparison states in Discovery")
    results = [solve_discovery(record, cfg) for record in data]
    observations, stats, pairs, errors, convergence = zip(*results)
    fit = fit_transfer(observations, cfg["stratum_shrinkage"])
    global_scores, local = fit
    model = dict(config_sha256=sha(config), discovery_env_seeds=sorted({row["env_seed"] for row in stats}),
                 cases=len(stats), pilot_inputs={name: sha(pilot / name) for name in (
                     "manifest.json", "audit.json", "contexts.csv", "frpr_pairs.csv", "action_returns.csv")},
                 discovery_context_ids=sorted({key(row)[1] for row, *_ in data}),
                 global_scores=[[*pair, *val] for pair, val in sorted(global_scores.items())],
                 local_scores=[[*cell, *val] for cell, val in sorted(local.items())])
    write(out / "frozen_discovery.json", model)
    csv_output(out / "discovery_contexts.csv", stats)
    csv_output(out / "discovery_shapley.csv", [r for group in pairs for r in group])
    csv_output(out / "exact_comparison.csv", [r for group in errors for r in group])
    csv_output(out / "convergence.csv", [r for group in convergence for r in group])
    chosen, outcomes = [], []

    def follow_frozen(item, spec, settings, writer, audit):
        if item[0]["episode_id"] in model["discovery_context_ids"]:
            raise ValueError("Validation episode overlaps Discovery")
        return validate_context(item, spec, cfg, audit, fit, chosen, outcomes)

    try:
        return run(out / "validation_config.json", validation, evaluator=follow_frozen,
                   sampler=sample_across_episodes, snapshot_paths=snapshot)
    finally:
        csv_output(out / "validation_selections.csv", chosen, KEYS)
        csv_output(out / "validation_values.csv", outcomes, KEYS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_case(args.config, args.output))
