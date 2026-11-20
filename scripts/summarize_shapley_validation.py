"""Audit a Shapley-FRPR validation run and export lightweight paired results."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import shutil

import numpy as np

from validate_shapley_frpr import KEYS, csv_output, key, load_csv, read, sha, stratum, write
from drivers.continual.analysis.shapley_mc import compare, fit_transfer, ordered_top, transferred_scores


ARMS = ("empty", "full", "random_topk", "shapley_topk")
CONTRASTS = (("shapley_topk", "random_topk"), ("shapley_topk", "full"),
             ("shapley_topk", "empty"), ("full", "empty"))


def bootstrap(rows, values, draws, seed):
    """Resample paired initial-seed blocks within each task, across endpoints and models."""
    clusters = defaultdict(lambda: defaultdict(list))
    for row, value in zip(rows, values):
        clusters[row["task"]][row["env_seed"]].append(value)
    count = sum(len(task) for task in clusters.values())
    if any(len(task) < 2 for task in clusters.values()):
        return count, None, None
    rng = np.random.default_rng(seed)
    totals = np.zeros(draws)
    sizes = np.zeros(draws, dtype=int)
    for group in clusters.values():
        values = list(group.values())
        n = len(values)
        ids = rng.integers(n, size=(draws, n))
        totals += np.array([sum(block) for block in values])[ids].sum(axis=1)
        sizes += np.array([len(block) for block in values])[ids].sum(axis=1)
    low, high = np.quantile(totals / sizes, (.025, .975)).tolist()
    return count, low, high


def paired_rows(case, folder, random_seeds):
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in load_csv(folder / "validation_values.csv"):
        arm, seed = row["arm"], row["random_seed"]
        context = grouped[key(row)][arm]
        if seed in context:
            raise ValueError(f"Duplicate arm/seed in {case}: {key(row)}, {arm}, {seed}")
        context[seed] = float(row["outcome_return"]), float(row["outcome_win"])
    for row in load_csv(folder / "validation" / "contexts.csv"):
        if row["status"] != "complete":
            continue
        arms = grouped.pop(key(row))
        if set(arms) != set(ARMS) or set(arms["random_topk"]) != set(map(str, random_seeds)) or any(
                set(arms[arm]) != {""} for arm in ARMS if arm != "random_topk"):
            raise ValueError(f"Unpaired validation arms in {case}: {key(row)}")
        result = {name: row[name] for name in (*KEYS, "battle_phase", "regret_stratum", "env_seed", "M",
                                                "transfer_observed_pairs", "transfer_observed_selected",
                                                "selected_count")}
        result.update(case=case, training_seed=int(case.split("_")[0]), outcome_phase=case.split("_", 1)[1],
                      m_regime="M>12" if int(row["M"]) > 12 else "M<=12")
        for arm in ARMS:
            samples = np.asarray(list(arms[arm].values()))
            for i, metric in enumerate(("return", "win")):
                result[f"{arm}_{metric}"] = float(samples[:, i].mean())
                if arm == "random_topk":
                    result[f"{arm}_{metric}_sd"] = float(samples[:, i].std(ddof=1))
        yield result
    if grouped:
        raise ValueError(f"Validation outcomes lack verified contexts: {case}")


def summaries(rows, cfg):
    groups = defaultdict(list)
    for row in rows:
        for category in ("case", "training_seed", "task", "battle_phase", "regret_stratum", "outcome_phase", "m_regime"):
            groups[category, row[category]].append(row)
    groups["aggregate", "all"] = rows
    for (dimension, name), group in groups.items():
        if not group:
            continue
        for metric in ("return", "win"):
            arm_means = {arm: float(np.mean([row[f"{arm}_{metric}"] for row in group])) for arm in ARMS}
            for arm, reference in CONTRASTS:
                diffs = [row[f"{arm}_{metric}"] - row[f"{reference}_{metric}"] for row in group]
                clusters, lo, hi = bootstrap(group, diffs, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
                yield dict(dimension=dimension, group=name, metric=metric, arm=arm, reference=reference,
                           contexts=len(group), seed_clusters=clusters, arm_mean=arm_means[arm],
                           reference_mean=arm_means[reference], paired_delta=float(np.mean(diffs)),
                           ci_low=lo, ci_high=hi, positive_fraction=float(np.mean(np.asarray(diffs) > 1e-12)))


def tagged(case, path):
    return [dict(row, case=case) for row in load_csv(path)]


def avg(rows, column):
    selected = [float(row[column]) for row in rows if row[column] not in ("", None)]
    return float(np.mean(selected)) if selected else None


def approx_summary(rows, checks, errors):
    groups = defaultdict(list)
    for row in rows:
        groups[row["case"], row["method"]].append(row)
        groups["all", row["method"]].append(row)
    for (case, method), group in groups.items():
        ids = {(row["case"], *key(row)) for row in group}
        for metric in ("return", "win"):
            reference = [r for r in errors if (r["case"], *key(r)) in ids and r["metric"] == metric]
            final_check = {}
            for row in checks:
                index = row["case"], *key(row)
                if index in ids and row["metric"] == metric:
                    final_check[index] = row
            last = list(final_check.values())
            yield dict(case=case, mode=method, metric=metric, contexts=len(group),
                       converged=sum(row["converged"] == "True" for row in group),
                       mean_permutations=avg(group, "permutations"), mean_mc_seconds=avg(group, "mc_seconds"),
                       mean_exact_seconds=avg(group, "exact_seconds"), rank_defined=sum(bool(r["spearman"]) for r in reference),
                       exact_mean_spearman=avg(reference, "spearman"), exact_mean_topk_fraction=avg(reference, "top_fraction"),
                       exact_mean_mae=avg(reference, "mae"), final_batch_spearman=avg(last, "batch_spearman"),
                       final_batch_topk_fraction=avg(last, "batch_top_fraction"),
                       final_cumulative_max_change=avg(last, "cumulative_max_error"))


def read_pairs(path):
    grouped = defaultdict(list)
    for row in load_csv(path):
        grouped[key(row)].append(row)
    return {context: sorted(rows, key=lambda row: int(row["participant"])) for context, rows in grouped.items()}


def relation(row):
    return int(row["preferred_action"]), int(row["dispreferred_action"])


def transfer_stability(case, folder, contexts, shrinkage):
    """Leave each Discovery trajectory seed out in turn, with no held-out outcomes used."""
    discovery_pairs = read_pairs(folder / "discovery_shapley.csv")
    known = [(int(row["env_seed"]), stratum(row), [relation(pair) for pair in discovery_pairs[key(row)]],
              [float(pair["phi_return"]) for pair in discovery_pairs[key(row)]])
             for row in load_csv(folder / "discovery_contexts.csv")]
    seeds = sorted({record[0] for record in known})
    fit = fit_transfer([data[1:] for data in known], shrinkage)
    leaveouts = [(seed, fit_transfer([data[1:] for data in known if data[0] != seed], shrinkage))
                 for seed in seeds] if len(seeds) > 1 else []
    candidates = read_pairs(folder / "validation" / "frpr_pairs.csv")
    selected = {key(row): [tuple(pair) for pair in json.loads(row["selected_pairs"])]
                for row in load_csv(folder / "validation_selections.csv") if row["arm"] == "shapley_topk"}
    for row in contexts:
        pairs = [relation(pair) for pair in candidates[key(row)]]
        baseline = np.asarray(transferred_scores(stratum(row), pairs, fit)[0])
        k = int(row["selected_count"])
        if [pairs[i] for i in ordered_top(baseline, pairs, k)] != selected[key(row)]:
            raise ValueError(f"Frozen ranking does not reproduce selected pairs for {key(row)}")
        for seed, leaveout in leaveouts:
            held = np.asarray(transferred_scores(stratum(row), pairs, leaveout)[0])
            yield dict(case=case, **dict(zip(KEYS, key(row))), removed_discovery_env_seed=seed,
                       **compare(baseline, held, pairs, k))


def pilot_equivalence(folder, pilot):
    """Check the exact bridge against the original Pilot using the same action evidence."""
    sources = {key(row): row["source"] for row in load_csv(folder / "discovery_contexts.csv")}
    current = [row for row in load_csv(folder / "discovery_shapley.csv")
               if sources[key(row)] == "pilot_exact"]
    original = {(key(row), relation(row)): row for row in load_csv(pilot / "shapley_values.csv")}
    errors = [abs(float(row[metric]) - float(original[key(row), relation(row)][metric]))
              for row in current for metric in ("phi_return", "phi_win")]
    max_error = max(errors, default=0.)
    if max_error > 1e-9:
        raise ValueError(f"Exact results differ from the Pilot in {folder}: {max_error}")
    return dict(relations=len(current), max_absolute_difference=max_error)


def check_case(folder, config):
    frozen = read(folder / "frozen_discovery.json")
    cfg = read(config)
    if frozen["config_sha256"] != sha(config):
        raise ValueError(f"Frozen Discovery config mismatch in {folder}")
    discovery = read(folder / "discovery" / "audit.json")
    validation = read(folder / "validation" / "audit.json")
    manifests = {label: read(folder / label / "manifest.json") for label in ("discovery", "validation")}
    observed_discovery = set(frozen["discovery_env_seeds"])
    observed_validation = {x["env_seed"] for x in validation["source_attempts"]}
    completed = [row for row in load_csv(folder / "validation" / "contexts.csv") if row["status"] == "complete"]
    if observed_discovery & observed_validation or any(row["episode_id"] in frozen["discovery_context_ids"] for row in completed):
        raise ValueError(f"Data split overlap in {folder}")
    if len(completed) != validation["completed_contexts"]:
        raise ValueError(f"Validation context count mismatch in {folder}")
    provenance = {label: {key: data[key] for key in ("code_commit", "code_dirty", "source_archive_sha256",
                                                     "python", "torch", "scipy", "numpy", "env_args", "traces", "inputs")}
                  for label, data in manifests.items()}
    return dict(discovery=discovery, validation=validation, input_provenance=provenance,
                frozen_discovery_sha256=sha(folder / "frozen_discovery.json"),
                pilot_input_hashes=frozen["pilot_inputs"],
                discovery_env_seeds=sorted(observed_discovery),
                validation_env_seeds=sorted(observed_validation),
                validation_winning_episodes=len({row["episode_id"] for row in completed}),
                exact_pilot_reproduction=pilot_equivalence(folder, Path(cfg["pilot"])), overlap=False)


def summarize(run, out, smoke=None):
    protocol = read(run / "protocol.json")
    plan = read(run / "plan.json")
    out.mkdir(parents=True, exist_ok=False)
    configs = out / "configs"
    configs.mkdir()
    models = out / "frozen_discovery"
    models.mkdir()
    all_contexts, checks, errors, wins, choices, fits, stability, inventory = [], [], [], [], [], [], [], []
    audit = dict(run=run.name, plan_sha256=sha(run / "plan.json"), cases={}, failures={})
    if smoke is not None:
        name = "2222_s_after_z"
        audit["smoke"] = check_case(smoke / name, smoke / f"{name}.json")
    for task in plan["tasks"]:
        case = task["id"][len(run.name) + 1:]
        folder = run / case
        config = run / f"{case}.json"
        shutil.copyfile(config, configs / config.name)
        try:
            evidence = check_case(folder, config)
            audit["cases"][case] = evidence
            if not (evidence["discovery"]["passed"] and evidence["validation"]["passed"]):
                audit["failures"][case] = "One or more trace or branch audits failed"
            case_contexts = list(paired_rows(case, folder, protocol["random_seeds"]))
            shutil.copyfile(folder / "frozen_discovery.json", models / f"{case}.json")
            inventory.extend({name: row.get(name) for name in (
                "case", *KEYS, "env_seed", "battle_phase", "regret_stratum", "M", "status", "branch_rollouts",
                "transfer_observed_pairs", "transfer_observed_selected", "selected_count")}
                for row in tagged(case, folder / "validation" / "contexts.csv"))
            if len(case_contexts) != evidence["validation"]["completed_contexts"]:
                raise ValueError(f"Paired rows differ from audited context count: {case}")
            for dest, file in ((all_contexts, "discovery_contexts.csv"), (checks, "convergence.csv"),
                               (errors, "exact_comparison.csv"), (choices, "validation_selections.csv"),
                               (fits, "discovery_shapley.csv")):
                dest.extend(tagged(case, folder / file))
            case_stability = list(transfer_stability(case, folder, case_contexts, protocol["stratum_shrinkage"]))
            stability.extend(case_stability)
            wins.extend(case_contexts)
        except (OSError, ValueError) as exc:
            audit["failures"][case] = str(exc)
    comparisons = list(summaries(wins, protocol))
    approx = list(approx_summary(all_contexts, checks, errors))
    for file, rows in (("discovery_contexts.csv", all_contexts), ("discovery_shapley.csv", fits),
                       ("convergence.csv", checks), ("exact_comparison.csv", errors),
                       ("approximation_summary.csv", approx), ("validation_selections.csv", choices),
                       ("validation_contexts.csv", wins), ("validation_inventory.csv", inventory),
                       ("transfer_stability.csv", stability),
                       ("paired_summary.csv", comparisons)):
        csv_output(out / file, rows, ("case",) if file != "paired_summary.csv" else ("dimension", "group"))
    audit["discovery_contexts"] = len(all_contexts)
    audit["discovery_by_method"] = dict(Counter(row["method"] for row in all_contexts))
    audit["discovery_episode_ids"] = len({(row["case"].split("_")[0], row["task"], row["episode_id"])
                                          for row in all_contexts})
    audit["validation_contexts"] = len(wins)
    audit["validation_episode_ids"] = len({(row["training_seed"], row["task"], row["episode_id"]) for row in wins})
    audit["selected_relations"] = sum(int(row["selected_count"]) for row in wins)
    audit["discovery_supported_selected"] = sum(int(row["transfer_observed_selected"]) for row in wins)
    audit["passed"] = len(audit["cases"]) == len(plan["tasks"]) and not audit["failures"]
    write(out / "audit.json", audit)
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", type=Path)
    args = parser.parse_args()
    raise SystemExit(summarize(args.run, args.output, args.smoke))
