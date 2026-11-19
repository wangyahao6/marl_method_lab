"""Run a frozen-policy SMACv2 Shapley-FRPR pilot, without training."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.stats import rankdata, t
import torch as th

from verify_smacv2_counterfactual import FrozenPolicy, add_action, add_observation, branch, offline_records, open_source, select
from verify_smacv2_replay import ROOT, differences, new_seeded_env, play, read, sha, store_trace, trace_comparator, write

sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "drivers/continual")]
from action_set_protocol import RANK_DEFAULTS, nonnegative, positive_integer
from drivers.continual.analysis.shapley_frpr import coalition_rows, exact_shapley, interactions
from drivers.continual.memory.episodes import pack_episode, preference_episode_id
from learners.action_regret import action_regret
from learners.action_set_preference import decision_masks
from learners.ranking_preference import reference_order, reference_scale
from learners.selected_frpr import choose_pairs, oriented_pairs, pair_inventory


def csv_output(path, rows, fields=()):
    import contextlib

    @contextlib.contextmanager
    def stream():
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            yield writer
    if rows is None:
        return stream()
    fields = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
    with stream() as writer:
        writer.writerows(rows)


def validate(cfg):
    frpr = {**RANK_DEFAULTS, **cfg.get("frpr", {})}
    if set(frpr) != set(RANK_DEFAULTS) or any(not nonnegative(v) for v in frpr.values()) or (
            not frpr["temperature"] or not frpr["q_scale_floor"]):
        raise ValueError("Invalid standard FRPR parameters")
    for key in ("max_exact_pairs", "rollout_repeats", "top_k"):
        if not positive_integer(cfg[key]):
            raise ValueError(f"{key} must be a positive integer")
    if (not nonnegative(cfg["projection_margin"]) or not nonnegative(cfg["softmax_temperature"])
            or not cfg["softmax_temperature"]):
        raise ValueError("Invalid projection margin or behavioral softmax temperature")
    quotas = cfg["per_stratum"]
    if type(quotas) is int:
        if quotas <= 0:
            raise ValueError("per_stratum must be positive")
    elif quotas != "all" and (not isinstance(quotas, dict) or not quotas or any(
            key not in {f"{time}/{loss}" for time in ("early", "middle", "late") for loss in ("low", "mid", "high")}
            or not positive_integer(value) for key, value in quotas.items())):
        raise ValueError("Invalid per_stratum mapping")
    return frpr


def online_source(policy, make_env, seeds, source=None):
    mac, batch = policy.episode()
    records = []

    def choose(frame, step):
        add_observation(batch, frame, step)
        # The real selector draws Torch RNG even in greedy test mode. Preserve replay RNG.
        with th.random.fork_rng(devices=[]):
            records.append(select(mac, batch, step))
        actions = records[-1]["actions"] if source is None else source["steps"][step]["actions"]
        add_action(batch, actions, step)
        return actions

    env = make_env()
    try:
        return play(env, choose=choose, source=source, **seeds), records
    finally:
        env.close()


def audit_records(offline, online):
    check, diff, errors = trace_comparator()
    check(len(offline), len(online), None, "length")
    for step, (left, right) in enumerate(zip(offline, online)):
        for key in left:
            check(left[key], right[key], step, key)
    return dict(passed=diff() is None, first_difference=diff(), max_abs_error=errors)


def packed_source(task, source, old, batch):
    length = len(source["steps"])
    for step, frame in enumerate(source["steps"]):
        info = frame["info"]
        batch.update({"reward": [[frame["reward"]]], "terminated": [[frame["terminated"]
                      and not info.get("episode_limit", False)]]}, ts=step)
    add_observation(batch, source["final"], length)
    episode = pack_episode(batch, length, won=True, seed=source["env_seed"], index=0)
    episode["data"]["old_q_values"] = th.tensor([record["q"][0] for record in old] + [old[-1]["q"][0]])
    # The FRPR unroll excludes the bootstrap Q and action.
    episode["data"]["old_q_values"][-1].zero_()
    episode["episode_id"] = preference_episode_id(task, episode)
    return episode


def source_contexts(task, episode, batch, ft, frpr):
    old_q = episode["data"]["old_q_values"][:-1]
    q = th.tensor([record["q"][0] for record in ft])
    legal, decisions = decision_masks(batch[:, :episode["length"] + 1])
    legal, decisions = legal[0], decisions[0] & (legal[0].sum(dim=-1) >= 2)
    gaps, pair_mask, _ = pair_inventory(old_q, legal, decisions, frpr)
    mean, _, scale = reference_scale(old_q, legal, frpr["q_scale_floor"])
    old_best = old_q.masked_fill(~legal, -th.inf).amax(dim=-1)
    ft_actions = q.masked_fill(~legal, -th.inf).argmax(dim=-1)
    _, regret_scale, _, _ = reference_order(old_q, legal, 0.1)
    raw, regret = action_regret(old_q, ft_actions, old_best, regret_scale.squeeze(-1))
    inventory = Counter(pair_mask.sum(dim=(-1, -2))[decisions].tolist())
    contexts = []
    for step, agent in pair_mask.flatten(-2).any(dim=-1).nonzero().tolist():
        phase = ("early", "middle", "late")[min(2, 3 * step // episode["length"])]
        loss = "low" if regret[step, agent] < 0.1 else "mid" if regret[step, agent] < 0.5 else "high"
        actions = legal[step, agent].nonzero().flatten().tolist()
        q_ft = q[step, agent].tolist()
        greedy = int(ft_actions[step, agent])
        pairs = oriented_pairs(gaps[step, agent], pair_mask[step, agent])
        row = dict(task=task, episode_id=episode["episode_id"], timestep=step, agent_id=agent,
                   battle_phase=phase, regret_stratum=loss, regret=float(raw[step, agent]),
                   regret_norm=float(regret[step, agent]), env_seed=episode["seed"],
                   M=len(pairs), coalition_count=1 << len(pairs), legal_actions=json.dumps(actions),
                   old_q=json.dumps(old_q[step, agent].tolist()), ft_q=json.dumps(q_ft),
                   old_mean=float(mean[step, agent]), old_scale=float(scale[step, agent]))
        site = dict(name=f"{phase}/{loss}", timestep=step, agent_id=agent, greedy_action=greedy,
                    greedy_q=q_ft[greedy], candidates=[dict(action=a, q=q_ft[a], q_gap=q_ft[greedy] - q_ft[a])
                                                      for a in actions])
        contexts.append((row, site, pairs, ((q[step, agent] - mean[step, agent]) / scale[step, agent]).tolist()))
    return contexts, inventory


def sampled(candidates, quotas, seed):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for item in candidates:
        row = item[0]
        groups[row["task"], f'{row["battle_phase"]}/{row["regret_stratum"]}'].append(item)
    for (_, key), group in sorted(groups.items()):
        count = len(group) if quotas == "all" else quotas if type(quotas) is int else quotas.get(key, 0)
        for index in sorted(rng.choice(len(group), min(count, len(group)), replace=False)):
            yield group[index]


@lru_cache(maxsize=None)
def t_critical(n):
    return float(t.ppf(0.975, n - 1))


def estimate(values):
    values = np.asarray(values)
    avg = float(values.mean())
    if len(values) < 2:
        return avg, None, None, None
    variance = float(values.var(ddof=1))
    half = t_critical(len(values)) * float(np.sqrt(variance / len(values)))
    return avg, variance, avg - half, avg + half


def metric_stats(values):
    return {f"{metric}_{label}": value
            for metric, estimates in zip(("return", "win"), (estimate(values[:, 0]), estimate(values[:, 1])))
            for label, value in zip(("mean", "variance", "ci_low", "ci_high"), estimates)}


KEYS = ("task", "episode_id", "timestep", "agent_id")
PAIR_FIELDS = (*KEYS, "participant", "preferred_action", "dispreferred_action")
COALITION_FIELDS = (*KEYS, "mask", "size", "projected_q_legal", "probabilities", "before_violations",
                    "after_violations", "max_violation_before", "projection_residual", "stationarity_residual",
                    "solver_iterations", *(f"{metric}_{field}" for metric in ("return", "win")
                                           for field in ("mean", "variance", "ci_low", "ci_high")))


def evaluate(item, spec, cfg, writer, audit):
    row, site, pairs, q = item
    key = {name: row[name] for name in KEYS}
    pair_rows = [dict(key, participant=i, preferred_action=p, dispreferred_action=d)
                 for i, (p, d) in enumerate(pairs)]
    if row["M"] > cfg["max_exact_pairs"]:
        row.update(status="over_exact_cap", enumerated=0, branch_rollouts=0)
        return pair_rows, [], [], []
    ft_policy, make_env, seeds, source, records, used_pids = spec
    actions = json.loads(row["legal_actions"])
    outcomes, returns = [], []
    start = perf_counter()
    for repeat in range(cfg["rollout_repeats"]):
        results = []
        for action in actions:
            started = perf_counter()
            run = branch(ft_policy, make_env, seeds, source, records, site, action, used_pids,
                         same_policy=False, trajectory_id=row["episode_id"])
            valid = (run["environment_replay_consistent"] is True and run["rnn_consistent"] is True
                     and run["first_difference"] is None and "error" not in run and run["won"] is not None)
            returns.append(dict(key, repeat=repeat, **seeds, **{
                field: json.dumps(value) if isinstance(value, (dict, list)) else value for field, value in run.items()
                if field not in key
            }, valid=valid, elapsed_seconds=perf_counter() - started))
            audit["branch_rollouts"] += 1
            if valid:
                audit["verified_branches"] += 1
                results.append((run["post_intervention_return"], int(run["won"])))
        if len(results) == len(actions):
            outcomes.append(results)
    row.update(branch_rollouts=len(returns), rollout_seconds=perf_counter() - start)
    if len(outcomes) != cfg["rollout_repeats"]:
        row.update(status="invalid_branch", enumerated=0)
        return pair_rows, returns, [], []

    start = perf_counter()
    outcomes = np.asarray(outcomes)
    values = []
    for mask, z, p, value, diagnostics in coalition_rows(
            q, actions, pairs, outcomes, cfg["projection_margin"], cfg["softmax_temperature"]):
        full_probability = np.zeros(len(q))
        full_probability[actions] = p
        writer.writerow(dict(key, mask=mask, size=bin(mask).count("1"), projected_q_legal=json.dumps(z.tolist()),
                             probabilities=json.dumps(full_probability.tolist()), **diagnostics, **metric_stats(value)))
        values.append(value)
        audit["coalitions"] += 1
        for name in ("projection_residual", "stationarity_residual"):
            audit[name] = max(audit[name], diagnostics[name])
        if not mask:
            row.update(base_q_legal=json.dumps([q[a] for a in actions]),
                       base_probability=json.dumps(full_probability.tolist()), base_max_probability=float(p.max()),
                       base_entropy=float(-np.dot(p[p > 0], np.log(p[p > 0])) / np.log(len(actions))))
    values = np.asarray(values)
    m = row["M"]
    phi, error = exact_shapley(values, m)
    audit["max_efficiency_error"] = max(audit["max_efficiency_error"], float(np.abs(error).max()))
    means = phi.mean(axis=1)
    ranks = [rankdata(-means[:, i], method="average") for i in range(2)]
    rand_settings = dict(selection_k=cfg["top_k"], selection_seed=cfg["sampling_seed"])
    scores = dict(zip(pairs, means[:, 0]))
    identity = tuple(key.values())
    top_sets = [set(choose_pairs(identity, dict(zip(pairs, run)), rand_settings, random_control=False))
                for run in phi[:, :, 0].T]
    selected = choose_pairs(identity, scores, rand_settings, random_control=False)
    random = choose_pairs(identity, scores, rand_settings, random_control=True)
    pair_index = {pair: i for i, pair in enumerate(pairs)}
    coalition = lambda selection: sum(1 << pair_index[pair] for pair in selection)
    jaccard = [len(a & b) / len(a | b) for i, a in enumerate(top_sets) for b in top_sets[i + 1:]]
    row.update(status="complete", enumerated=len(values), coalition_seconds=perf_counter() - start,
               efficiency_error=float(np.abs(error).max()),
               topk_repeat_jaccard=float(np.mean(jaccard)) if jaccard else None,
               rank_agreement=float(np.corrcoef(*ranks)[0, 1]) if all(np.ptp(r) > 0 for r in ranks) else None,
               **{f"{name}_{metric}": float(value[:, i].mean())
                  for name, value in (("empty_softmax", values[0]), ("vdn_ft_greedy", outcomes[:, actions.index(site["greedy_action"])]),
                                      ("frpr_full", values[-1]), ("shapley_selected", values[coalition(selected)]),
                                      ("random_selected", values[coalition(random)]))
                  for i, metric in enumerate(("return", "win"))})
    shapley = [dict(pair_rows[i], phi_return=float(means[i, 0]), phi_win=float(means[i, 1]),
                    rank_return=float(ranks[0][i]), rank_win=float(ranks[1][i]),
                    singleton_return=float((values[1 << i] - values[0])[:, 0].mean()),
                    top_k=pairs[i] in selected, random_top_k=pairs[i] in random,
                    repeat_top_k_frequency=sum(pairs[i] in subset for subset in top_sets) / len(top_sets),
                    **metric_stats(phi[i])) for i in range(m)]
    inter = [dict(key, participant_i=i, participant_j=j, **metric_stats(index))
             for i, j, index in interactions(values, m)]
    row["coalition_seconds"] = perf_counter() - start
    return pair_rows, returns, shapley, inter


def history_inventory(path, old_policy, frpr):
    from learners.action_set_preference import batch_episodes, unroll

    pool = th.load(path, map_location="cpu")
    histogram = Counter()
    recovered_q = 0
    mac, _ = old_policy.episode()
    for episode in pool["episodes"]:
        # Only these fields are used by batch_episodes and decision_masks.
        scheme = {key: {} for key in episode["data"]}
        saved_q = "old_q_values" in episode["data"]
        batch, _ = batch_episodes([episode], scheme, {}, "cpu", include_good=False, include_old_q=saved_q)
        legal, valid = decision_masks(batch)
        valid &= legal.sum(dim=-1) >= 2
        if saved_q:
            reference_q = batch["old_q_values"][:, :-1]
        else:
            recovered_q += 1
            reference_q = unroll(mac, batch, batch.max_seq_length - 1)
            actions = reference_q.masked_fill(~legal, -th.inf).argmax(dim=-1)
            executed = batch["actions_onehot"][:, :-1].argmax(dim=-1)
            if not th.equal(actions[valid], executed[valid]):
                raise ValueError(f"Recomputed reference Q fails greedy action audit in {path}")
        _, mask, _ = pair_inventory(reference_q, legal, valid, frpr)
        histogram.update(mask.sum(dim=(-1, -2))[valid].tolist())
    return histogram, dict(episodes=len(pool["episodes"]), recomputed_reference_episodes=recovered_q)


def aggregate(rows, pairs):
    by_context = defaultdict(list)
    for pair in pairs:
        by_context[tuple(pair[k] for k in KEYS)].append(pair)
    categories = defaultdict(list)
    for row in rows:
        categories[("all", "all", "all")].append(row)
        categories[(row["task"], row["battle_phase"], row["regret_stratum"])].append(row)
    for group, items in sorted(categories.items()):
        measured = [row for row in items if row["status"] == "complete"]
        these_pairs = [pair for row in measured for pair in by_context[tuple(row[k] for k in KEYS)]]
        fields = ("base_entropy", "base_max_probability", "rank_agreement", "topk_repeat_jaccard",
                  *(f"{name}_{metric}" for name in ("empty_softmax", "vdn_ft_greedy", "frpr_full",
                                                    "random_selected", "shapley_selected")
                    for metric in ("return", "win")))
        values = {key: [row[key] for row in measured if row[key] is not None] for key in fields}
        yield dict(kind="context_summary", task=group[0], battle_phase=group[1], regret_stratum=group[2],
                   contexts=len(items), completed=len(measured), branch_rollouts=sum(row["branch_rollouts"] for row in items),
                   enumerated=sum(row["enumerated"] for row in items),
                   rollout_seconds=sum(row.get("rollout_seconds", 0) for row in items),
                   coalition_seconds=sum(row.get("coalition_seconds", 0) for row in items),
                   **{key: float(np.mean(data)) if data else None for key, data in values.items()},
                   **{f"{m}_{stat}": val for m in ("phi_return", "phi_win")
                      for stat, val in distribution([p[m] for p in these_pairs]).items()})
        if group == ("all", "all", "all"):
            by_pair = defaultdict(list)
            for pair in these_pairs:
                by_pair[pair["task"], pair["preferred_action"], pair["dispreferred_action"]].append(pair)
            for (task, preferred, dispreferred), records in sorted(by_pair.items()):
                yield dict(kind="relation_summary", task=task, preferred_action=preferred,
                           dispreferred_action=dispreferred, contexts=len(records),
                           mean_return_rank=float(np.mean([p["rank_return"] for p in records])),
                           mean_win_rank=float(np.mean([p["rank_win"] for p in records])),
                           topk_frequency=float(np.mean([p["top_k"] for p in records])),
                           positive_fraction=float(np.mean([p["phi_return"] > 1e-12 for p in records])))


def distribution(data):
    if not len(data):
        return {}
    return dict(min=float(np.min(data)), p10=float(np.quantile(data, .1)), median=float(np.median(data)),
                p90=float(np.quantile(data, .9)), max=float(np.max(data)), mean=float(np.mean(data)),
                negative_fraction=float(np.mean(np.asarray(data) < -1e-12)))


def checked_policy(checkpoint):
    run = checkpoint.parents[2]
    info = read(run / "env_info.json")
    return FrozenPolicy(run, checkpoint, info), info


@th.inference_mode()
def run(config_path, out):
    cfg = dict(projection_margin=0., softmax_temperature=1., rollout_repeats=1)
    cfg.update(read(config_path))
    frpr = validate(cfg)
    out.mkdir(parents=True, exist_ok=False)
    th.set_num_threads(1)
    os.environ.setdefault("SC2PATH", str(ROOT / "runtime/SMACv2StarCraftII"))
    old_path, ft_path = (ROOT / cfg[k] for k in ("old_checkpoint", "ft_checkpoint"))
    if old_path.resolve() == ft_path.resolve() or sha(old_path) == sha(ft_path):
        raise ValueError("Old policy and VDN-FT need distinct checkpoints")
    old_protocol_path = old_path.parents[2] / "protocol.json"
    os.environ["MARL_PROTOCOL"] = str(old_protocol_path)
    old_policy, env_info = checked_policy(old_path)
    ft_policy, ft_info = checked_policy(ft_path)
    ft_protocol = read(ft_path.parents[2] / "protocol.json")
    if ft_protocol.get("method") != "baseline" or ft_protocol.get("replay_on_switch", "ft") != "ft":
        raise ValueError("The continuation policy requires a VDN-FT baseline checkpoint")
    if ft_protocol.get("algorithm") not in ("vdn", "vdn_1ff", "vdn_1ff_128") or env_info != ft_info:
        raise ValueError("Model environment dimensions or action layout differ")
    task = cfg["task"]
    inputs = [{ "path": str(path), "sha256": sha(path)} for path in (
        config_path, old_path, ft_path, old_protocol_path, ft_path.parents[2] / "protocol.json",
        old_path.parents[2] / "config.json", ft_path.parents[2] / "config.json",
        old_path.parents[2] / "env_info.json", ft_path.parents[2] / "env_info.json")]
    manifest = dict(config=cfg, frpr=frpr, inputs=inputs, argv=sys.argv,
                    code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    code_dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT)),
                    python=platform.python_version(), torch=th.__version__, scipy=scipy.__version__,
                    numpy=np.__version__, platform=platform.platform(), environment=env_info,
                    env_args=old_policy.args.env_args, sc2path=os.environ["SC2PATH"], traces=[])
    write(out / "manifest.json", manifest)
    audit = dict(passed=False, source_attempts=[], replays=[], branch_rollouts=0, verified_branches=0,
                 coalitions=0, max_efficiency_error=0., projection_residual=0., stationarity_residual=0.)
    rows, pairs, returns, shapley, inter, episodes = [], [], [], [], [], []
    eligible, sources, histograms = [], {}, defaultdict(Counter)
    used_pids = set()
    started = perf_counter()
    try:
        if cfg.get("inventory_pool"):
            path = ROOT / cfg["inventory_pool"]
            inputs.append(dict(path=str(path), sha256=sha(path)))
            histograms["historical_preference_pool"], audit["historical_inventory"] = history_inventory(
                path, old_policy, frpr)
        archive = cfg.get("archive")
        for i, seed_pair in enumerate(cfg["seeds"]):
            seeds = dict(zip(("env_seed", "sc2_seed"), seed_pair))
            make_env = lambda seed=seeds["env_seed"]: new_seeded_env(old_policy.args.env_args, task, seed)
            initial = perf_counter()
            if archive:
                incoming = ROOT / archive
                meta, prior = (read(incoming / name) for name in ("metadata.json", "summary.json"))
                original = incoming / "original.json.gz"
                if (meta["task"] != task or any(meta[k] != v for k, v in seeds.items())
                        or meta["checkpoint_sha256"] != sha(old_path) or prior["original_sha256"] != sha(original)
                        or prior["failed"] or prior["passed"] < 3):
                    raise ValueError("Source archive fails provenance checks")
                source = open_source(original)
            else:
                source, _ = online_source(old_policy, make_env, seeds)
                original = out / f"source_{i:03d}.json.gz"
                store_trace(original, source)
                source = open_source(original)
            clean = source["terminated"] and source["force_restarts"] == 0 and (
                source["create_game_seeds"] == [seeds["sc2_seed"]])
            audit["source_attempts"].append(dict(**seeds, clean=clean, won=source["battle_won"],
                                                 archived=bool(archive), seconds=perf_counter() - initial))
            manifest["traces"].append(dict(path=str(original), sha256=sha(original), seeds=seeds,
                                           settings=source["settings"], sc2_version=source["sc2_version"]))
            write(out / "manifest.json", manifest)
            if not clean:
                raise ValueError(f"Invalid source rollout {seeds}")
            if not source["battle_won"]:
                continue
            used_pids.add(source["sc2_pid"])
            old, batch = offline_records(old_policy, source, include_batch=True)
            ft = offline_records(ft_policy, source)
            if any(record["actions"] != frame["actions"] for record, frame in zip(old, source["steps"])):
                raise ValueError(f"Old checkpoint does not reproduce original greedy actions: {seeds}")
            replay, online = online_source(ft_policy, make_env, seeds, source)
            env_audit = differences(source, replay)
            rnn = audit_records(ft, online)
            fresh = replay["sc2_pid"] is not None and replay["sc2_pid"] not in used_pids
            used_pids.add(replay["sc2_pid"])
            audit["replays"].append(dict(**seeds, environment=env_audit, rnn=rnn, distinct_process=fresh))
            if not (env_audit["passed"] and rnn["passed"] and fresh):
                raise ValueError(f"Full replay or offline/online VDN-FT GRU audit failed for {seeds}")
            episode = packed_source(task, source, old, batch)
            context_rows, inventory = source_contexts(task, episode, batch, ft, frpr)
            histograms["trace_pool"].update(inventory)
            if episode["episode_id"] in sources:
                raise ValueError("Duplicate traces produce the same FRPR episode ID")
            sources[episode["episode_id"]] = ft_policy, make_env, seeds, source, ft, used_pids
            eligible.extend(context_rows)
            episodes.append(episode)

        with csv_output(out / "coalition_values.csv", None, COALITION_FIELDS) as writer:
            for item in sampled(eligible, cfg["per_stratum"], cfg["sampling_seed"]):
                row = item[0]
                rows.append(row)
                p, r, s, ii = evaluate(item, sources[row["episode_id"]], cfg, writer, audit)
                pairs.extend(p)
                returns.extend(r)
                shapley.extend(s)
                inter.extend(ii)
                print(row["task"], row["battle_phase"], row["regret_stratum"],
                      f"M={row['M']}", row["status"], flush=True)
        audit["passed"] = bool(shapley) and audit["branch_rollouts"] == audit["verified_branches"]
        if episodes:
            th.save(dict(task=task, phase=cfg["source_phase"], layout=env_info["action_layout"],
                         reference_sha256=sha(old_path), episodes=episodes), out / "trace_pool.th")
    finally:
        complete = sum(row.get("status") == "complete" for row in rows)
        audit.update(eligible_contexts=len(eligible), sampled_contexts=len(rows), completed_contexts=complete,
                     skipped_exact_cap=sum(row.get("status") == "over_exact_cap" for row in rows),
                     full_trace_pool_coverage=complete / len(eligible) if eligible else None,
                     source_collection_rollouts=sum(not row["archived"] for row in audit["source_attempts"]),
                     verified_replay_rollouts=len(audit["replays"]), total_seconds=perf_counter() - started)
        write(out / "audit.json", audit)
        csv_output(out / "contexts.csv", rows, KEYS)
        csv_output(out / "frpr_pairs.csv", pairs, PAIR_FIELDS)
        csv_output(out / "action_returns.csv", returns, (*KEYS, "action", "repeat"))
        csv_output(out / "shapley_values.csv", shapley, (*PAIR_FIELDS, "phi_return", "phi_win"))
        csv_output(out / "interactions.csv", inter, (*KEYS, "participant_i", "participant_j"))
        summary = list(aggregate(rows, shapley)) if all("status" in row for row in rows) else []
        for source, counts in sorted(histograms.items()):
            flat = [m for m, count in counts.items() for _ in range(count)]
            if flat:
                summary.append(dict(kind="inventory_summary", source=source, contexts=len(flat),
                                    mean_M=float(np.mean(flat)), median_M=float(np.median(flat)), max_M=max(flat)))
            for m, count in sorted(counts.items()):
                summary.append(dict(kind="inventory_histogram", source=source, M=m, contexts=count,
                                    coalition_count=1 << m))
        csv_output(out / "summary.csv", summary, ("kind",))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.exit(run(args.config, args.output))
