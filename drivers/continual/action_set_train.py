"""Standalone SMACv2 VDN sequence with optional decision preference replay."""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import json
import logging
import os
from pathlib import Path
import random
from time import perf_counter
import traceback
from types import SimpleNamespace

import train as base
from hetero_environment import ExactProtoss
from learners.action_set_preference import (
    ActionSetPreference, batch_episodes, decision_masks, preference_loss,
    reference_gaps, set_counts, set_statistics, unroll,
)
from components.task_replay import log_first_replay_batch, make_replay_buffer, start_replay_task
from learners.ranking_preference import RankPositionPreference, RankingPreference, aggregate_pair_filter_statistics
from learners.selected_frpr import SelectedFRPR
from learners.pair_ablation import PairAblationPreference, aggregate_ablation_statistics
from learners.pair_distance import PairDistancePreference
from action_set_protocol import (
    ASPR_MAX_METHODS, FORK_ROLES, PAIR_METHODS, PAIR_DISTANCE, PAIR_GAP_FAR_MID,
    PREFERENCE_RANK_METHODS, HISTORY_RANK_METHODS, SELECTED_METHODS, TOPK_VARIANTS,
    RANK_POSITION_METHODS, matching_pair_reference, matching_shared_protocol, validate, validate_fork,
)
from common.preference_timing import capture_update
from common.checkpoints import cpu_snapshot, parent_hash_fields, save_models
from common.metrics import Metrics, record_evaluation
from common.records import attach_legacy_metrics
from common.exploration import epsilon_on_switch
from common.runtime import episode_spec, restore_rng, rng_state, td_sample
from drivers.continual.memory import History, PreferenceHistory
from drivers.continual.memory.episodes import pack_episode, preference_episode_id
import numpy as np
import torch as th
import yaml


base.environments["action_set_protoss"] = ExactProtoss
ROOT = base.ROOT


def config(protocol, seed, out):
    cfg = {}
    for file in ("src/config/default.yaml", "vendor/smacv2/smacv2/examples/configs/sc2_gen_protoss.yaml",
                 f"src/config/algs/{protocol['algorithm']}.yaml"):
        base.merge(cfg, yaml.safe_load((ROOT / file).read_text()))
    use_cuda = th.cuda.is_available()
    cfg.update(env="action_set_protoss", seed=seed, use_cuda=use_cuda,
               device="cuda:0" if use_cuda else "cpu", local_results_path=str(out),
               algorithm=protocol["algorithm"], method=protocol["method"],
               t_max=sum(stage["steps"] for stage in protocol["stages"]),
               epsilon_on_switch=epsilon_on_switch(protocol),
               replay_on_switch=protocol.get("replay_on_switch", "ft"),
               gradient_monitor_interval=protocol.get("gradient_monitor_interval", 0),
               optimizer=protocol.get("optimizer", "rmsprop"),
               action_set=protocol["action_set"])
    if protocol.get("rank_position_diagnostics"):
        cfg["rank_position_diagnostics"] = True
    if protocol["method"] in PREFERENCE_RANK_METHODS:
        cfg[protocol["method"]] = protocol[protocol["method"]]
    cfg["env_args"].update(seed=seed, game_version=protocol.get("sc2_game_version", "4.10"),
                           rosters=protocol["tasks"])
    return cfg


def augmented_scheme(spec):
    scheme, groups, preprocess = spec
    return base.ReplayBuffer(scheme, groups, 1, 2, preprocess=preprocess).scheme


def layout(runner):
    info = runner.get_env_info()
    raw = runner.env.wrapped.env
    return dict(obs_shape=info["obs_shape"], state_shape=info["state_shape"],
                n_agents=info["n_agents"], n_actions=info["n_actions"],
                n_enemies=raw.n_enemies, map_type=raw.map_type,
                attack_offset=raw.n_actions_no_attack, unit_type_bits=raw.unit_type_bits)


@contextmanager
def isolated_rollouts():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with th.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def greedy_runner(args, spec, expected_layout, agent_state, task, seed, *, q_steps=None):
    ea = copy.deepcopy(args)
    ea.device, ea.use_cuda, ea.seed = "cpu", False, seed
    ea.env_args["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    th.random.default_generator.manual_seed(seed)
    runner = base.EpisodeRunner(ea, base.Logger(logging.getLogger("action_set_eval")))
    try:
        actual = layout(runner)
        if actual != expected_layout:
            raise ValueError(f"shared Agent layout mismatch on {task}: {actual} vs {expected_layout}")
        scheme, groups, preprocess = spec
        mac = base.controllers[ea.mac](augmented_scheme(spec), groups, ea)
        mac.agent.load_state_dict(agent_state)
        mac.agent.eval().requires_grad_(False)
        runner.env.set_task(task)
        runner.setup(scheme, groups, preprocess, mac)
        # The Agent returns raw Q before the action selector masks illegal actions.
        def capture_q(_agent, _inputs, output):
            q, _ = output
            q_steps.append(q.detach().to(dtype=th.float32))

        hook = mac.agent.register_forward_hook(capture_q) if q_steps is not None else None
        try:
            with th.no_grad():
                yield runner
        finally:
            if hook is not None:
                hook.remove()
    finally:
        runner.close_env()


def reference_record(runner, batch, episode, *, q_steps=None, label_action_sets=False, store_old_q=False):
    if q_steps is not None:
        old_q = th.stack(q_steps)
        if old_q.shape != episode["data"]["avail_actions"].shape:
            raise ValueError("Collected Q values do not align with the stored episode")
        episode["data"]["old_q_values"] = old_q
        if not label_action_sets:
            return episode
        q = old_q[None, :-1]
    batch = batch[:, :runner.t + 1]
    if q_steps is None and store_old_q:
        old_q = unroll(runner.mac, batch, runner.t + 1)
        episode["data"]["old_q_values"] = old_q[0].to(dtype=th.float32)
        q = old_q[:, :-1]
    elif q_steps is None:
        q = unroll(runner.mac, batch, runner.t)
    _, decision = decision_masks(batch)
    return episode, reference_gaps(batch, q)[0], decision[0]


def label_records(records, delta, candidates=()):
    counts = {value: th.zeros(4, dtype=th.long) for value in candidates}
    episodes = []
    for episode, gaps, decision in records:
        legal = episode["data"]["avail_actions"][:-1].bool()
        eligible = legal & decision.unsqueeze(-1)
        good = eligible & (gaps <= delta)
        if (decision & ~good.any(dim=-1)).any():
            raise ValueError(f"empty reference good set, seed {episode['seed']}, episode {episode['index']}")
        final = th.zeros_like(good[:1])
        episodes.append(dict(episode, good_mask=th.cat((good, final), dim=0)))
        for value in candidates:
            counts[value] += set_counts(legal, decision, eligible & (gaps <= value))
    return episodes, [dict(delta=value, **set_statistics(*count.tolist()))
                      for value, count in counts.items()]


def calibrate_delta(records, settings, task):
    second_gaps = []
    for _, gaps, decision in records:
        gap = gaps.topk(2, largest=False, dim=-1).values[..., 1]
        second_gaps.append(gap[decision & gap.isfinite()])
    values = th.cat(second_gaps) if second_gaps else th.empty(0)
    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantiles = th.quantile(values, th.tensor(levels)).tolist() if values.numel() else [None] * len(levels)
    explicit = settings["delta_by_task"].get(task)
    requested = settings["second_gap_quantile"]
    if explicit is not None:
        delta, source = explicit, "configured"
    elif values.numel():
        delta, source = th.quantile(values, requested).item(), "second_gap_quantile"
    else:
        delta, source = 0.0, "no_two_action_decisions"
    return delta, dict(source=source, requested_quantile=requested, decision_states=values.numel(),
                       quantiles=dict(zip(map(str, levels), quantiles)))


def label_successful_memory(records, settings, task):
    delta, calibration = calibrate_delta(records, settings, task)
    candidates = list(dict.fromkeys([delta, *settings["delta_candidates_by_task"].get(task, [])]))
    episodes, statistics = label_records(records, delta, candidates)
    return episodes, dict(delta=delta), dict(calibration=calibration, candidates=statistics)


def restore_aspr_shared_memory(history, parent, phases, settings, scheme, groups, out):
    history.restore_success(parent, phases, require_old_q=True)
    for phase in phases:
        pool = history[phase]
        records = []
        for episode in pool["episodes"]:
            batch, _ = batch_episodes([episode], scheme, groups, "cpu", include_good=False, include_old_q=True)
            q = batch["old_q_values"]
            if q.shape != batch["avail_actions"].shape:
                raise ValueError("historical Q values do not align with avail_actions")
            _, decision = decision_masks(batch)
            records.append((episode, reference_gaps(batch, q[:, :-1])[0], decision[0]))
        episodes, labels, diagnostics = label_successful_memory(records, settings, pool["task"])
        history[phase] = dict(pool, episodes=episodes, **labels)
        path = history.phase_path(out, phase)
        path.mkdir(parents=True, exist_ok=False)
        history.save(path / "success.th", history[phase])
        base.write(path / "summary.json", dict(task=pool["task"], phase=phase,
                   source_pool=str(history.phase_path(parent, phase) / "success.th"),
                   reference_checkpoint=pool["reference_checkpoint"], stored_wins=len(episodes),
                   **labels, **diagnostics))


def collect_memory(args, spec, expected_layout, reference, checkpoint, task, phase, settings, out):
    start = perf_counter()
    successful, failed, blocks = [], [], []
    offsets = settings["seed_offsets"]
    target = settings["success_episodes"]
    label_action_sets = args.method in (*ASPR_MAX_METHODS, "aspr_preference")
    q_steps = [] if args.method in HISTORY_RANK_METHODS else None
    with isolated_rollouts():
        for block, offset in enumerate(offsets):
            seed = (args.seed + offset) % 2**31
            quota = (target * (block + 1) + len(offsets) - 1) // len(offsets)
            wins = attempts = 0
            with greedy_runner(args, spec, expected_layout,
                               reference, task, seed, q_steps=q_steps) as runner:
                # Give each seed block at least one rollout, even for very small targets.
                for index in range(settings["attempts_per_seed"]):
                    if index and len(successful) >= quota:
                        break
                    if q_steps is not None:
                        q_steps.clear()
                    batch = runner.run(test_mode=True)
                    won = bool(runner.last_episode_info.get("battle_won", False))
                    episode = pack_episode(batch, runner.t, seed=seed, index=index, won=won)
                    attempts += 1
                    if not won:
                        failed.append(episode)
                        continue
                    wins += 1
                    if len(successful) >= target:
                        continue
                    successful.append(reference_record(runner, batch, episode, q_steps=q_steps,
                                                       label_action_sets=label_action_sets))
            blocks.append(dict(seed=seed, attempts=attempts, wins=wins))
    if args.method in SELECTED_METHODS:
        for episode in successful:
            episode["episode_id"] = preference_episode_id(task, episode)
    labels, diagnostics = {}, {}
    if label_action_sets:
        successful, labels, diagnostics = label_successful_memory(successful, settings, task)
    path = PreferenceHistory.phase_path(out, phase)
    path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = str(checkpoint.relative_to(out) / "agent.th")
    pool = dict(task=task, phase=phase, layout=expected_layout,
                reference_checkpoint=checkpoint_path, episodes=successful, **labels)
    History.save(path / "success.th", pool)
    History.save(path / "failure.th", dict(task=task, phase=phase, episodes=failed))
    base.write(path / "summary.json", dict(
        task=task, phase=phase, reference_checkpoint=checkpoint_path,
        requested_wins=target, stored_wins=len(successful), shortfall=target - len(successful),
        failed=len(failed), blocks=blocks, collection_seconds=perf_counter() - start,
        **labels, **diagnostics))
    return pool


def evaluation_settings(protocol, task, focus):
    settings = protocol["evaluation"]
    order = list(dict.fromkeys(stage["task"] for stage in protocol["stages"]))
    offset = order.index(task) * settings["task_seed_stride"]
    seeds, count = settings["seeds"], settings["episodes_per_seed"]
    if focus and settings.get("focus_seeds") is not None:
        seeds, count = settings["focus_seeds"], settings["focus_episodes_per_seed"]
    return dict(seeds=[seed + offset for seed in seeds], episodes_per_seed=count)


def evaluate_task(args, spec, expected_layout, reference, task, evaluation, episodes=None,
                  probe_records=None, probe_limit=0, log_probe_error=None):
    results = []
    with isolated_rollouts():
        remaining = episodes
        for i, seed in enumerate(evaluation["seeds"]):
            count = (evaluation["episodes_per_seed"] if remaining is None else
                     (remaining + len(evaluation["seeds"]) - i - 1) // (len(evaluation["seeds"]) - i))
            if count == 0:
                break
            with greedy_runner(args, spec, expected_layout, reference, task, seed) as runner:
                for index in range(count):
                    batch = runner.run(test_mode=True)
                    won = bool(runner.last_episode_info.get("battle_won", False))
                    results.append(dict(seed=seed, index=index, length=runner.t, won=won))
                    if probe_records is not None and index < probe_limit:
                        try:
                            episode = pack_episode(batch, runner.t, seed=seed, index=index, won=won)
                            probe_records.append(reference_record(runner, batch, episode,
                                                                   store_old_q=args.method == "aspr_preference"))
                        except Exception:
                            log_probe_error(traceback.format_exc())
            if remaining is not None:
                remaining -= count
    wins = sum(row["won"] for row in results)
    return dict(task=task, wins=wins, episodes=len(results), win_rate=wins / len(results), outcomes=results)


def train_sequence(protocol, seed, out, role=None, parent=None, capture_timing=False):
    validate_fork(protocol, role, parent)
    if capture_timing and (role != "shared-branch" or protocol["method"] != "topk_preference"
                           or protocol["topk_preference"]["k"] != 1):
        raise ValueError("timing capture requires a shared TKPR-K1 branch")
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    args = SimpleNamespace(**config(protocol, seed, out))
    if role:
        args.preference_role = role
        args.parent = str(parent) if parent else None
        args.preference_weight = (protocol[protocol["method"]]["weight"]
                                  if role not in ("release", "pair-reference")
                                  and protocol["method"] in PREFERENCE_RANK_METHODS else 0.0)
    if capture_timing:
        args.capture_timing = True
    logger = base.Logger(logging.getLogger(out.name))
    attach_legacy_metrics(logger, out)
    metric_method = "aspr_max_preference" if args.method in ASPR_MAX_METHODS else args.method
    metrics = Metrics(out, args.algorithm, metric_method, seed, **({"role": role} if role else {}))
    runner = base.EpisodeRunner(args, logger)
    phase = episode = updates = 0

    def status(state, **extra):
        base.write(out / "status.json", dict(status=state, phase=phase, t_env=runner.t_env,
                                            episode=episode, updates=updates, **extra))

    try:
        info = runner.get_env_info()
        expected_layout = layout(runner)
        if (expected_layout["n_agents"], expected_layout["n_enemies"], expected_layout["map_type"],
                expected_layout["attack_offset"], expected_layout["n_actions"]) != (5, 5, "protoss_gen", 6, 11):
            raise ValueError(f"unsupported SMACv2 Protoss action/agent layout: {expected_layout}")
        args.n_agents, args.n_actions, args.state_shape = info["n_agents"], info["n_actions"], info["state_shape"]
        if args.method == "topk_preference" and protocol[args.method]["k"] >= args.n_actions:
            raise ValueError(f"topk_preference.k must be at most n_actions - 1 ({args.n_actions - 1})")
        base.write(out / "config.json", vars(args))
        base.write(out / "env_info.json", dict(info, action_layout=expected_layout))
        scheme, groups, preprocess = spec = episode_spec(info)
        # Obtain the same derived action-onehot spec used by both the runner and MAC.
        augmented = augmented_scheme(spec)
        mac = base.controllers[args.mac](augmented, groups, args)
        runner.setup(*spec, mac)
        learner = base.learners[args.learner](mac, augmented, logger, args)
        for network in (mac.agent, learner.target_mac.agent):
            for module in network.modules():
                module._forward_hooks.clear()  # No activation instrumentation in this driver.
        if args.use_cuda:
            learner.cuda()
        settings = protocol["action_set"]
        method = protocol["method"]
        aspr = method == "aspr_preference"
        selected_frpr = method in SELECTED_METHODS
        pair_ablation = method in PAIR_METHODS
        shared_history = PreferenceHistory()
        anchor_rank = RANK_POSITION_METHODS.get(method)
        if method == "topk_preference" and protocol.get("rank_position_diagnostics"):
            anchor_rank = 1

        def record_preference(row):
            row["learner_update"] = updates + 1
            if pair_ablation:
                row["pair_ablation_stats"] = aggregate_ablation_statistics(row["historical_tasks"])
            if method in ("topk_preference", *TOPK_VARIANTS):
                counts = Counter()
                for historical in row["historical_tasks"]:
                    counts.update(historical["effective_k_histogram"])
                row["effective_k_histogram"] = dict(counts)
            elif method in ("full_rank_preference", "full_rank_preference_unfiltered",
                            "full_rank_preference_raw", "full_rank_preference_raw_loss", "aspr_preference"):
                row["pair_filter_stats"] = aggregate_pair_filter_statistics(row["historical_tasks"])
            if aspr:
                history = row["historical_tasks"]
                totals = row["pair_filter_stats"]
                violations = sum(item["pair_violations"] for item in history)
                pairs = totals["valid_pairs"]
                totals.update(pair_violations=violations, pair_violation_rate=violations / pairs if pairs else None)
                for name in ("average_good_set_size", "singleton_ratio", "all_action_ratio"):
                    weighted = sum(item[name] * item["decision_states"] for item in history if item["decision_states"])
                    totals[name] = weighted / row["decision_states"] if row["decision_states"] else None
            base.append(out / f"{update_log}_updates.jsonl", row)
            if method == "action_set_preference":
                base.append(out / "action_set_updates.jsonl", row)
            metrics.preference(row, event="preference_update")

        def record_gradients(values):
            metrics.gradients(phase, task, runner.t_env, updates + 1, values)
            for name, value in values.items():
                if value is not None:
                    logger.log_stat(name, value, runner.t_env)

        preference = None
        if method in ASPR_MAX_METHODS and settings["weight"]:
            update_log = "aspr_max_preference"
            preference = ActionSetPreference(settings, np.random.default_rng(seed), augmented, groups,
                                             record_preference)
        elif method in PREFERENCE_RANK_METHODS and protocol[method]["weight"] and role != "release":
            update_log = method
            ranking = dict(protocol[method], batch_episodes=settings["batch_episodes"])
            if pair_ablation:
                preference_type = (PairDistancePreference if method in (*PAIR_DISTANCE, *PAIR_GAP_FAR_MID)
                                   else PairAblationPreference)
                preference = preference_type(PAIR_METHODS[method], ranking, np.random.default_rng(seed),
                                             augmented, groups, record_preference)
            elif anchor_rank is not None:
                preference = RankPositionPreference(anchor_rank, ranking, np.random.default_rng(seed),
                                                    augmented, groups, record_preference)
            elif selected_frpr:
                preference = SelectedFRPR(method, ranking, np.random.default_rng(seed), augmented, groups,
                                          record_preference)
            else:
                preference = RankingPreference(method, ranking, np.random.default_rng(seed), augmented, groups,
                                                record_preference)

        stages = protocol["stages"]
        evaluation = protocol["evaluation"]
        initial, before, latest, results = {}, {}, {}, []
        probes = History()
        nominal_end = 0

        def agent_snapshot():
            return cpu_snapshot(mac.agent)

        def probe_error(task, error):
            base.append(out / "action_set_probe_errors.jsonl", dict(phase=phase, task=task, error=error))

        def evaluate(reference, task, label, probe_records=None):
            focus = label in ("stage_end", "initial_reference", "next_unseen")
            row = evaluate_task(args, spec, expected_layout, reference, task,
                                evaluation_settings(protocol, task, focus), probe_records=probe_records,
                                probe_limit=settings["probe_episodes_per_seed"],
                                log_probe_error=lambda error: probe_error(task, error))
            record_evaluation(out, metrics, dict(phase=phase, label=label, t_env=runner.t_env, **row))
            return row["win_rate"]

        def save_fork_state():
            th.save(dict(
                fork_role=role, protocol=protocol, seed=seed, t_env=runner.t_env, episode=episode, updates=updates,
                target_agent=learner.target_mac.agent.state_dict(),
                last_target_update_episode=learner.last_target_update_episode,
                learner_log_t=learner.log_stats_t, runner_log_t=runner.log_train_stats_t,
                train_returns=runner.train_returns, train_stats=runner.train_stats,
                epsilon=mac.action_selector.epsilon, exploration_start=mac.action_selector.exploration_start,
                environment_distributions=runner.env.distribution_state(), rng=rng_state(),
                preference_rng=preference.rng.bit_generator.state, model_hash=base.parameter_hash(learner),
            ), out / "return_state.th")

        def prepare_shared_pair_pool(source_phase):
            pool = shared_history[source_phase]
            prior_budget = sum(item["pair_budget"] for item in preference.history.values())
            episodes, audit = preference.prepare_pool(pool, prior_budget=prior_budget)
            preference.history[source_phase] = dict(pool, episodes=episodes, pair_budget=audit["valid_pairs"])
            path = out / "pair_ablation_memory" / f"phase{source_phase:04d}"
            History.save(path / "selection.th", [ep["pair_mask"] for ep in episodes])
            base.write(path / "summary.json", dict(
                selection=PAIR_METHODS[method], pair_seed=preference.settings["pair_seed"],
                source_pool=str(PreferenceHistory.phase_path(parent, source_phase) / "success.th"),
                **audit))

        def prepare_selected_history(current_task):
            import hashlib

            latest_pools = {pool["task"]: (source_phase, pool)
                            for source_phase, pool in preference.history.items()
                            if pool["task"] != current_task and pool["episodes"]}
            audits = []
            for source_phase, pool in latest_pools.values():
                if "shapley_coverage" in pool:
                    audits.append(pool["shapley_coverage"])
                    continue
                trace_path = preference.settings.get("trace_pools", {}).get(str(source_phase))
                if trace_path:
                    trace_pool = History.load(trace_path)
                    checkpoint = out / pool["reference_checkpoint"]
                    if (trace_pool["task"] != pool["task"] or trace_pool["phase"] != source_phase
                            or trace_pool["layout"] != expected_layout or not trace_pool["episodes"]
                            or trace_pool["reference_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
                        raise ValueError(f"Trace pool does not match task, layout and old checkpoint: {trace_path}")
                    pool = dict(pool, episodes=trace_pool["episodes"])
                prepared, audit = preference.prepare_pool(pool)
                audits.append(audit)
                if audit["covered"] == audit["required"]:
                    preference.history[source_phase] = dict(prepared, shapley_coverage=audit)
            required = sum(row["required"] for row in audits)
            covered = sum(row["covered"] for row in audits)
            base.append(out / "shapley_coverage.jsonl", dict(
                phase=phase, task=current_task, required=required, covered=covered,
                coverage=covered / required if required else None, pools=audits))
            if covered != required:
                raise ValueError(f"Incomplete {method} coverage before phase {phase}: {covered}/{required}; "
                                 "see shapley_coverage.jsonl")

        first_phase = 0
        if parent:
            state = th.load(parent / "return_state.th", map_location="cpu")
            prefix_role = "shared-prefix" if role == "shared-branch" else "prefix"
            # Older direct launches used per-task exploration when the field was absent.
            parent_protocol = {"epsilon_on_switch": "per_task", **state["protocol"]}
            if pair_ablation:
                prefix_role = "pair-reference"
                same_protocol = matching_pair_reference(parent_protocol, protocol)
            else:
                same_protocol = (matching_shared_protocol(parent_protocol, protocol)
                                 if role == "shared-branch" else parent_protocol == protocol)
            if state.get("fork_role", "prefix") != prefix_role or not same_protocol or state["seed"] != seed:
                raise ValueError("preference branch must match its prefix role, protocol and seed")
            report = json.loads((parent / "retention.json").read_text())
            if pair_ablation:
                report["stages"] = report["stages"][:1]
                # Z must be treated as unseen when restoring at the S boundary.
                next_task = stages[1]["task"]
                before_first_training = {next_task: report["before_first_training"][next_task]}
            else:
                before_first_training = report["before_first_training"]
            if role == "shared-branch" and len(report["stages"]) != 1:
                raise ValueError("a shared prefix must contain exactly its first stage")
            initial, before = report["initial_reference"], before_first_training
            for entry in report["stages"]:
                latest[entry["task"]] = entry["win_rates"][entry["task"]]
                results.append(dict(entry, checkpoint=str(parent / entry["checkpoint"])))
            first_phase = len(results)
            phase = first_phase
            nominal_end = sum(stage["steps"] for stage in stages[:first_phase])
            learner.load_models(results[-1]["checkpoint"])
            learner.target_mac.agent.load_state_dict(state["target_agent"])
            learner.last_target_update_episode = state["last_target_update_episode"]
            learner.log_stats_t = state["learner_log_t"]
            runner.t_env, episode, updates = state["t_env"], state["episode"], state["updates"]
            runner.train_returns = state["train_returns"]
            runner.train_stats = state["train_stats"]
            runner.log_train_stats_t = state["runner_log_t"]
            mac.action_selector.exploration_start = state["exploration_start"]
            mac.action_selector.epsilon = state["epsilon"]
            runner.env.restore_distribution(state["environment_distributions"])
            if preference is not None:
                preference.rng.bit_generator.state = state["preference_rng"]
                if pair_ablation:
                    shared_history.restore_success(parent, range(len(stages) - 1), require_old_q=True)
                    prepare_shared_pair_pool(0)
                elif aspr and role == "shared-branch":
                    restore_aspr_shared_memory(preference.history, parent, range(first_phase),
                                               settings, augmented, groups, out)
                else:
                    preference.history.restore_success(parent, range(first_phase),
                                                       require_old_q=method in HISTORY_RANK_METHODS, require_good=aspr)
            restore_rng(state["rng"])
            parent_hash = dict(parent_model_hash=state["model_hash"])
            if getattr(mac.agent, "loaded_legacy_fc1", False):
                # VDN mixers are parameter-free. Only the two Agent states enter the hash.
                online = th.load(Path(results[-1]["checkpoint"]) / "agent.th", map_location="cpu")
                parent_hash = parent_hash_fields(state, (online, state["target_agent"]))
            base.write(out / "branch_restore.json", dict(
                parent=str(parent), checkpoint=results[-1]["checkpoint"], role=role,
                preference_weight=args.preference_weight,
                t_env=runner.t_env, episode=episode, updates=updates,
                last_target_update_episode=learner.last_target_update_episode,
                model_hash=base.parameter_hash(learner), **parent_hash,
                engine="new SC2 process, distribution RNG restored"))
        status("running")
        if not parent:
            reference = agent_snapshot()
            for task in dict.fromkeys(stage["task"] for stage in stages if stage["task"] != stages[0]["task"]):
                initial[task] = evaluate(reference, task, "initial_reference")

        er_buffer = None
        if args.replay_on_switch == "er":
            er_buffer = make_replay_buffer(args, [stage["task"] for stage in stages],
                                           scheme, groups, info["episode_limit"] + 1, preprocess)

        stop_phase = {"prefix": len(stages) - 1, "shared-prefix": 1,
                      "pair-reference": len(stages) - 1}.get(role, len(stages))
        training_preference = None if role in ("shared-prefix", "pair-reference") else preference
        monitor_interval = args.gradient_monitor_interval
        for phase in range(first_phase, stop_phase):
            stage = stages[phase]
            task = stage["task"]
            if selected_frpr and training_preference is not None:
                prepare_selected_history(task)
            begin = runner.t_env
            nominal_end += stage["steps"]
            end = nominal_end if protocol.get("phase_boundary") == "cumulative" else begin + stage["steps"]
            buffer = er_buffer
            if buffer is None:
                buffer = base.ReplayBuffer(scheme, groups, args.buffer_size, info["episode_limit"] + 1,
                                           preprocess=preprocess, device="cpu")
            replay_state = start_replay_task(buffer, task)
            runner.env.set_task(task)
            base.log_task_epsilon(runner, out, phase, task, "start")
            if preference is not None:
                preference.phase, preference.task = phase, task
            base.append(out / "transitions.jsonl", dict(phase=phase, task=task, start=begin, target=end, **replay_state))
            status("running")
            evaluate(agent_snapshot(), task, "phase_start")
            interval = evaluation["interval"]
            next_eval = begin + interval
            timing_points = ([(part, begin + (end - begin) * part // 4) for part in (1, 2, 3)]
                             if capture_timing else [])
            while runner.t_env < end:
                with th.no_grad():
                    batch = runner.run()
                buffer.insert_episode_batch(batch)
                if buffer.can_sample(args.batch_size):
                    sample = td_sample(buffer, args.batch_size, args.device)
                    diagnostics = {}
                    if monitor_interval and (updates + 1) % monitor_interval == 0:
                        diagnostics["gradient_monitor"] = record_gradients
                    while timing_points and runner.t_env >= timing_points[0][1]:
                        part, _ = timing_points.pop(0)
                        try:
                            capture_update(out, learner, sample, preference, phase, task,
                                           part, runner.t_env, episode, updates + 1)
                        except Exception:
                            logging.getLogger(out.name).exception("Could not capture a timing sample")
                    learner.train(sample, runner.t_env, episode, preference=training_preference, **diagnostics)
                    updates += 1
                    if replay_state:
                        log_first_replay_batch(buffer, base.append, out, phase, task, runner.t_env)
                        replay_state = {}
                episode += 1
                if episode % 25 == 0:
                    status("running")
                if interval and next_eval <= runner.t_env < end:
                    evaluate(agent_snapshot(), task, "periodic")
                    next_eval = begin + ((runner.t_env - begin) // interval + 1) * interval
            del buffer
            base.log_task_epsilon(runner, out, phase, task, "end")
            checkpoint = out / "models" / f"phase{phase:04d}_{runner.t_env}"
            save_models(learner, checkpoint)
            reference = agent_snapshot()
            probe_records = ([] if method in (*ASPR_MAX_METHODS, "aspr_preference") and preference is not None
                             and settings["probe_episodes_per_seed"] else None)
            row = {seen: evaluate(reference, seen, "stage_end", probe_records if seen == task else None)
                   for seen in dict.fromkeys(stage["task"] for stage in stages[:phase + 1])}
            latest[task] = row[task]
            bwt = [value - latest[seen] for seen, value in row.items() if seen != task]
            fwt = [before[seen] - initial[seen] for seen in row if seen in before]
            results.append(dict(phase=phase, task=task, start=begin, t_env=runner.t_env,
                                checkpoint=str(checkpoint.relative_to(out)), win_rates=row,
                                acc_percent=100 * np.mean(list(row.values())),
                                bwt_pp=100 * np.mean(bwt) if bwt else None,
                                fwt_pp=100 * np.mean(fwt) if fwt else None))
            metrics.retention(phase, task, runner.t_env, results[-1])
            if phase + 1 < len(stages):
                next_task = stages[phase + 1]["task"]
                if next_task not in row:
                    before[next_task] = evaluate(reference, next_task, "next_unseen")
            base.write(out / "retention.json", dict(stages=results, initial_reference=initial,
                                                   before_first_training=before))
            if pair_ablation:
                if phase + 1 < stop_phase:
                    prepare_shared_pair_pool(phase)
            elif preference is not None:
                status("collecting_memory")
                pool = collect_memory(args, spec, expected_layout, reference, checkpoint,
                                      task, phase, settings, out)
                preference.history[phase] = pool
                if probe_records:
                    try:
                        labeled, _ = label_records(probe_records, pool["delta"])
                        probe = dict(task=task, source_phase=phase, delta=pool["delta"], episodes=labeled)
                        probes.save(PreferenceHistory.phase_path(out, phase) / "probe.th", probe)
                        probes[phase] = probe
                    except Exception:
                        probe_error(task, traceback.format_exc())
                for probe in probes.values():
                    try:
                        with th.no_grad():
                            if aspr:
                                _, _, stats = preference.task_loss(mac, probe, probe["episodes"])
                            else:
                                probe_batch, good = batch_episodes(probe["episodes"], augmented, groups, args.device)
                                _, _, stats = preference_loss(mac, probe_batch, good, settings["margin"])
                        row_metrics = dict(
                            phase=phase, t_env=runner.t_env, task=probe["task"],
                            source_phase=probe["source_phase"], delta=probe["delta"],
                            episodes=len(probe["episodes"]), win_rate=row[probe["task"]],
                            **preference.statistics(stats.cpu().tolist()))
                        base.append(out / f"{update_log}_probes.jsonl", row_metrics)
                        if method == "action_set_preference":
                            base.append(out / "action_set_probes.jsonl", row_metrics)
                        metrics.preference(row_metrics, event="fixed_probe")
                    except Exception:
                        probe_error(probe["task"], traceback.format_exc())
            if role == "pair-reference" and phase == 0:
                save_fork_state()
        if role in ("prefix", "shared-prefix"):
            save_fork_state()
        status("completed")
    except BaseException:
        status("failed", error=traceback.format_exc())
        raise
    finally:
        runner.close_env()


def evaluate_final(run, output, episodes, protocol):
    status = json.loads((run / "status.json").read_text())
    if status["status"] != "completed":
        raise ValueError("final evaluation requires a completed action-set sequence")
    args = SimpleNamespace(**json.loads((run / "config.json").read_text()))
    info = json.loads((run / "env_info.json").read_text())
    spec = episode_spec(info)
    final = json.loads((run / "retention.json").read_text())["stages"][-1]
    reference = th.load(run / final["checkpoint"] / "agent.th", map_location="cpu")
    rows = [evaluate_task(args, spec, info["action_layout"],
                          reference, task, evaluation_settings(protocol, task, focus=True), episodes)
            for task in final["win_rates"]]
    base.write(output / "results.json", dict(checkpoint=final["checkpoint"], t_env=final["t_env"], tasks=rows))
    metrics = Metrics(output, protocol["algorithm"], protocol["method"], args.seed, run_id=run.name)
    for row in rows:
        metrics.evaluation(dict(row, phase=final["phase"], t_env=final["t_env"], label="independent_final"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", type=int)
    mode.add_argument("--evaluate-run", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--preference-role", choices=FORK_ROLES)
    parser.add_argument("--capture-timing", action="store_true")
    parser.add_argument("--parent", type=lambda path: Path(path).resolve())
    args = parser.parse_args()
    out = Path(os.environ["MARL_RUN_DIR"])
    protocol = base.PROTOCOL
    protocol = validate(protocol, protocol["algorithm"], protocol["method"])
    th.set_num_threads(1)
    logging.basicConfig(level=logging.INFO)
    if args.evaluate_run:
        evaluate_final(args.evaluate_run, out, args.episodes, protocol)
    else:
        train_sequence(protocol, args.seed, out, role=args.preference_role, parent=args.parent,
                       capture_timing=args.capture_timing)


if __name__ == "__main__":
    main()
