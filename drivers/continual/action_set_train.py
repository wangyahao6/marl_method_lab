"""Standalone SMACv2 VDN sequence with optional decision preference replay."""
import argparse
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
    ActionSetPreference, batch_episodes, decision_masks, pack_episode, preference_loss,
    preference_statistics, reference_gaps, set_counts, set_statistics, unroll,
)
from learners.ranking_preference import RankingPreference
from action_set_protocol import RANK_METHODS, validate
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
               epsilon_on_switch=protocol.get("epsilon_on_switch", "per_task"),
               action_set=protocol["action_set"])
    if protocol["method"] in RANK_METHODS:
        cfg[protocol["method"]] = protocol[protocol["method"]]
    cfg["env_args"].update(seed=seed, game_version=protocol.get("sc2_game_version", "4.10"),
                           rosters=protocol["tasks"])
    return cfg


def episode_spec(info):
    scheme = dict(state=dict(vshape=info["state_shape"]),
                  obs=dict(vshape=info["obs_shape"], group="agents"),
                  actions=dict(vshape=(1,), group="agents", dtype=th.long),
                  avail_actions=dict(vshape=(info["n_actions"],), group="agents", dtype=th.int),
                  reward=dict(vshape=(1,)), terminated=dict(vshape=(1,), dtype=th.uint8))
    groups = dict(agents=info["n_agents"])
    preprocess = dict(actions=("actions_onehot", [base.OneHot(out_dim=info["n_actions"])]))
    return scheme, groups, preprocess


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
def greedy_runner(args, spec, expected_layout, agent_state, task, seed):
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
        with th.no_grad():
            yield runner
    finally:
        runner.close_env()


def reference_record(runner, batch, episode):
    batch = batch[:, :runner.t + 1]
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


def collect_memory(args, spec, expected_layout, reference, checkpoint, task, phase, settings, out):
    start = perf_counter()
    successful, failed, blocks = [], [], []
    offsets = settings["seed_offsets"]
    target = settings["success_episodes"]
    label_action_sets = args.method == "action_set_preference"
    with isolated_rollouts():
        for block, offset in enumerate(offsets):
            seed = (args.seed + offset) % 2**31
            quota = (target * (block + 1) + len(offsets) - 1) // len(offsets)
            wins = attempts = 0
            with greedy_runner(args, spec, expected_layout,
                               reference, task, seed) as runner:
                # Give each seed block at least one rollout, even for very small targets.
                for index in range(settings["attempts_per_seed"]):
                    if index and len(successful) >= quota:
                        break
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
                    successful.append(reference_record(runner, batch, episode) if label_action_sets else episode)
            blocks.append(dict(seed=seed, attempts=attempts, wins=wins))
    labels, diagnostics = {}, {}
    if label_action_sets:
        delta, calibration = calibrate_delta(successful, settings, task)
        candidates = list(dict.fromkeys([delta, *settings["delta_candidates_by_task"].get(task, [])]))
        successful, statistics = label_records(successful, delta, candidates)
        labels = dict(delta=delta)
        diagnostics = dict(calibration=calibration, candidates=statistics)
    path = out / "action_set_memory" / f"phase{phase:04d}"
    path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = str(checkpoint.relative_to(out) / "agent.th")
    pool = dict(task=task, phase=phase, layout=expected_layout,
                reference_checkpoint=checkpoint_path, episodes=successful, **labels)
    th.save(pool, path / "success.th")
    th.save(dict(task=task, phase=phase, episodes=failed), path / "failure.th")
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
                            probe_records.append(reference_record(runner, batch, episode))
                        except Exception:
                            log_probe_error(traceback.format_exc())
            if remaining is not None:
                remaining -= count
    wins = sum(row["won"] for row in results)
    return dict(task=task, wins=wins, episodes=len(results), win_rate=wins / len(results), outcomes=results)


def train_sequence(protocol, seed, out):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    args = SimpleNamespace(**config(protocol, seed, out))
    logger = base.Logger(logging.getLogger(out.name))
    original_log = logger.log_stat

    def log(key, value, t_env, **kwargs):
        base.append(out / "metrics.jsonl", dict(key=key, value=float(value), t_env=int(t_env)))
        original_log(key, value, t_env, **kwargs)

    logger.log_stat = log
    runner = base.EpisodeRunner(args, logger)
    phase = episode = updates = 0
    try:
        info = runner.get_env_info()
        expected_layout = layout(runner)
        if (expected_layout["n_agents"], expected_layout["n_enemies"], expected_layout["map_type"],
                expected_layout["attack_offset"], expected_layout["n_actions"]) != (5, 5, "protoss_gen", 6, 11):
            raise ValueError(f"unsupported SMACv2 Protoss action/agent layout: {expected_layout}")
        args.n_agents, args.n_actions, args.state_shape = info["n_agents"], info["n_actions"], info["state_shape"]
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
        preference = None
        if method == "action_set_preference" and settings["weight"]:
            preference = ActionSetPreference(settings, np.random.default_rng(seed), augmented, groups,
                                             lambda row: base.append(out / "action_set_updates.jsonl", row))
        elif method in RANK_METHODS and protocol[method]["weight"]:
            ranking = dict(protocol[method], batch_episodes=settings["batch_episodes"])
            preference = RankingPreference(method, out, ranking, np.random.default_rng(seed), augmented, groups,
                                            lambda row: base.append(out / f"{method}_updates.jsonl", row))

        stages = protocol["stages"]
        evaluation = protocol["evaluation"]
        initial, before, latest, results = {}, {}, {}, []
        probes = {}
        nominal_end = 0

        def status(state, **extra):
            base.write(out / "status.json", dict(status=state, phase=phase, t_env=runner.t_env,
                                                episode=episode, updates=updates, **extra))

        def agent_snapshot():
            return {name: value.detach().cpu().clone() for name, value in mac.agent.state_dict().items()}

        def probe_error(task, error):
            base.append(out / "action_set_probe_errors.jsonl", dict(phase=phase, task=task, error=error))

        def evaluate(reference, task, label, probe_records=None):
            focus = label in ("stage_end", "initial_reference", "next_unseen")
            row = evaluate_task(args, spec, expected_layout, reference, task,
                                evaluation_settings(protocol, task, focus), probe_records=probe_records,
                                probe_limit=settings["probe_episodes_per_seed"],
                                log_probe_error=lambda error: probe_error(task, error))
            base.append(out / "evaluations.jsonl", dict(phase=phase, label=label, t_env=runner.t_env, **row))
            return row["win_rate"]

        status("running")
        reference = agent_snapshot()
        for task in dict.fromkeys(stage["task"] for stage in stages if stage["task"] != stages[0]["task"]):
            initial[task] = evaluate(reference, task, "initial_reference")

        for phase, stage in enumerate(stages):
            task = stage["task"]
            begin = runner.t_env
            nominal_end += stage["steps"]
            end = nominal_end if protocol.get("phase_boundary") == "cumulative" else begin + stage["steps"]
            # Only current-task training episodes enter the TD buffer.
            buffer = base.ReplayBuffer(scheme, groups, args.buffer_size, info["episode_limit"] + 1,
                                       preprocess=preprocess, device="cpu")
            runner.env.set_task(task)
            base.log_task_epsilon(runner, out, phase, task, "start")
            if preference is not None:
                preference.phase, preference.task = phase, task
            base.append(out / "transitions.jsonl", dict(phase=phase, task=task, start=begin, target=end))
            status("running")
            evaluate(agent_snapshot(), task, "phase_start")
            interval = evaluation["interval"]
            next_eval = begin + interval
            while runner.t_env < end:
                with th.no_grad():
                    batch = runner.run()
                buffer.insert_episode_batch(batch)
                if buffer.can_sample(args.batch_size):
                    sample = buffer.sample(args.batch_size)
                    sample = sample[:, :int(sample.max_t_filled())]
                    sample.to(args.device)
                    learner.train(sample, runner.t_env, episode, preference=preference)
                    updates += 1
                episode += 1
                if episode % 25 == 0:
                    status("running")
                if interval and next_eval <= runner.t_env < end:
                    evaluate(agent_snapshot(), task, "periodic")
                    next_eval = begin + ((runner.t_env - begin) // interval + 1) * interval
            del buffer
            base.log_task_epsilon(runner, out, phase, task, "end")
            checkpoint = out / "models" / f"phase{phase:04d}_{runner.t_env}"
            checkpoint.mkdir(parents=True, exist_ok=False)
            learner.save_models(str(checkpoint))
            reference = agent_snapshot()
            probe_records = ([] if method == "action_set_preference" and preference is not None
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
            if phase + 1 < len(stages):
                next_task = stages[phase + 1]["task"]
                if next_task not in row:
                    before[next_task] = evaluate(reference, next_task, "next_unseen")
            base.write(out / "retention.json", dict(stages=results, initial_reference=initial,
                                                   before_first_training=before))
            if preference is not None:
                status("collecting_memory")
                pool = collect_memory(args, spec, expected_layout, reference, checkpoint,
                                      task, phase, settings, out)
                preference.history[phase] = pool
                if probe_records:
                    try:
                        labeled, _ = label_records(probe_records, pool["delta"])
                        probe = dict(task=task, source_phase=phase, delta=pool["delta"], episodes=labeled)
                        th.save(probe, out / "action_set_memory" / f"phase{phase:04d}" / "probe.th")
                        probes[phase] = probe
                    except Exception:
                        probe_error(task, traceback.format_exc())
                for probe in probes.values():
                    try:
                        with th.no_grad():
                            probe_batch, good = batch_episodes(probe["episodes"], augmented, groups, args.device)
                            _, _, stats = preference_loss(mac, probe_batch, good, settings["margin"])
                        metrics = preference_statistics(stats.cpu().tolist())
                        base.append(out / "action_set_probes.jsonl", dict(
                            phase=phase, t_env=runner.t_env, task=probe["task"],
                            source_phase=probe["source_phase"], delta=probe["delta"],
                            episodes=len(probe["episodes"]), win_rate=row[probe["task"]], **metrics))
                    except Exception:
                        probe_error(probe["task"], traceback.format_exc())
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", type=int)
    mode.add_argument("--evaluate-run", type=Path)
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()
    out = Path(os.environ["MARL_RUN_DIR"])
    protocol = base.PROTOCOL
    protocol = validate(protocol, protocol["algorithm"], protocol["method"])
    th.set_num_threads(1)
    logging.basicConfig(level=logging.INFO)
    if args.evaluate_run:
        evaluate_final(args.evaluate_run, out, args.episodes, protocol)
    else:
        train_sequence(protocol, args.seed, out)


if __name__ == "__main__":
    main()
