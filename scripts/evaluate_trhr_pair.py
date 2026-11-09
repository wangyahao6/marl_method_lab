"""Independent, paired final SMAC evaluation and A2 counterfactual analysis."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("SC2PATH", str(ROOT / "runtime/StarCraftII"))

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers import REGISTRY as controllers
from runners.episode_runner import EpisodeRunner
from utils.logging import Logger


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def rows(run):
    return [json.loads(line) for line in (run / "a2_windows.jsonl").read_text().splitlines()]


def identify_woken(trace, target, consecutive):
    """Find the first increase and later sustained recovery of each adjusted unit."""
    width = len(trace[0]["offset"])
    previous = np.zeros(width)
    adjusted, recovered = {}, {}
    streak = np.zeros(width, dtype=int)
    for row in trace:
        offset = np.asarray(row["offset"])
        rates = row["activation_rate"]
        if rates is not None:
            seen = np.array([str(i) in adjusted for i in range(width)])
            streak = np.where(seen & (np.asarray(rates) >= target), streak + 1, 0)
            for i in np.flatnonzero(streak >= consecutive):
                recovered.setdefault(str(i), row["t_env"])
        else:
            streak[:] = 0
        for i in np.flatnonzero(offset > previous):
            adjusted.setdefault(str(i), row["t_env"])
        previous = offset
    return dict(adjusted=adjusted, recovered=recovered)


def construct(config, seed, logger):
    args = SimpleNamespace(**config)
    args.device = "cpu"
    args.use_cuda = False
    args.test_nepisode = 32
    args.env_args = dict(args.env_args, seed=seed)
    runner = EpisodeRunner(args, logger)
    info = runner.get_env_info()
    args.n_agents, args.n_actions = info["n_agents"], info["n_actions"]
    scheme = dict(
        state=dict(vshape=info["state_shape"]),
        obs=dict(vshape=info["obs_shape"], group="agents"),
        actions=dict(vshape=(1,), group="agents", dtype=th.long),
        avail_actions=dict(vshape=(args.n_actions,), group="agents", dtype=th.int),
        reward=dict(vshape=(1,)), terminated=dict(vshape=(1,), dtype=th.uint8),
    )
    groups = dict(agents=args.n_agents)
    preprocess = dict(actions=("actions_onehot", [OneHot(out_dim=args.n_actions)]))
    batch = EpisodeBatch(scheme, groups, 1, 2, preprocess=preprocess)
    mac = controllers[args.mac](batch.scheme, groups, args)
    runner.setup(scheme, groups, preprocess, mac)
    return runner, mac


@th.no_grad()
def evaluate(config, state, seed, replacement=None, collect=False):
    logger = Logger(logging.getLogger("trhr_eval"))
    runner, mac = construct(config, seed, logger)
    mac.agent.load_state_dict(state)
    mac.agent.eval()
    episodes, replay = [], []
    hook = None
    if replacement is not None:
        indices, means = replacement

        def replace(_module, _inputs, output):
            result = output.clone()
            result[:, indices] = means
            return result

        hook = mac.agent.net[4].register_forward_hook(replace)
    try:
        for i in range(32):
            batch = runner.run(test_mode=True)
            episodes.append(dict(index=i, length=runner.t, won=bool(runner.last_episode_info.get("battle_won", False))))
            if collect and i < 8:
                replay.append(batch[:, :runner.t+1])
        return episodes, replay
    finally:
        if hook is not None:
            hook.remove()
        runner.close_env()


@th.no_grad()
def replay_stats(mac, replay):
    values = []
    handle = mac.agent.net[4].register_forward_hook(lambda m, i, o: values.append(o.detach()))
    try:
        chunks = []
        for batch in replay:
            values.clear()
            mac.init_hidden(1)
            for t in range(batch.max_seq_length - 1):
                mac.forward(batch, t, test_mode=True)
            available = batch["avail_actions"][0, :-1, :, 1:].bool().any(dim=-1)
            chunks.append(th.stack(values)[available])
        act = th.cat(chunks)
        return dict(rows=len(act), activation_rate=(act > 0).float().mean(dim=0).tolist(),
                    mean=act.mean(dim=0).tolist(), std=act.std(dim=0, unbiased=False).tolist())
    finally:
        handle.remove()


def matching_controls(stats, recovered, adjusted):
    rate = np.asarray(stats["activation_rate"])
    std = np.asarray(stats["std"])
    features = np.column_stack((rate, np.log1p(std)))
    unused = set(range(len(rate))) - set(adjusted)
    if len(unused) < len(recovered):
        return []
    scale = np.maximum(features.std(axis=0), 1e-8)
    matches = []
    for i in recovered:
        j = min(unused, key=lambda k: (np.linalg.norm((features[i] - features[k]) / scale), k))
        unused.remove(j)
        matches.append(j)
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trhr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    th.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    runs = dict(baseline=args.baseline, trhr=args.trhr)
    configs = {group: json.loads((path / "config.json").read_text()) for group, path in runs.items()}
    final = {group: th.load(path / "final_model/agent.th", map_location="cpu") for group, path in runs.items()}
    settings = configs["trhr"]["trhr"]
    trace = rows(args.trhr)
    selection = identify_woken(
        [row for row in trace if row["window_updates"] == settings["window_updates"]],
        settings["target_rate"], settings["low_windows"]
    )
    write_json(args.output / "selection.json", selection)
    seed = configs["trhr"]["seed"]
    seeds = [916001 + seed * 10 + k for k in range(4)]
    results = {}
    replay = []
    with (args.output / "episodes.jsonl").open("a", buffering=1) as stream:
        def blocks(group, name, replacement=None, collect=False):
            combined = []
            for block_seed in seeds:
                episodes, examples = evaluate(configs[group], final[group], block_seed, replacement, collect)
                replay.extend(examples)
                for episode in episodes:
                    stream.write(json.dumps(dict(condition=name, env_seed=block_seed, **episode)) + "\n")
                combined.extend(episodes)
            results[name] = dict(episodes=len(combined), wins=sum(row["won"] for row in combined),
                                 win_rate=float(np.mean([row["won"] for row in combined])))

        blocks("baseline", "baseline", collect=True)
        th.save(replay, args.output / "fixed_replay.th")
        blocks("trhr", "trhr")

        # Use the same baseline trajectories with all four weight/offset combinations.
        fixed = {}
        for group in ("baseline", "trhr"):
            runner, mac = construct(configs[group], seeds[0], Logger(logging.getLogger("trhr_replay")))
            runner.close_env()
            mac.agent.eval()
            mac.agent.load_state_dict(final[group])
            fixed[group + "_final"] = replay_stats(mac, replay)
            if group == "trhr":
                offset = mac.agent.net[4].offset.clone()
                mac.agent.net[4].offset.zero_()
                fixed["trhr_final_zero_offset"] = replay_stats(mac, replay)
                mac.agent.load_state_dict(th.load(args.trhr / "initial_agent.th", map_location="cpu"))
                fixed["trhr_initial"] = replay_stats(mac, replay)
                mac.agent.net[4].offset.copy_(offset)
                fixed["trhr_initial_final_offset"] = replay_stats(mac, replay)
        write_json(args.output / "fixed_replay_stats.json", fixed)
        recovered = sorted(map(int, selection["recovered"]))
        adjusted = set(map(int, selection["adjusted"]))
        control = matching_controls(fixed["trhr_final"], recovered, adjusted) if recovered else []
        means = th.tensor(fixed["trhr_final"]["mean"])
        write_json(args.output / "ablation.json", dict(recovered=recovered, control=control,
                   definition="Replace selected A2 outputs by their fixed-replay means"))
        if recovered:
            blocks("trhr", "trhr_mean_recovered", (recovered, means[recovered]))
        if control:
            blocks("trhr", "trhr_mean_control", (control, means[control]))
    write_json(args.output / "final.json", dict(seed=seed, eval_seeds=seeds, conditions=results))


if __name__ == "__main__":
    main()
