"""Offline S-history joint action retention for shared-prefix VDN64 FT and TKPR-K1.

Run this against the training-source commit (2c761c44) and its original run dirs.
No TKPR success memory or training replay is read.
"""

import argparse
import csv
import gzip
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th

import train as base
from drivers.continual.action_set_train import augmented_scheme, greedy_runner
from drivers.continual.common.runtime import episode_spec
from drivers.continual.memory.episodes import pack_episode
from learners.action_set_preference import batch_episodes, decision_masks, unroll


SOURCE_COMMIT = "2c761c44b3aa2ab4dda68f71ec9b7dccbdb86818"
SEEDS = (2222, 3535)
METHODS = ("VDN-FT", "TKPR-K1")
TASKS = ("stalker", "zealot", "colossus")
EPISODES = 32
ANALYSIS_SEED_OFFSET = 1_500_000


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def input_runs(index, runs_root):
    rows = list(csv.DictReader(Path(index).open()))
    selected = {}
    for seed in SEEDS:
        prefix = next(row for row in rows if int(row["seed"]) == seed and row["method"] == "Shared S")
        branches = {method: next(row for row in rows if int(row["seed"]) == seed and
                                 row["method"] == method) for method in METHODS}
        for row in (prefix, *branches.values()):
            if row["commit"] != SOURCE_COMMIT or row["status"] != "completed":
                raise ValueError(f"Unexpected source commit or status: {row['run_id']}")
        if any(row["parent_run_id"] != prefix["run_id"] for row in branches.values()):
            raise ValueError(f"Branch parent mismatch for seed {seed}")
        selected[seed] = (prefix, branches)
    return selected


def validate_seed(seed, protocol):
    candidate = ANALYSIS_SEED_OFFSET + seed
    evaluation = protocol["evaluation"]
    task_offsets = [i * evaluation["task_seed_stride"] for i in range(len(TASKS))]
    used = {seed}
    for offset in task_offsets:
        for key in ("seeds", "focus_seeds"):
            used.update(value + offset for value in evaluation.get(key, []) or [])
    used.update((seed + offset) % 2**31 for offset in protocol["action_set"]["seed_offsets"])
    if candidate in used:
        raise ValueError(f"Analysis seed overlaps training, evaluation or history: {candidate}")
    return candidate


def resolve_checkpoints(prefix, branches, runs_root):
    parent = Path(runs_root) / prefix["run_id"]
    parent_retention = read_json(parent / "retention.json")
    parent_state = th.load(parent / "return_state.th", map_location="cpu")
    first = parent_retention["stages"][0]
    if (first["phase"], first["task"]) != (0, "stalker"):
        raise ValueError("Shared parent does not end at S")
    s_dir = parent / first["checkpoint"]
    checkpoint_paths = {"S": s_dir / "agent.th"}
    detail = {"S": dict(run_id=prefix["run_id"], t_env=first["t_env"],
                        path=str(checkpoint_paths["S"]), sha256=digest(checkpoint_paths["S"]))}
    for method, row in branches.items():
        branch = Path(runs_root) / row["run_id"]
        restore = read_json(branch / "branch_restore.json")
        if (Path(restore["parent"]).resolve() != parent.resolve() or
                Path(restore["checkpoint"]).resolve() != s_dir.resolve() or
                restore["model_hash"] != parent_state["model_hash"] or
                row["restore_model_hash"] != restore["model_hash"] or
                row["prefix_model_hash"] != restore["model_hash"]):
            raise ValueError(f"Shared-prefix restore evidence mismatch: {row['run_id']}")
        stages = read_json(branch / "retention.json")["stages"]
        if [(item["phase"], item["task"]) for item in stages] != list(enumerate(TASKS)):
            raise ValueError(f"Stage/task mismatch: {row['run_id']}")
        for phase, label in ((1, "Z"), (2, "C")):
            checkpoint = branch / stages[phase]["checkpoint"] / "agent.th"
            checkpoint_paths[(method, label)] = checkpoint
            detail[(method, label)] = dict(run_id=row["run_id"], phase=phase,
                                           t_env=stages[phase]["t_env"],
                                           path=str(checkpoint), sha256=digest(checkpoint),
                                           s_win_rate=stages[phase]["win_rates"]["stalker"])
    return checkpoint_paths, detail, parent_state["model_hash"]


def collect_history(prefix_dir, checkpoint, seed, output):
    config = read_json(prefix_dir / "config.json")
    protocol = read_json(prefix_dir / "protocol.json")
    env_info = read_json(prefix_dir / "env_info.json")
    args = SimpleNamespace(**config)
    if not args.test_greedy:
        raise ValueError("Original evaluation must use greedy actions")
    independent_seed = validate_seed(seed, protocol)
    spec = episode_spec(env_info)
    layout = env_info["action_layout"]
    agent_state = th.load(checkpoint, map_location="cpu")
    episodes = []
    with greedy_runner(args, spec, layout, agent_state, "stalker", independent_seed) as runner:
        for index in range(EPISODES):
            batch = runner.run(test_mode=True)
            won = bool(runner.last_episode_info.get("battle_won", False))
            episode = pack_episode(batch, runner.t, seed=independent_seed, index=index, won=won)
            required = {"obs", "avail_actions", "actions_onehot", "terminated", "filled", "state"}
            if not required <= set(episode["data"]):
                raise ValueError(f"Missing complete-episode fields: {required - set(episode['data'])}")
            episodes.append(episode)
    path = output / f"historical_trajectories_seed{seed}.th.gz"
    with gzip.open(path, "wb", compresslevel=6) as stream:
        th.save(dict(task="stalker", seed=independent_seed,
                     source_checkpoint_sha256=digest(checkpoint), episodes=episodes), stream)
    return episodes, spec, args, dict(path=path.name, sha256=digest(path), seed=independent_seed,
                                      episodes=len(episodes), wins=sum(ep["won"] for ep in episodes),
                                      failures=sum(not ep["won"] for ep in episodes),
                                      total_steps=sum(ep["length"] for ep in episodes))


def greedy_actions(agent_state, args, spec, episodes):
    scheme, groups, _ = spec
    augmented = augmented_scheme(spec)
    offline_args = SimpleNamespace(**vars(args))
    offline_args.device, offline_args.use_cuda = "cpu", False
    mac = base.controllers[offline_args.mac](augmented, groups, offline_args)
    mac.agent.load_state_dict(agent_state)
    mac.agent.eval().requires_grad_(False)
    batch, _ = batch_episodes(episodes, augmented, groups, "cpu", include_good=False)
    with th.no_grad():
        q = unroll(mac, batch, batch.max_seq_length - 1)
    legal, decision = decision_masks(batch)
    actions = q.masked_fill(~legal, -th.inf).argmax(dim=-1)
    return actions, decision, batch


def episode_rows(seed, method, phase, episodes, reference_actions, current_actions, decision):
    records, per_agent = [], []
    for index, episode in enumerate(episodes):
        valid = decision[index].cpu().numpy().astype(bool)
        changed = ((reference_actions[index] != current_actions[index]).cpu().numpy() & valid)
        alive_counts = valid.sum(axis=-1)
        joint_valid = alive_counts >= 2
        changed_counts = changed.sum(axis=-1)
        decisions = int(valid.sum())
        joint_steps = int(joint_valid.sum())
        if not decisions or not joint_steps:
            raise ValueError(f"Episode has no valid live decisions: seed {seed}, episode {index}")
        retained = int(decisions - changed.sum())
        unchanged_joint = int((joint_valid & (changed_counts == 0)).sum())
        multi_change = int((joint_valid & (changed_counts >= 2)).sum())
        records.append(dict(seed=seed, method=method, phase=phase, episode=index,
                            trajectory_seed=episode["seed"], length=episode["length"],
                            won=int(episode["won"]), valid_agent_decisions=decisions,
                            retained_agent_decisions=retained, joint_steps=joint_steps,
                            unchanged_joint_steps=unchanged_joint,
                            changed_agents_on_joint_steps=int(changed_counts[joint_valid].sum()),
                            multi_agent_change_steps=multi_change,
                            iar=retained / decisions, jaa=unchanged_joint / joint_steps,
                            mean_changed_agents=float(changed_counts[joint_valid].mean()),
                            multi_agent_change_rate=multi_change / joint_steps))
        for agent in range(valid.shape[1]):
            count = int(valid[:, agent].sum())
            per_agent.append(dict(seed=seed, method=method, phase=phase, episode=index,
                                  agent_id=agent, valid_decisions=count,
                                  retained_decisions=int(count - changed[:, agent].sum()),
                                  iar=(float(1 - changed[:, agent].sum() / count) if count else None)))
    return records, per_agent


def summarize(seed, method, phase, episodes, agents, s_win_rate):
    fields = ("iar", "jaa", "mean_changed_agents", "multi_agent_change_rate")
    result = dict(seed=seed, method=method, phase=phase, checkpoint_task=TASKS[phase],
                  episodes=len(episodes), wins=sum(row["won"] for row in episodes),
                  valid_agent_decisions=sum(row["valid_agent_decisions"] for row in episodes),
                  joint_steps=sum(row["joint_steps"] for row in episodes),
                  s_win_rate_original=s_win_rate)
    for field in fields:
        values = np.asarray([row[field] for row in episodes], dtype=np.float64)
        result[field] = float(values.mean())
        result[field + "_episode_sd"] = float(values.std(ddof=1))
    agent_rows = []
    for agent_id in range(5):
        rows = [row for row in agents if row["agent_id"] == agent_id and row["iar"] is not None]
        values = np.asarray([row["iar"] for row in rows], dtype=np.float64)
        agent_rows.append(dict(seed=seed, method=method, phase=phase,
                               checkpoint_task=TASKS[phase], agent_id=agent_id,
                               episodes_with_live_decisions=len(rows),
                               valid_decisions=sum(row["valid_decisions"] for row in rows),
                               iar=float(values.mean()) if len(values) else None,
                               iar_episode_sd=float(values.std(ddof=1)) if len(values) > 1 else None))
    return result, agent_rows


def analyze(index, runs_root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selected = input_runs(index, runs_root)
    summary_rows, agent_rows, all_episodes = [], [], []
    manifest = dict(training_source_commit=SOURCE_COMMIT, task="stalker",
                    training_seeds=list(SEEDS), episodes_per_seed=EPISODES,
                    trajectory_seed_rule="1500000 + training_seed",
                    trajectory_source="fresh greedy S rollout from the shared parent checkpoint",
                    teacher_forced=True, history_memory_used=False, seeds={})
    for seed in SEEDS:
        prefix, branches = selected[seed]
        parent_dir = Path(runs_root) / prefix["run_id"]
        paths, model_details, parent_hash = resolve_checkpoints(prefix, branches, runs_root)
        episodes, spec, args, trajectory = collect_history(parent_dir, paths["S"], seed, output)
        reference_state = th.load(paths["S"], map_location="cpu")
        reference_actions, decision, _ = greedy_actions(reference_state, args, spec, episodes)
        counts = dict(valid_agent_decisions=int(decision.sum().item()),
                      joint_steps=int((decision.sum(dim=-1) >= 2).sum().item()))
        model_manifest = {"S": model_details["S"]}
        for method in METHODS:
            branch = Path(runs_root) / branches[method]["run_id"]
            retention = read_json(branch / "retention.json")["stages"]
            model_manifest[method] = {"Z": model_details[(method, "Z")],
                                      "C": model_details[(method, "C")],
                                      "parent_run_id": prefix["run_id"],
                                      "restore_model_hash": branches[method]["restore_model_hash"]}
            for phase, label in ((0, "S"), (1, "Z"), (2, "C")):
                current_actions = (reference_actions if phase == 0 else
                                   greedy_actions(th.load(paths[(method, label)], map_location="cpu"),
                                                  args, spec, episodes)[0])
                rows, per_agent = episode_rows(seed, method, phase, episodes,
                                               reference_actions, current_actions, decision)
                all_episodes.extend(rows)
                aggregate, agents = summarize(seed, method, phase, rows, per_agent,
                                              retention[phase]["win_rates"]["stalker"])
                summary_rows.append(aggregate)
                agent_rows.extend(agents)
        manifest["seeds"][str(seed)] = dict(parent_run_id=prefix["run_id"],
                                            parent_model_hash=parent_hash,
                                            trajectory=trajectory, masks=counts,
                                            models=model_manifest)
    write_csv(output / "joint_action_retention.csv", summary_rows)
    write_csv(output / "per_agent_retention.csv", agent_rows)
    write_csv(output / "episode_metrics.csv", all_episodes)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    th.set_num_threads(1)
    analyze(args.runs_csv, args.runs_root, args.output)


if __name__ == "__main__":
    main()
