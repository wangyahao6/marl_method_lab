"""Evaluate stage-end VDN agents on identical histories from the reference policy."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]
TASKS = ("stalker", "zealot", "colossus")
STAGE_NAMES = ("S", "Z", "C")
METHODS = ("K1", "K2", "K3", "K5", "FRPR")
RUN_PATTERN = re.compile(
    r"szc_vdn64_(tkpr_k[1235]|(?:ft_)?frpr)_seed(\d+)(?:_\d{8})?_attempt\d+"
)
INDEPENDENT_Z = "vdnh_Z_seed1111_20261103_attempt001"
BINS = (0.0, 0.25, 0.5, 1.0, 1.5, None)


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir(run_id):
    return ROOT / "runs" / run_id


def checkpoint(run, phase, step, path):
    return dict(run_id=run.name, phase=phase, t_env=step,
                path=str(path.relative_to(ROOT)), sha256=sha(path),
                commit=read(run / "launch.json")["git_commit"])


def discover():
    runs = []
    refs = {}
    independently_trained = run_dir(INDEPENDENT_Z)
    config = read(independently_trained / "config.json")
    end, = map(json.loads, (independently_trained / "phase_end.jsonl").read_text().splitlines())
    z_reference = checkpoint(independently_trained, 0, end["actual_end"],
                             independently_trained / "models" /
                             f"phase0_zealot_{end['actual_end']}" / "agent.th")
    if (read(independently_trained / "status.json")["status"] != "completed"
            or config["tasks"] != ["zealot"] or end["nominal_end"] != 2000000):
        raise ValueError("The independent Z checkpoint is not a completed 2M single-task run")

    for run in sorted((ROOT / "runs").iterdir()):
        match = RUN_PATTERN.fullmatch(run.name)
        if not match:
            continue
        variant, seed = match.groups()
        seed = int(seed)
        method = variant.upper().replace("TKPR_", "") if variant.startswith("tkpr") else "FRPR"
        cohort = "earlier_frpr" if variant == "ft_frpr" else "k_ablation"
        cfg, proto = (read(run / name) for name in ("config.json", "protocol.json"))
        settings = proto[proto["method"]]
        if (read(run / "status.json")["status"] != "completed"
                or cfg["seed"] != seed or proto["algorithm"] != "vdn"
                or cfg["rnn_hidden_dim"] != 64 or cfg["agent"] != "rnn"
                or cfg["mac"] != "basic_mac" or not cfg["obs_last_action"]
                or not cfg["obs_agent_id"] or cfg.get("trhr")
                or cfg["agent_output_type"] != "q"
                or cfg["action_selector"] != "epsilon_greedy"
                or [stage["task"] for stage in proto["stages"]] != list(TASKS)
                or [stage["steps"] for stage in proto["stages"]] != [2000000] * 3
                or proto["epsilon_on_switch"] != "per_task" or proto.get("replay_on_switch", "ft") != "ft"):
            raise ValueError(f"Incompatible or incomplete candidate: {run.name}")
        stages = read(run / "retention.json")["stages"]
        info = read(run / "env_info.json")
        checkpoints = []
        for phase, stage in enumerate(stages):
            step = stage["t_env"]
            if (stage["phase"] != phase or stage["task"] != TASKS[phase]
                    or not 0 <= step - 2000000 * (phase + 1) < info["episode_limit"]
                    or stage["checkpoint"] != f"models/phase{phase:04d}_{step}"):
                raise ValueError(f"Stage/checkpoint mismatch: {run.name} phase {phase}")
            checkpoints.append(checkpoint(run, phase, step, run / stage["checkpoint"] / "agent.th"))
        if len(checkpoints) != 3:
            raise ValueError(f"Missing endpoints: {run.name}")
        evaluations = {}
        with (run / "evaluations.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["label"] != "stage_end":
                    continue
                key = (row["phase"], row["task"])
                if key in evaluations:
                    raise ValueError(f"Duplicate stage-end evaluation: {run.name} {key}")
                _, task = key
                stage = stages[row["phase"]]
                if (row["t_env"] != stage["t_env"] or row["episodes"] != 128
                        or sum(outcome["won"] for outcome in row["outcomes"]) != row["wins"]
                        or len(row["outcomes"]) != row["episodes"]
                        or row["wins"] / row["episodes"] != stage["win_rates"][task]):
                    raise ValueError(f"Stage-end win mismatch: {run.name} {key}")
                evaluations[key] = {k: row[k] for k in ("wins", "episodes", "win_rate")}
        runs.append(dict(run_id=run.name, method=method, seed=seed, cohort=cohort,
                         training_settings=settings, checkpoints=checkpoints))
        for task, phase, reference in (
                ("stalker", 0, "own_stage"), ("zealot", 1, "own_stage"),
                ("zealot", 1, "independent_z")):
            policy = z_reference if reference == "independent_z" else checkpoints[phase]
            env_args = copy.deepcopy(cfg["env_args"])
            env_args.pop("seed")
            comparable_env = copy.deepcopy(env_args)
            comparable_env.pop("rosters")
            comparable_env["capability_config"]["team_gen"].pop("weights")
            standalone_env = copy.deepcopy(config["env_args"])
            standalone_env.pop("seed")
            standalone_env["capability_config"]["team_gen"].pop("weights")
            if comparable_env != standalone_env:
                raise ValueError(f"Independent Z runtime input settings mismatch: {run.name}")
            env_key = hashlib.sha256(json.dumps(env_args, sort_keys=True).encode()).hexdigest()
            key = f"{task}-{policy['sha256'][:16]}-{env_key[:8]}"
            ref = refs.setdefault(key, dict(id=key, task=task, policy=policy, env_args=env_args,
                                           policy_evaluation=evaluations[(phase, task)]
                                           if reference == "own_stage" else None, comparisons=[]))
            if reference == "own_stage" and ref["policy_evaluation"] != evaluations[(phase, task)]:
                raise ValueError(f"Matched old weights have inconsistent historical wins: {run.name}")
            for new in range(phase + 1, 3):
                ref["comparisons"].append(dict(
                    id=f"{run.name}:{task}:{new}:{reference}", run_id=run.name,
                    method=method, seed=seed, cohort=cohort, reference=reference,
                    task=task, stage=STAGE_NAMES[new], phase=new,
                    old_checkpoint=policy if reference == "independent_z" else checkpoints[phase],
                    new_checkpoint=checkpoints[new], evaluation=evaluations[(new, task)]))
    if not runs:
        raise ValueError("No compatible runs found")
    return dict(base_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                analysis_sha256={name: sha(Path(__file__).with_name(name))
                                 for name in ("run_analysis.py", "summarize.py")},
                episodes_per_seed=32, runs=runs, references=list(refs.values()))


def plan(output):
    manifest = discover()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "manifest.json", manifest)
    source = output / "source"
    source.mkdir()
    from shutil import copy2
    for name in manifest["analysis_sha256"]:
        copy2(Path(__file__).with_name(name), source / name)
    batch_id = output.name
    common = dict(cwd=str(ROOT), queue=batch_id, queue_limit=3,
                  resources={"ram_mib": 2000}, metadata={"results": str(output)})
    tasks = []
    for ref in manifest["references"]:
        tasks.append(dict(common, id=f"{batch_id}-{ref['id']}", type="evaluation",
                          dependencies=[], argv=[sys.executable, str(Path(__file__).resolve()),
                                                 "collect", str(output), ref["id"]]))
    tasks.append(dict(common, id=f"{batch_id}-summary", type="summary", dependency_mode="terminal",
                      dependencies=[task["id"] for task in tasks],
                      argv=[sys.executable, str(Path(__file__).with_name("summarize.py")),
                            str(output)]))
    write(output / "plan.json", dict(id=batch_id, tasks=tasks))


def collect(output, ref_id):
    manifest = read(output / "manifest.json")
    ref, = [ref for ref in manifest["references"] if ref["id"] == ref_id]
    for key in manifest["analysis_sha256"]:
        if sha(Path(__file__).with_name(key)) != manifest["analysis_sha256"][key]:
            raise ValueError(f"Analysis source changed since submission: {key}")
    sys.path.insert(0, str(ROOT / "drivers" / "continual"))
    os.environ["MARL_PROTOCOL"] = str(run_dir(ref["comparisons"][0]["run_id"]) / "protocol.json")
    from action_set_train import augmented_scheme, base, greedy_runner
    from common.runtime import episode_spec
    from drivers.continual.memory.episodes import pack_episode
    from learners.action_set_preference import decision_masks, unroll
    from learners.ranking_preference import reference_order
    import torch as th

    def load(point):
        file = ROOT / point["path"]
        if sha(file) != point["sha256"]:
            raise ValueError(f"Checkpoint changed: {file}")
        return th.load(file, map_location="cpu")

    dest = output / ref_id
    dest.mkdir(exist_ok=False)
    th.set_num_threads(1)
    first = ref["comparisons"][0]
    old = load(ref["policy"])
    info = read(run_dir(first["run_id"]) / "env_info.json")
    keys = ("obs_shape", "state_shape", "n_agents", "n_actions", "episode_limit",
            "agent_features", "enemy_features")
    old_info = read(run_dir(ref["policy"]["run_id"]) / "env_info.json")
    if any(info[k] != old_info[k] for k in keys):
        raise ValueError("Reference has different observation or action dimensions/features")
    source = SimpleNamespace(**read(run_dir(ref["policy"]["run_id"]) / "config.json"))
    source.env = "action_set_protoss"
    source.env_args = ref["env_args"]
    source.device = "cpu"
    spec = episode_spec(info)
    current = {}
    for pair in ref["comparisons"]:
        args = SimpleNamespace(**read(run_dir(pair["run_id"]) / "config.json"))
        args.device = "cpu"
        mac = base.controllers[args.mac](augmented_scheme(spec), spec[1], args)
        mac.agent.load_state_dict(load(pair["new_checkpoint"]))
        mac.agent.eval().requires_grad_(False)
        current[pair["id"]] = mac

    episodes = []
    seed_base = 91101 + 100 * TASKS.index(ref["task"])
    audit = []
    for seed in range(seed_base, seed_base + 4):
        trajectories = []
        with greedy_runner(source, spec, info["action_layout"], old, ref["task"], seed) as runner:
            for index in range(manifest["episodes_per_seed"]):
                batch = runner.run(test_mode=True)[:, :runner.t + 1]
                won = bool(runner.env.last_info.get("battle_won", False))
                trajectories.append(pack_episode(batch, runner.t, seed=seed, index=index, won=won))
                legal, alive = decision_masks(batch)
                acts = batch["actions"][:, :-1]
                if not legal.gather(-1, acts).all():
                    raise ValueError(f"Illegal executed action in seed {seed} episode {index}")
                valid = alive & (legal.sum(-1) > 1)
                old_q = unroll(runner.mac, batch, runner.t)
                greedy = old_q.masked_fill(~legal, -th.inf).argmax(-1)
                if not th.equal(greedy, acts.squeeze(-1)) or not old_q.isfinite().all():
                    raise ValueError(f"Replay does not reproduce rollout actions: {seed}/{index}")
                _, _, _, gap = reference_order(old_q, legal, q_scale_floor=0.1)
                for pair_id, mac in current.items():
                    new_q = unroll(mac, batch, runner.t)
                    if not new_q.isfinite().all():
                        raise ValueError(f"Non-finite Q: {pair_id} {seed}/{index}")
                    agreed = (greedy == new_q.masked_fill(~legal, -th.inf).argmax(-1)) & valid
                    bins = []
                    for lo, hi in zip(BINS, BINS[1:]):
                        mask = valid & (gap >= lo)
                        if hi is not None:
                            mask &= gap < hi
                        bins.append([int(mask.sum()), int((agreed & mask).sum())])
                    episodes.append(dict(
                        comparison_id=pair_id, env_seed=seed, episode=index, length=runner.t,
                        old_won=won, valid=int(valid.sum()), agreed=int(agreed.sum()),
                        agent_valid=valid.sum(dim=(0, 1)).tolist(),
                        agent_agreed=agreed.sum(dim=(0, 1)).tolist(), bins=bins))
        archive = dest / f"trajectories-{seed}.th"
        th.save(trajectories, archive)
        audit.append(dict(env_seed=seed, episodes=len(trajectories),
                          wins=sum(ep["won"] for ep in trajectories),
                          transitions=sum(ep["length"] for ep in trajectories),
                          archive=archive.name, sha256=sha(archive)))
        print(f"{ref_id}: finished seed block {seed}", flush=True)
    write(dest / "metrics.json", dict(episodes=episodes, replay_action_mismatches=0,
                                      trajectory_blocks=audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan").add_argument("output", type=Path)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("output", type=Path)
    collect_parser.add_argument("reference_id")
    options = parser.parse_args()
    if options.command == "plan":
        plan(options.output.resolve())
    else:
        collect(options.output.resolve(), options.reference_id)
