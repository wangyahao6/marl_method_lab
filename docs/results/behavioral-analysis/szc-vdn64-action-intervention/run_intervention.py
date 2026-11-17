"""Greedy SMACv2 evaluation with frozen-agent overrides on the executed history."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import random
from shutil import copy2
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "szc-vdn64-greedy-action-agreement"))
from run_analysis import read, sha, write

ARMS = ("A", "B", "C", "D")
THRESHOLD = 0.5
AGRD = ROOT / "results/szc_vdn64_greedy_action_agreement_20261117_attempt002"
AUDIT = HERE.parent / "szc-vdn64-action-regret/audit.json"


def inputs(pilot):
    manifest_path = AGRD / "manifest.json"
    source = read(manifest_path)
    audit = read(AUDIT)
    if (audit["source_run"] != AGRD.name
            or sha(manifest_path) != audit["agr_inputs"]["manifest.json"]):
        raise ValueError("AGR manifest differs from the action-regret audit")
    points = {p["path"]: p for p in audit["checkpoints"]}
    cases = []
    for ref in source["references"]:
        for case in ref["comparisons"]:
            if case["reference"] != "own_stage":
                continue
            env = ref["env_args"]
            case = dict(case, env=env, reference_id=ref["id"])
            seeds = (91101,) if case["task"] == "stalker" else (91201,)
            case["env_seeds"] = list(range(seeds[0], seeds[0] + (1 if pilot else 4)))
            for field in ("old_checkpoint", "new_checkpoint"):
                p = case[field]
                if p != points[p["path"]] or sha(ROOT / p["path"]) != p["sha256"]:
                    raise ValueError(f"Checkpoint differs from action-regret audit: {p['path']}")
            cases.append(case)
    if pilot:
        checks = {("FRPR", "stalker", "C"), ("K1", "zealot", "C")}
        cases = [c for c in cases if c["seed"] == 1111 and
                 (c["method"], c["task"], c["stage"]) in checks]
    else:
        from collections import Counter
        cohorts = Counter(c["cohort"] for c in cases)
        if cohorts != {"k_ablation": 15, "earlier_frpr": 6}:
            raise ValueError(f"Unexpected case coverage: {cohorts}")
    return cases


def plan(output, pilot, pilot_audit):
    if not pilot and (pilot_audit is None or not read(pilot_audit)["implementation_checks"]["pilot_passed"]):
        raise ValueError("Full evaluation requires a successful pilot")
    cases = inputs(pilot)
    cohorts = defaultdict(list)
    jobs = []
    for i, case in enumerate(cases):
        key = f"case-{i:02d}"
        jobs.extend(dict(job_id=f"{key}-{arm}", arm=arm, case_ids=[case["id"]]) for arm in ("A", "B"))
        cohorts[case["reference_id"]].append(case["id"])
    for i, case_ids in enumerate(cohorts.values()):
        jobs.extend(dict(job_id=f"reference-{i:02d}-{arm}", arm=arm, case_ids=case_ids)
                    for arm in ("C", "D"))
    files = ("run_intervention.py", "summarize_intervention.py")
    manifest = dict(run_id=output.name, is_pilot=pilot, episodes_per_seed=2 if pilot else 32,
                    threshold=THRESHOLD, audit_input=str(AUDIT.relative_to(ROOT)),
                    audit_input_sha256=sha(AUDIT), agr_manifest_sha256=sha(AGRD / "manifest.json"),
                    base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        cwd=ROOT, text=True).strip(),
                    source_sha256={f: sha(HERE / f) for f in files}, cases=cases, jobs=jobs)
    if pilot_audit:
        pilot_record = read(pilot_audit)
        if manifest["source_sha256"] != pilot_record["source_sha256"]:
            raise ValueError("Pilot and full evaluation analysis sources differ")
        manifest["pilot"] = dict(run_id=pilot_record["run_id"], sha256=sha(pilot_audit),
                                 checks=pilot_record["implementation_checks"])
    output.mkdir(parents=True, exist_ok=False)
    write(output / "manifest.json", manifest)
    source_dir = output / "source"
    source_dir.mkdir()
    for f in files:
        copy2(HERE / f, source_dir / f)

    common = dict(cwd=str(ROOT), queue=output.name, queue_limit=6,
                  priority=1, resources={"ram_mib": 2000}, metadata={"results": str(output)})
    tasks = [dict(common, id=f"{output.name}-{job['job_id']}", type="evaluation", dependencies=[],
                  argv=[sys.executable, str(HERE / files[0]), "evaluate", str(output), job["job_id"]])
             for job in jobs]
    tasks.append(dict(common, id=f"{output.name}-summary", type="summary",
                      dependency_mode="terminal", dependencies=[ta["id"] for ta in tasks],
                      argv=[sys.executable, str(HERE / files[1]), str(output)]))
    write(output / "plan.json", dict(id=output.name, tasks=tasks))


def load_checkpoint(checkpoint, layout, scheme, groups, deps):
    th, controllers, _, _, _ = deps
    f = ROOT / checkpoint["path"]
    if sha(f) != checkpoint["sha256"]:
        raise ValueError(f"Checkpoint changed: {f}")
    run = ROOT / "runs" / checkpoint["run_id"]
    cfg = read(run / "config.json")
    info = read(run / "env_info.json")
    if (info["action_layout"] != layout or cfg["mixer"] != "vdn"
            or cfg["mac"] != "basic_mac" or cfg["agent"] != "rnn" or cfg["rnn_hidden_dim"] != 64
            or cfg["agent_output_type"] != "q" or not cfg["obs_last_action"]
            or not cfg["obs_agent_id"] or cfg.get("trhr") or cfg["env_args"].get("heuristic_ai")):
        raise ValueError(f"Unexpected frozen controller or environment layout: {run}")
    cfg.update(device="cpu", use_cuda=False)
    mac = controllers[cfg["mac"]](scheme, groups, SimpleNamespace(**cfg))
    mac.agent.load_state_dict(th.load(f, map_location="cpu"))
    mac.agent.eval().requires_grad_(False)
    return mac


def select_actions(arm, old_actions, new_actions, valid, normalized):
    flips = valid & (old_actions != new_actions)
    if arm == "A":
        return new_actions, flips & False
    if arm == "D":
        return old_actions, flips & False
    override = flips if arm == "C" else flips & (normalized >= THRESHOLD)
    return old_actions.where(override, new_actions), override


def check_selection(th):
    old = th.tensor([[1, 1, 1, 1, 1, 1]])
    new = th.tensor([[1, 2, 2, 2, 2, 2]])
    valid = th.tensor([[True, True, True, True, False, True]])
    norm = th.tensor([[0., 0.499, 0.5, 0.501, 2., 0.]])
    want = {"A": [], "B": [2, 3], "C": [1, 2, 3, 5], "D": []}
    for arm in ARMS:
        chosen, override = select_actions(arm, old, new, valid, norm)
        indices = override[0].nonzero().flatten().tolist()
        expected = old if arm == "D" else new.clone()
        if arm in ("B", "C"):
            expected[0, want[arm]] = old[0, want[arm]]
        if indices != want[arm] or not th.equal(chosen, expected):
            raise ValueError(f"Intervention rule failed for {arm}")


def restore_seeded_game_each_episode(env):
    """Snapshot the freshly seeded map before spawning the first roster."""
    from s2clientprotocol import sc2api_pb2 as sc_pb

    raw = env.wrapped.env
    launch = raw._launch

    def save_initial():
        launch()
        raw._controller._client.send(quick_save=sc_pb.RequestQuickSave())

    def restore_initial():
        raw._controller._client.send(quick_load=sc_pb.RequestQuickLoad())

    # Keep the normal spawn distribution and reset bookkeeping in the wrapper.
    # Replace only the isolated evaluation instance's game reset.
    raw._launch = save_initial
    raw._restart = restore_initial


def initial_signature(env, obs, state, legal):
    import numpy as np
    raw = env.wrapped.env
    units = [[u.unit_type, u.pos.x, u.pos.y, u.health, u.shield, u.energy]
             for side in (raw.agents, raw.enemies) for _, u in sorted(side.items())]
    starts = [np.asarray(raw.episode_config[k]["item"]).tolist()
              for k in ("ally_start_positions", "enemy_start_positions")]
    def digest(o):
        return hashlib.sha256(json.dumps(o, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return digest(starts), digest([units, obs.tolist(), state.tolist(), legal.tolist()])


def play(env, batch, old, new_macs, arm, deps, verify_history):
    th, _, _, reference_order, np = deps
    env.reset()
    old.init_hidden(1)
    for mac in new_macs.values():
        mac.init_hidden(1)
    team_return = 0.0
    length = 0
    done = False
    raw_env = env.wrapped.env
    n = env.n_agents
    # Each row stores slot-wise counts and sums, in float64 for accumulation.
    counts = {key: np.zeros((8, n), dtype=np.float64) for key in new_macs}
    action_trace = hashlib.sha256()
    while not done:
        obs = np.asarray(env.get_obs(), dtype=np.float32)
        state = np.asarray(env.get_state(), dtype=np.float32)
        legal_np = np.asarray(env.get_avail_actions(), dtype=np.int32)
        if length == 0:
            spawn_hash, initial_hash = initial_signature(env, obs, state, legal_np)
            initial_game_loop = int(raw_env._obs.observation.game_loop)
        batch.update({"obs": obs[None], "state": state[None],
                      "avail_actions": legal_np[None]}, ts=length)
        legal = batch["avail_actions"][:, length].bool()
        if not legal.any(dim=-1).all():
            raise ValueError("An agent has no legal action")
        valid = (legal.sum(dim=-1) > 1) & legal[..., 1:].any(dim=-1)
        if verify_history:
            previous = batch["actions_onehot"][:, length - 1] if length else th.zeros_like(
                batch["actions_onehot"][:, 0])
            for mac in [old, *new_macs.values()]:
                built = mac._build_inputs(batch, length)
                n_actions = legal.shape[-1]
                cached = built[:, obs.shape[1]:obs.shape[1] + n_actions]
                expected = th.from_numpy(raw_env.last_action).to(cached.dtype)
                if not th.equal(cached, expected) or not th.equal(cached, previous[0]):
                    raise ValueError("GRU input did not use the executed history")
        old_q = old.forward(batch, length, test_mode=True)
        if not th.isfinite(old_q[legal]).all():
            raise ValueError("Non-finite legal old Q")
        old_q = old_q.masked_fill(~legal, -th.inf)
        top, old_actions = old_q.max(dim=-1)
        _, scale, _, _ = reference_order(old_q, legal, q_scale_floor=0.1)
        scale = scale.squeeze(-1)
        actual = None
        for key, mac in new_macs.items():
            new_q = mac.forward(batch, length, test_mode=True)
            if not th.isfinite(new_q[legal]).all():
                raise ValueError("Non-finite legal new Q")
            new_actions = new_q.masked_fill(~legal, -th.inf).argmax(dim=-1)
            pre = top - old_q.gather(-1, new_actions.unsqueeze(-1)).squeeze(-1)
            pre_norm = pre / scale
            chosen, override = select_actions(arm, old_actions, new_actions, valid, pre_norm)
            if actual is None:
                actual = chosen
            elif not th.equal(actual, chosen):
                raise ValueError("Referenced policies in a shared Old/All-Override rollout differ")
            post = top - old_q.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
            flips = valid & (old_actions != new_actions)
            stats = th.stack([valid, flips, flips & (pre_norm >= THRESHOLD), override,
                              pre.masked_fill(~valid, 0), post.masked_fill(~valid, 0),
                              pre_norm.masked_fill(~valid, 0), (post / scale).masked_fill(~valid, 0)])
            counts[key] += stats[:, 0].numpy()
        if not legal.gather(-1, actual.unsqueeze(-1)).all():
            raise ValueError("Illegal action selected")
        actions = actual.squeeze(0).numpy()
        # Use the same joint action for both the environment and the history.
        reward, done, info = env.step(actions)
        batch.update({"actions": actions[None, :, None]}, ts=length)
        action_trace.update(actions.astype("<i8", copy=False).tobytes())
        team_return += float(reward)
        length += 1
    return (dict(won=bool(info.get("battle_won", False)), episode_return=team_return,
                 length=length, spawn_sha256=spawn_hash, initial_state_sha256=initial_hash,
                 initial_game_loop=initial_game_loop, action_sha256=action_trace.hexdigest()),
            counts)


def evaluate(output, job_id):
    import numpy as np
    import torch as th
    th.set_num_threads(1)
    manifest = read(output / "manifest.json")
    for name, digest in manifest["source_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError(f"Analysis source changed: {name}")
    if sha(AUDIT) != manifest["audit_input_sha256"]:
        raise ValueError("Action-regret audit changed")
    job, = (j for j in manifest["jobs"] if j["job_id"] == job_id)
    cases = [c for c in manifest["cases"] if c["id"] in job["case_ids"]]
    first = cases[0]
    run = ROOT / "runs" / first["run_id"]
    os.environ["MARL_PROTOCOL"] = str(run / "protocol.json")
    sys.path.insert(0, str(ROOT / "drivers/continual"))
    from action_set_train import base, layout, augmented_scheme
    from common.runtime import episode_spec
    from components.episode_buffer import EpisodeBatch
    from hetero_environment import ExactProtoss
    from learners.ranking_preference import reference_order

    deps = th, base.controllers, EpisodeBatch, reference_order, np
    check_selection(th)
    env_info = read(run / "env_info.json")
    spec = episode_spec(env_info)
    scheme, groups, preprocess = spec
    expected = env_info["action_layout"]
    old = load_checkpoint(first["old_checkpoint"], expected, augmented_scheme(spec), groups, deps)
    new_macs = {}
    for c in cases:
        if (c["old_checkpoint"]["sha256"] != first["old_checkpoint"]["sha256"]
                or c["env_seeds"] != first["env_seeds"] or c["task"] != first["task"]
                or c["env"] != first["env"]):
            raise ValueError("Incompatible old reference environments in shared rollout")
        new_macs[c["id"]] = load_checkpoint(c["new_checkpoint"], expected, augmented_scheme(spec), groups, deps)
    dest = output / job_id
    dest.mkdir(exist_ok=False)
    columns = ("valid", "flips", "high_flips", "overrides", "pre_sum_regret", "post_sum_regret",
               "pre_sum_regret_norm", "post_sum_regret_norm")
    for seed in first["env_seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        env = ExactProtoss(**dict(copy.deepcopy(first["env"]), seed=seed))
        restore_seeded_game_each_episode(env)
        rows = []
        try:
            if layout(SimpleNamespace(env=env, get_env_info=env.get_env_info)) != expected:
                raise ValueError("Live SMACv2 layout differs from checkpoint layout")
            env.set_task(first["task"])
            with th.inference_mode():
                for index in range(manifest["episodes_per_seed"]):
                    eid = dict(seed=seed, episode=index)
                    batch = EpisodeBatch(scheme, groups, 1, env.episode_limit + 1, preprocess=preprocess)
                    restarts = env.force_restarts
                    episode, counts = play(env, batch, old, new_macs, job["arm"], deps, manifest["is_pilot"])
                    if env.force_restarts != restarts:
                        raise RuntimeError(f"SC2 forced restart during {job_id} {eid}")
                    row = dict(eid, **episode, cases={key: dict(zip(columns, values.tolist()))
                                                       for key, values in counts.items()})
                    rows.append(row)
        finally:
            env.close()
        write(dest / f"{seed}.json", rows)
        print(f"{job_id}: seed {seed}, {len(rows)} completed", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("output", type=Path)
    planning.add_argument("--pilot", action="store_true")
    planning.add_argument("--pilot-audit", type=Path)
    worker = sub.add_parser("evaluate")
    worker.add_argument("output", type=Path)
    worker.add_argument("job")
    opt = p.parse_args()
    if opt.command == "plan":
        plan(opt.output.resolve(), opt.pilot, opt.pilot_audit)
    else:
        evaluate(opt.output.resolve(), opt.job)
