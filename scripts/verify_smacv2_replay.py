"""Compare a successful ExactProtoss episode with three fresh action-only replays."""
import argparse
import copy
from contextlib import contextmanager
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch as th
from google.protobuf.json_format import MessageToDict

ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-6
REPLAYS = 3


def read(path):
    return json.loads(path.read_text())


def write(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(value):
    if isinstance(value, np.ndarray):
        return {"array_dtype": value.dtype.str, "data": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    return value


def restore_arrays(value):
    if isinstance(value, dict):
        if value.keys() == {"array_dtype", "data"}:
            return np.array(value["data"], dtype=value["array_dtype"])
        return {key: restore_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_arrays(item) for item in value]
    return value


def global_rng():
    return dict(python=portable(random.getstate()), numpy=portable(np.random.get_state()))


def restore_rng(record):
    def tuples(item):
        return tuple(map(tuples, item)) if isinstance(item, list) else item

    random.setstate(tuples(record["python"]))
    np.random.set_state(tuple(restore_arrays(record["numpy"])))


def distribution_rng(distribution):
    result = {}
    if hasattr(distribution, "rng"):
        result["rng"] = portable(distribution.rng.bit_generator.state)
    for key, value in vars(distribution).items():
        if value.__class__.__module__.endswith(".distributions"):
            result[key] = distribution_rng(value)
    return result


def trace_frame(env):
    state = np.asarray(env.get_state()).tolist()
    avail = np.asarray(env.get_avail_actions()).tolist()
    obs = np.asarray(env.get_obs()).tolist()
    raw = env.wrapped.env
    observation = raw._obs.observation
    # Use the raw API observation because adapter slot objects can retain dead units.
    units = {str(unit.tag): MessageToDict(unit, preserving_proto_field_name=True)
             for unit in observation.raw_data.units if unit.owner in (1, 2)}
    slots = {side: {str(i): str(unit.tag) for i, unit in sorted(items.items())}
             for side, items in (("ally", raw.agents), ("enemy", raw.enemies))}
    return dict(state=state, obs=obs, avail_actions=avail,
                unit_state=units, unit_slots=slots, game_loop=observation.game_loop)


@contextmanager
def initialized_episode(env, *, source=None, sc2_seed, env_seed):
    from pysc2.lib.remote_controller import RemoteController

    raw = env.wrapped.env
    raw._seed = sc2_seed
    launches = []
    original_create_game = RemoteController.create_game

    def capture_seed(controller, request):
        launches.append(int(request.random_seed) if request.HasField("random_seed") else None)
        return original_create_game(controller, request)

    positions = env.wrapped.env_key_to_distribution_map["start_positions"]
    position_before = distribution_rng(positions)
    random_before = global_rng()
    if source is not None:
        restore_rng(source["global_rng_before_reset"])
        random_before = global_rng()

    with patch.object(RemoteController, "create_game", capture_seed):
        if source is None:
            env.reset()
        else:
            # Consume and audit the seeded distribution draws, then explicitly
            # inject the saved config through this one test instance.
            generated = {}
            for distribution in env.wrapped.env_key_to_distribution_map.values():
                generated.update(distribution.generate())
            saved = restore_arrays(source["episode_config"])
            with patch.object(env.wrapped, "reset", side_effect=lambda: raw.reset(copy.deepcopy(saved))):
                env.reset()
        random_after = global_rng()
        position_after = distribution_rng(positions)
        episode_config = portable(raw.episode_config)
        pid = raw._sc2_proc.pid
        version = raw.version
        settings = dict(stochastic_attack=bool(raw.stochastic_attack),
                        stochastic_health=bool(raw.stochastic_health),
                        prob_obs_enemy=float(raw.prob_obs_enemy))
        if source is not None:
            restore_rng(source["global_rng_after_reset"])
        yield dict(
            sc2_seed=sc2_seed, env_seed=env_seed, create_game_seeds=launches, sc2_pid=pid,
            sc2_version=portable(version), settings=settings, episode_config=episode_config,
            position_rng_before=position_before, position_rng_after=position_after,
            global_rng_before_reset=random_before, global_rng_after_reset=random_after,
            generated_episode_config_matches=(portable(generated) == source["episode_config"]
                                              if source is not None else None))


def new_seeded_env(env_args, task, env_seed):
    from hetero_environment import ExactProtoss
    random.seed(env_seed)
    np.random.seed(env_seed)
    th.manual_seed(env_seed)
    env = ExactProtoss(**dict(copy.deepcopy(env_args), seed=env_seed))
    env.set_task(task)
    return env


def play(env, *, choose=None, source=None, sc2_seed, env_seed):
    raw = env.wrapped.env
    with initialized_episode(env, source=source, sc2_seed=sc2_seed, env_seed=env_seed) as initial:
        frames = []
        done = False
        for step in range(env.episode_limit if choose is not None else len(source["steps"])):
            current = trace_frame(env)
            actions = choose(current, step) if choose is not None else source["steps"][step]["actions"]
            if any(not mask[a] for mask, a in zip(current["avail_actions"], actions)):
                frames.append(dict(current, actions=actions))
                break
            reward, done, info = env.step(actions)
            frames.append(dict(current, actions=actions, reward=float(reward),
                               terminated=bool(done), battle_won=bool(info.get("battle_won", False)),
                               info=portable(info)))
            if done:
                break
        final = trace_frame(env)
    return dict(initial, force_restarts=raw.force_restarts, steps=frames, final=final,
                terminated=bool(done), battle_won=bool(env.last_info.get("battle_won", False)),
                global_rng_after_episode=global_rng())


def trace_comparator():
    first = None
    errors = {}

    def check(a, b, step, field, path=""):
        nonlocal first
        if (type(a) is float or type(b) is float) and type(a) in (int, float) and type(b) in (int, float):
            delta = abs(a - b)
            errors[field] = max(errors.get(field, 0.), delta)
            if math.isclose(a, b, abs_tol=ATOL, rel_tol=0):
                return
        elif isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for key in a:
                check(a[key], b[key], step, field, f"{path}/{key}")
            return
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (u, v) in enumerate(zip(a, b)):
                check(u, v, step, field, f"{path}/{i}")
            return
        elif type(a) is type(b) and a == b:
            return
        if first is None:
            first = dict(step=step, field=field, location=path)

    def result():
        if first is not None:
            return dict(first, field_max_abs_error=errors.get(first["field"]))
        return None

    return check, result, errors


def differences(reference, actual):
    check, result, errors = trace_comparator()
    checks = {}
    for field in ("episode_config", "settings", "sc2_version", "create_game_seeds",
                  "position_rng_before", "position_rng_after", "global_rng_before_reset",
                  "global_rng_after_reset", "global_rng_after_episode"):
        checks[field] = reference[field] == actual[field]
    checks["same_generated_episode_config"] = actual["generated_episode_config_matches"]
    checks["fresh_game_without_forced_restart"] = (
        actual["create_game_seeds"] == [reference["sc2_seed"]] and actual["force_restarts"] == 0)
    for step, (a, b) in enumerate(zip(reference["steps"], actual["steps"])):
        for field in a:
            check(a[field], b.get(field), step, field)
    if len(reference["steps"]) != len(actual["steps"]):
        check(len(reference["steps"]), len(actual["steps"]),
              min(len(reference["steps"]), len(actual["steps"])), "length")
    check(reference["terminated"], actual["terminated"], len(actual["steps"]), "terminated")
    check(reference["battle_won"], actual["battle_won"], len(actual["steps"]), "battle_won")
    if len(reference["steps"]) == len(actual["steps"]):
        for field, value in reference["final"].items():
            check(value, actual["final"][field], len(actual["steps"]), field)
    first = result()
    if first is None:
        for field, ok in checks.items():
            if not ok:
                first = dict(step=None, field=field, location="")
                break
    if first is not None:
        first["field_max_abs_error"] = errors.get(first["field"])
    return dict(passed=first is None, first_difference=first, max_abs_error=errors,
                steps=len(actual["steps"]), sc2_pid=actual["sc2_pid"], checks=checks,
                create_game_seeds=actual["create_game_seeds"])


def store_trace(path, record):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(record, stream, allow_nan=False)


def run(options):
    run_dir = options.run.resolve()
    checkpoint = options.checkpoint.resolve()
    out = options.output.resolve()
    config = read(run_dir / "config.json")
    config.update(device="cpu", use_cuda=False)
    os.environ["MARL_PROTOCOL"] = str(run_dir / "protocol.json")
    sys.path[:0] = [str(ROOT), str(ROOT / "drivers/continual")]
    from action_set_train import base, augmented_scheme
    from common.runtime import episode_spec
    from components.episode_buffer import EpisodeBatch

    out.mkdir(parents=True, exist_ok=False)
    th.set_num_threads(1)
    sc2_seed, env_seed = options.sc2_seed, options.env_seed
    info = read(run_dir / "env_info.json")
    spec = episode_spec(info)
    scheme, groups, preprocess = spec
    mac = base.controllers[config["mac"]](augmented_scheme(spec), groups, SimpleNamespace(**config))
    mac.agent.load_state_dict(th.load(checkpoint, map_location="cpu"))
    mac.agent.eval().requires_grad_(False)
    batch = EpisodeBatch(scheme, groups, 1, info["episode_limit"] + 1, preprocess=preprocess)

    def choose(frame, step):
        batch.update({key: np.asarray(frame[key])[None, ...]
                      for key in ("state", "obs", "avail_actions")}, ts=step)
        legal = batch["avail_actions"][:, step].bool()
        actions = mac.forward(batch, step, test_mode=True).masked_fill(~legal, -th.inf).argmax(dim=-1)
        batch.update({"actions": actions.unsqueeze(-1)}, ts=step)
        return actions[0].tolist()

    metadata = dict(engine_inputs=config["env_args"], task=options.task, env_seed=env_seed,
                    sc2_seed=sc2_seed, torch_seed=env_seed,
                    checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    script_sha256=sha(Path(__file__)), argv=shlex.join([sys.executable, *sys.argv]))
    write(out / "metadata.json", metadata)
    env = new_seeded_env(config["env_args"], options.task, env_seed)
    try:
        mac.init_hidden(1)
        with th.inference_mode():
            source = play(env, choose=choose, sc2_seed=sc2_seed, env_seed=env_seed)
    finally:
        env.close()
    source_path = out / "original.json.gz"
    store_trace(source_path, source)
    if not (source["terminated"] and source["battle_won"] and source["force_restarts"] == 0
            and source["create_game_seeds"] == [sc2_seed]):
        raise RuntimeError("The fixed-seed source episode is not a clean win; recorded the episode")
    with gzip.open(source_path, "rt", encoding="utf-8") as stream:
        source = json.load(stream)
    results = []
    pids = {source["sc2_pid"]}
    for index in range(1, REPLAYS + 1):
        env = new_seeded_env(config["env_args"], options.task, env_seed)
        try:
            replay = play(env, source=source, sc2_seed=sc2_seed, env_seed=env_seed)
            store_trace(out / f"replay_{index}.json.gz", replay)
            result = differences(source, replay)
            distinct = replay["sc2_pid"] is not None and replay["sc2_pid"] not in pids
            pids.add(replay["sc2_pid"])
            result["checks"]["distinct_process"] = distinct
            if not distinct and result["passed"]:
                result.update(passed=False, first_difference=dict(step=None, field="distinct_process",
                                                                   location="", field_max_abs_error=None))
        except Exception as exc:
            result = dict(passed=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            env.close()
        results.append(dict(replay=index, **result))
        print(f"Replay {index}: {'pass' if result['passed'] else 'fail'}", flush=True)
    passed = sum(r["passed"] for r in results)
    write(out / "summary.json", dict(passed=passed, failed=REPLAYS - passed, atol=ATOL, rtol=0.,
                                     original_steps=len(source["steps"]), original_sc2_pid=source["sc2_pid"],
                                     original_sha256=sha(source_path), settings=source["settings"], replays=results))
    return 0 if passed == REPLAYS else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="stalker")
    parser.add_argument("--env-seed", type=int, default=91101)
    parser.add_argument("--sc2-seed", type=int, default=91101)
    sys.exit(run(parser.parse_args()))
