"""Verify recurrent reconstruction and branch one legal action in a frozen-policy rollout."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch as th

from verify_smacv2_replay import (
    ROOT, differences, initialized_episode, new_seeded_env, play, portable, read, sha,
    store_trace, trace_comparator, trace_frame, write,
)

INPUTS = ("state", "avail_actions", "obs")
BOOT_KEYS = ("episode_config", "settings", "sc2_version", "position_rng_before",
             "position_rng_after", "global_rng_before_reset", "global_rng_after_reset")


def add_observation(batch, frame, step):
    batch.update({key: np.asarray(frame[key])[None, ...] for key in INPUTS}, ts=step)


def add_action(batch, actions, step):
    batch.update({"actions": th.tensor(actions, dtype=th.long).view(1, -1, 1)}, ts=step)


def open_source(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


class FrozenPolicy:
    def __init__(self, run, checkpoint, info):
        from action_set_train import augmented_scheme, base
        from common.runtime import episode_spec
        from components.episode_buffer import EpisodeBatch

        cfg = read(run / "config.json")
        if cfg["mac"] != "basic_mac" or cfg["agent_output_type"] != "q":
            raise ValueError("This experiment uses the archived BasicMAC Q policy")
        cfg.update(device="cpu", use_cuda=False)
        self.args = SimpleNamespace(**cfg)
        self.spec = episode_spec(info)
        self.scheme = augmented_scheme(self.spec)
        self.make_mac = base.controllers[cfg["mac"]]
        self.make_batch = EpisodeBatch
        self.weights = th.load(checkpoint, map_location="cpu")
        self.episode_limit = info["episode_limit"]

    def episode(self):
        scheme, groups, preprocess = self.spec
        mac = self.make_mac(self.scheme, groups, self.args)
        mac.agent.load_state_dict(self.weights)
        mac.agent.eval().requires_grad_(False)
        mac.init_hidden(1)
        batch = self.make_batch(scheme, groups, 1, self.episode_limit + 1, preprocess=preprocess)
        return mac, batch


def select(mac, batch, step):
    """Capture the actual online forward once, inside BasicMAC.select_actions."""
    before = mac.hidden_states.tolist()
    outputs = []
    forward = mac.forward

    def capture(*args, **kwargs):
        q = forward(*args, **kwargs)
        outputs.append(q.tolist())
        return q

    with patch.object(mac, "forward", side_effect=capture):
        actions = mac.select_actions(batch, t_ep=step, t_env=0, test_mode=True)[0].tolist()
    q, = outputs
    return dict(q=q, actions=actions, hidden_before=before, hidden_after=mac.hidden_states.tolist())


def offline_records(policy, source):
    mac, batch = policy.episode()
    records = []
    for step, frame in enumerate(source["steps"]):
        add_observation(batch, frame, step)
        before = mac.hidden_states.tolist()
        q = mac.forward(batch, step, test_mode=True)
        legal = batch["avail_actions"][:, step].bool()
        greedy = mac.action_selector.select_action(q, legal, 0, test_mode=True)[0].tolist()
        records.append(dict(q=q.tolist(), actions=greedy, hidden_before=before,
                            hidden_after=mac.hidden_states.tolist()))
        add_action(batch, frame["actions"], step)
    return records


def selected_sites(source, records):
    sites = []
    length = len(records)
    for name, step in zip(("early", "middle", "late"), (length // 6, length // 2, 5 * length // 6)):
        frame = source["steps"][step]
        agent, actions = next((i, [a for a, legal in enumerate(mask) if legal])
                              for i, mask in enumerate(frame["avail_actions"])
                              if sum(mask) > 1 and not mask[0])
        q = records[step]["q"][0][agent]
        greedy = records[step]["actions"][agent]
        sites.append(dict(name=name, timestep=step, agent_id=agent, legal_actions=actions,
                          state=frame["state"], obs=frame["obs"][agent], avail_actions=frame["avail_actions"],
                          unit_state=frame["unit_state"], unit_slots=frame["unit_slots"],
                          source_joint_action=frame["actions"], greedy_action=greedy, greedy_q=q[greedy],
                          candidates=[dict(action=a, q=q[a], q_gap=q[greedy] - q[a]) for a in actions]))
    return sites


def branch(policy, make_env, seed_args, source, records, site, action, used_pids):
    target, agent = site["timestep"], site["agent_id"]
    candidate, = (item for item in site["candidates"] if item["action"] == action)
    baseline = action == site["greedy_action"]
    result = dict(trajectory_id="szc-frpr-1111-s-91101-episode0", site=site["name"],
                  timestep=target, agent_id=agent, greedy_action=site["greedy_action"],
                  greedy_q=site["greedy_q"], **candidate, baseline=baseline,
                  label="Uncertain", environment_replay_consistent=None, rnn_consistent=None,
                  first_difference=None, won=None, termination_reason=None, rollout_steps=0)
    compare, diff, errors = trace_comparator()
    compare_rnn, rnn_diff, rnn_errors = trace_comparator()
    env = None
    try:
        mac, batch = policy.episode()
        env = make_env()
        with initialized_episode(env, source=source, **seed_args) as init:
            result["sc2_pid"] = init["sc2_pid"]
            result["create_game_seeds"] = init["create_game_seeds"]
            pid = init["sc2_pid"]
            fresh = pid is not None and pid not in used_pids
            used_pids.add(pid)
            for key in BOOT_KEYS:
                compare(source[key], init[key], None, key)
            compare(True, init["generated_episode_config_matches"], None, "seeded_config")
            compare([source["sc2_seed"]], init["create_game_seeds"], None, "create_game_seeds")
            compare(True, fresh, None, "distinct_sc2_process")
            if diff():
                result["environment_replay_consistent"] = False
                raise ValueError("Fresh environment initialization mismatch")
            rewards = []
            trace_actions = []
            done = False
            for step in range(policy.episode_limit):
                frame = trace_frame(env)
                if step <= target or (baseline and step < len(source["steps"])):
                    for key, value in frame.items():
                        compare(source["steps"][step][key], value, step, key)
                    if diff():
                        result["environment_replay_consistent"] = False if step <= target else True
                        raise ValueError("Recorded frame mismatch")
                add_observation(batch, frame, step)
                if step < target:
                    actions = source["steps"][step]["actions"]
                elif step == target:
                    record = records[step]
                    mac.hidden_states = th.tensor(record["hidden_before"], dtype=mac.hidden_states.dtype)
                    decision = select(mac, batch, step)
                    for key, value in decision.items():
                        compare_rnn(record[key], value, step, key)
                    result.update(rnn_consistent=rnn_diff() is None, environment_replay_consistent=True,
                                  target_joint_action_before=decision["actions"])
                    if rnn_diff():
                        raise ValueError("Restored MAC output mismatch")
                    actions = list(decision["actions"])
                    actions[agent] = action
                    result["target_joint_action_executed"] = actions
                else:
                    actions = mac.select_actions(batch, t_ep=step, t_env=0, test_mode=True)[0].tolist()
                if any(not mask[a] for mask, a in zip(frame["avail_actions"], actions)):
                    compare(True, False, step, "executed_action_legal")
                    raise ValueError("Illegal executed action")
                reward, done, info = env.step(actions)
                add_action(batch, actions, step)
                rewards.append(float(reward))
                trace_actions.append(actions)
                result["rollout_steps"] = len(rewards)
                if step < target or (baseline and step < len(source["steps"])):
                    expected = source["steps"][step]
                    for key, value in dict(actions=actions, reward=float(reward), terminated=bool(done),
                                           battle_won=bool(info.get("battle_won")), info=portable(info)).items():
                        compare(expected[key], value, step, key)
                    if diff():
                        result["environment_replay_consistent"] = False if step < target else True
                        raise ValueError("Recorded transition mismatch")
                if env.force_restarts:
                    compare(0, env.force_restarts, step, "force_restarts")
                    raise ValueError("SC2 force restart")
                if done:
                    break
            if baseline:
                compare(len(source["steps"]), len(rewards), min(len(rewards), len(source["steps"])), "length")
                for key, value in trace_frame(env).items():
                    compare(source["final"][key], value, len(rewards), key)
                if diff():
                    raise ValueError("Baseline did not reproduce the original full rollout")
            won = bool(info.get("battle_won"))
            timeout = bool(env.timeouts or info.get("episode_limit") or not done)
            result.update(won=won, label="Uncertain" if timeout else "Good" if won else "Bad",
                          termination_reason="timeout" if timeout else "victory" if won else "battle_loss",
                          final_info=portable(info), episode_return=sum(rewards), actions=trace_actions,
                          post_intervention_steps=len(rewards) - target - 1)
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", termination_reason="invalid_or_environment_error")
    finally:
        if env is not None:
            env.close()
    result["first_difference"] = diff() or rnn_diff()
    result["environment_max_abs_error"] = errors
    result["rnn_max_abs_error"] = rnn_errors
    return result


@th.inference_mode()
def run(args):
    incoming = args.source.resolve()
    source_path = incoming / "original.json.gz"
    meta = read(incoming / "metadata.json")
    prior = read(incoming / "summary.json")
    checkpoint = ROOT / meta["checkpoint"]
    if sha(checkpoint) != meta["checkpoint_sha256"] or sha(source_path) != prior["original_sha256"]:
        raise ValueError("Source trajectory or checkpoint differs from previous evidence")
    source = open_source(source_path)
    if len(source["steps"]) != 57 or not source["battle_won"]:
        raise ValueError("Expected the archived 57-step winning trajectory")
    checkpoint_run = checkpoint.parents[2]
    cfg = read(checkpoint_run / "config.json")
    info = read(checkpoint_run / "env_info.json")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "manifest.json", dict(source=str(incoming), source_sha256=sha(source_path),
          source_report_commit="1401aeb", checkpoint=meta["checkpoint"],
          checkpoint_sha256=meta["checkpoint_sha256"],
          source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
          script_sha256=sha(Path(__file__)), replay_helper_sha256=sha(Path(__file__).with_name("verify_smacv2_replay.py")),
          argv=shlex.join([sys.executable, *sys.argv]), task=meta["task"], env_seed=meta["env_seed"],
          sc2_seed=meta["sc2_seed"]))
    os.environ["MARL_PROTOCOL"] = str(checkpoint_run / "protocol.json")
    sys.path[:0] = [str(ROOT), str(ROOT / "drivers/continual")]
    th.set_num_threads(1)
    seed_args = {k: meta[k] for k in ("sc2_seed", "env_seed")}
    policy = FrozenPolicy(checkpoint_run, checkpoint, info)

    def make_env():
        return new_seeded_env(cfg["env_args"], meta["task"], meta["env_seed"])

    records = offline_records(policy, source)
    online = []
    mac, batch = policy.episode()

    def choose(frame, step):
        add_observation(batch, frame, step)
        record = select(mac, batch, step)
        online.append(record)
        add_action(batch, record["actions"], step)
        return record["actions"]

    env = make_env()
    try:
        trace = play(env, choose=choose, source=source, **seed_args)
    except Exception as exc:
        write(out / "rnn_summary.json", dict(passed=False, error=f"{type(exc).__name__}: {exc}"))
        return 1
    finally:
        env.close()
    check, diff, errors = trace_comparator()
    environment = differences(source, trace)
    check(len(records), len(online), min(len(records), len(online)), "length")
    for step, (a, b) in enumerate(zip(records, online)):
        for key in a:
            check(a[key], b[key], step, key)
        check(source["steps"][step]["actions"], a["actions"], step, "recorded_actions")
    rnn_ok = diff() is None
    store_trace(out / "rnn_trace.json.gz", dict(offline=records, online=online))
    write(out / "rnn_summary.json", dict(passed=rnn_ok and environment["passed"], environment=environment,
          rnn_first_difference=diff(), rnn_max_abs_error=errors, steps=len(records), online_win=trace["battle_won"]))
    if not rnn_ok or not environment["passed"] or not trace["battle_won"]:
        return 1

    sites = selected_sites(source, records)
    write(out / "sites.json", sites)
    used_pids = {source["sc2_pid"], trace["sc2_pid"]}
    runs = []
    for site in sites:
        result = branch(policy, make_env, seed_args, source, records, site, site["greedy_action"], used_pids)
        runs.append(result)
        write(out / "branches.json", runs)
        if result["label"] != "Good":
            return 1
    for site in sites:
        for item in site["candidates"]:
            if item["action"] == site["greedy_action"]:
                continue
            result = branch(policy, make_env, seed_args, source, records, site, item["action"], used_pids)
            runs.append(result)
            write(out / "branches.json", runs)
            print(site["name"], item["action"], result["label"], flush=True)
            if result["rnn_consistent"] is not True or result["environment_replay_consistent"] is not True:
                return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    sys.exit(run(parser.parse_args()))
