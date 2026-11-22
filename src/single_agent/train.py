"""Seven-interval continual MinAtar CNN-DQN and TKPR training."""

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import random
import subprocess
import tarfile

import numpy as np
import torch

from single_agent.dqn import DQNLearner
from single_agent.evaluate import evaluate_stage, retention_metrics
from single_agent.memory import PreferenceMemory, collect_reference
from single_agent.minatar_env import GAMES, make_env


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/single_agent/minatar_tkpr.json"


def make_schedule(seed, intervals=7):
    rng = random.Random(seed)
    first = list(GAMES)
    rng.shuffle(first)
    return first + [rng.choice(GAMES) for _ in range(intervals - len(GAMES))]


def snapshot_source(output):
    files = [*sorted((ROOT / "src/single_agent").glob("*.py")),
             ROOT / "src/learners/ranking_preference.py",
             ROOT / "src/learners/action_set_preference.py",
             ROOT / "configs/single_agent/minatar_tkpr.json"]
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              check=True, capture_output=True, text=True).stdout.strip()
    (output / "git_revision.txt").write_text(revision + "\n")


def append_json(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def run(config, method, seed, output, *, schedule=None, device=None):
    config = dict(config)
    schedule = list(schedule) if schedule is not None else make_schedule(seed, config["intervals"])
    if not schedule or any(game not in GAMES for game in schedule):
        raise ValueError("Schedule must contain MinAtar games")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    learner = DQNLearner(config, method, seed, device)
    memory = PreferenceMemory()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot_source(output)
    (output / "config.json").write_text(json.dumps(dict(config, method=method, seed=seed), indent=2) + "\n")
    (output / "task_schedule.json").write_text(json.dumps(dict(seed=seed, games=schedule), indent=2) + "\n")
    evaluations, summary = [], []
    for stage, game in enumerate(schedule):
        changed = learner.switch_game(game)
        env = make_env(game)
        observation, _ = env.reset(seed=seed + stage * 100000)
        completed, episode_return, episode_length, preference_updates = 0, 0.0, 0, 0
        recent_returns = deque(maxlen=100)
        latest_update = None
        try:
            for task_step in range(1, config["interval_steps"] + 1):
                action = learner.act(observation)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                learner.observe(observation, action, reward, next_observation, terminated, truncated)
                metrics = learner.update(memory,
                    diagnostics=learner.steps % config["diagnostics_interval"] == 0)
                if metrics is not None:
                    latest_update = metrics
                    if metrics["preference_loss"] != 0:
                        preference_updates += 1
                    if "constrained_states" in metrics:
                        append_json(output / "updates.jsonl", dict(stage=stage, game=game,
                            task_step=task_step, environment_steps=learner.steps, **metrics))
                episode_return += reward
                episode_length += 1
                observation = next_observation
                if terminated or truncated:
                    completed += 1
                    recent_returns.append(episode_return)
                    append_json(output / "episodes.jsonl", dict(stage=stage, game=game,
                        task_step=task_step, episode=completed, return_=episode_return,
                        length=episode_length, complete=True))
                    episode_return, episode_length = 0.0, 0
                    observation, _ = env.reset()
                if task_step % config["progress_interval"] == 0 or task_step == config["interval_steps"]:
                    append_json(output / "progress.jsonl", dict(stage=stage, game=game,
                        task_step=task_step, environment_steps=learner.steps,
                        completed_episodes=completed,
                        recent_100_mean_return=(float(np.mean(recent_returns)) if recent_returns else None),
                        preference_updates=preference_updates,
                        latest_td_loss=latest_update["td_loss"] if latest_update else None,
                        latest_preference_loss=latest_update["preference_loss"] if latest_update else None))
            if episode_length:
                append_json(output / "episodes.jsonl", dict(stage=stage, game=game,
                    task_step=config["interval_steps"], episode=completed + 1,
                    return_=episode_return, length=episode_length, complete=False))
        finally:
            env.close()
        torch.save(dict(online=learner.online.state_dict(), target=learner.target.state_dict(),
                        optimizer=learner.optimizer.state_dict(), environment_steps=learner.steps,
                        stage=stage, game=game), output / f"stage_{stage}_checkpoint.pt")
        stage_rows = evaluate_stage(learner.online, stage, episodes=config["evaluation_episodes"],
                                    seed_base=config["evaluation_seed_base"])
        evaluations.extend(stage_rows)
        for row in stage_rows:
            append_json(output / "evaluations.jsonl", row)
        if stage == len(schedule) - 1:
            final_rows = evaluate_stage(learner.online, stage,
                episodes=config["final_evaluation_episodes"],
                seed_base=config["final_evaluation_seed_base"], kind="final")
            evaluations.extend(final_rows)
            for row in final_rows:
                append_json(output / "evaluations.jsonl", row)
        if method.startswith("DQN-TKPR") and game not in memory.tasks:
            collection_seed = seed + config["collection_seed_base"] + GAMES.index(game) * 100000
            reference = collect_reference(game, learner.online, config["reference_steps"],
                                          collection_seed, stage)
            memory.add_reference(**reference)
            memory.save(game, output / f"memory_{game}.pt")
        summary.append(dict(stage=stage, game=game, switched_game=changed,
                            environment_steps=learner.steps, trained_steps=config["interval_steps"],
                            completed_episodes=completed, preference_updates=preference_updates,
                            memory_states={name: len(record["states"])
                                           for name, record in memory.tasks.items()},
                            memory_unique_states={name: record["unique_state_count"]
                                                  for name, record in memory.tasks.items()},
                            memory_unique_ratios={name: record["unique_state_ratio"]
                                                  for name, record in memory.tasks.items()}))
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    retention = retention_metrics(schedule, evaluations)
    (output / "retention.json").write_text(json.dumps(retention, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--method", choices=DQNLearner.METHODS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output or Path("runs") / f"minatar_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    print(json.dumps(run(config, args.method, args.seed, output, device=args.device), indent=2))


if __name__ == "__main__":
    main()
