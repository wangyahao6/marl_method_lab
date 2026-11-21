"""Run T1→T2→T3 continual MountainCar training.

Example: PYTHONPATH=src python -m single_agent.train --method DQN-TKPR-K1
"""

import argparse
from collections import deque
import copy
from datetime import datetime
import json
from pathlib import Path
import random
import subprocess
import tarfile

import numpy as np
import torch

from single_agent.dqn import DQNLearner
from single_agent.memory import PreferenceMemory
from single_agent.mountaincar import GOALS, GoalMountainCar


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/single_agent/mountaincar_tkpr.json"
REPO_ROOT = Path(__file__).resolve().parents[2]


def snapshot_source(output):
    files = [*sorted((REPO_ROOT / "src/single_agent").glob("*.py")),
             REPO_ROOT / "src/learners/ranking_preference.py",
             REPO_ROOT / "src/learners/action_set_preference.py",
             REPO_ROOT / "configs/single_agent/mountaincar_tkpr.json"]
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(REPO_ROOT))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              check=True, capture_output=True, text=True).stdout.strip()
    (output / "git_revision.txt").write_text(revision + "\n")


def collect_successes(task, model, episodes, seed):
    """Collect only successful episodes under the frozen task-end policy."""
    frozen = copy.deepcopy(model).eval().requires_grad_(False)
    env = GoalMountainCar(task)
    successes = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            trajectory = []
            while True:
                trajectory.append(observation.copy())
                with torch.no_grad():
                    action = int(frozen(torch.as_tensor(observation)).argmax().item())
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    if terminated:
                        successes.append(trajectory)
                    break
    finally:
        env.close()
    return successes


def run(config, method, seed, output, steps_per_task=None):
    if method not in ("DQN-FT", "DQN-TKPR-K1", "DQN-TKPR-K1-W", "DQN-TKPR-K2"):
        raise ValueError(method)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    learner = DQNLearner(config, method, seed)
    memory = PreferenceMemory()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot_source(output)
    (output / "config.json").write_text(json.dumps(dict(config, method=method, seed=seed,
        steps_per_task=steps_per_task or config["steps_per_task"]), indent=2) + "\n")
    report = []
    progress_path = output / "progress.jsonl"
    episodes_path = output / "episodes.jsonl"
    for task_index, task in enumerate(GOALS):
        if task_index:
            learner.switch_task()
        env = GoalMountainCar(task)
        budget = steps_per_task or config["steps_per_task"]
        completed = successes = preference_updates = 0
        episode_length = 0
        recent_successes = deque(maxlen=100)
        latest_update = None
        observation, _ = env.reset(seed=seed + task_index * 10000)
        try:
            for step in range(budget):
                action = learner.act(observation)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                learner.observe(observation, action, reward, next_observation, terminated, truncated)
                result = learner.update(memory)
                if result is not None:
                    latest_update = result
                if result and result["preference_loss"] != 0:
                    preference_updates += 1
                episode_length += 1
                observation = next_observation
                if terminated or truncated:
                    completed += 1
                    successes += int(terminated)
                    recent_successes.append(int(terminated))
                    with episodes_path.open("a") as stream:
                        stream.write(json.dumps(dict(task=task, method=method, task_step=step + 1,
                            episode=completed, length=episode_length, return_=-episode_length,
                            success=bool(terminated))) + "\n")
                    episode_length = 0
                    observation, _ = env.reset(seed=seed + task_index * 10000 + completed)
                if (step + 1) % config.get("progress_interval", 10000) == 0 or step + 1 == budget:
                    with progress_path.open("a") as stream:
                        stream.write(json.dumps(dict(task=task, method=method, task_step=step + 1,
                            environment_steps=learner.steps, episodes=completed, successes=successes,
                            recent_100_success_rate=(sum(recent_successes) / len(recent_successes)
                                                     if recent_successes else None),
                            preference_updates=preference_updates, latest_update=latest_update)) + "\n")
        finally:
            env.close()
        torch.save({"online": learner.online.state_dict(), "target": learner.target.state_dict(),
                    "optimizer": learner.optimizer.state_dict(), "environment_steps": learner.steps},
                   output / f"{task}_checkpoint.pt")
        trajectories = collect_successes(task, learner.online,
                         config["reference_collection_episodes"], seed + task_index * 10000 + 5000)
        recorded = memory.add_successful_trajectories(task, trajectories, learner.online)
        torch.save(memory.tasks[task], output / f"{task}_preference_memory.pt")
        report.append(dict(task=task, steps=budget, episodes=completed, training_successes=successes,
                           reference_successes=len(trajectories), reference_states=recorded,
                           preference_updates=preference_updates))
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--method", choices=["DQN-FT", "DQN-TKPR-K1", "DQN-TKPR-K1-W", "DQN-TKPR-K2"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps-per-task", type=int, help="Short smoke runs only; default is 300000")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output or Path("runs") / f"mountaincar_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    print(json.dumps(run(config, args.method, args.seed, output, args.steps_per_task), indent=2))


if __name__ == "__main__":
    main()
