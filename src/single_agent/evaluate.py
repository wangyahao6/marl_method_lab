"""Independent greedy evaluation of each task-end checkpoint on all goals."""

import argparse
import json
from pathlib import Path

import torch

from single_agent.dqn import QNetwork
from single_agent.mountaincar import GOALS, GoalMountainCar


def evaluate(run_dir, episodes=100):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    torch.set_num_threads(1)
    rows = []
    for stage in GOALS:
        checkpoint = torch.load(run_dir / f"{stage}_checkpoint.pt", map_location="cpu")
        model = QNetwork()
        model.load_state_dict(checkpoint["online"])
        model.eval().requires_grad_(False)
        for task_index, task in enumerate(GOALS):
            env = GoalMountainCar(task)
            successes = 0
            lengths = []
            try:
                for episode in range(episodes):
                    observation, _ = env.reset(seed=91000 + task_index * 1000 + episode)
                    length = 0
                    while True:
                        with torch.no_grad():
                            action = int(model(torch.as_tensor(observation)).argmax().item())
                        observation, _, terminated, truncated, _ = env.step(action)
                        length += 1
                        if terminated or truncated:
                            successes += int(terminated)
                            lengths.append(length)
                            break
            finally:
                env.close()
            rows.append(dict(method=config["method"], seed=config["seed"], stage=stage,
                             evaluated_task=task, episodes=episodes, successes=successes,
                             success_rate=successes / episodes,
                             mean_return=-sum(lengths) / episodes,
                             mean_episode_length=sum(lengths) / episodes))
            (run_dir / "evaluations.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run_dir, args.episodes), indent=2))


if __name__ == "__main__":
    main()
