"""Independent greedy MinAtar evaluation and per-game retention metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from single_agent.dqn import QNetwork
from single_agent.minatar_env import GAMES, make_env


def evaluate_model(model, game, episodes, seed_start):
    model.eval()
    device = next(model.parameters()).device
    env = make_env(game)
    returns, lengths = [], []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed_start + episode)
            total, length = 0.0, 0
            while True:
                with torch.no_grad():
                    q = model(torch.as_tensor(observation, dtype=torch.float32,
                                               device=device).unsqueeze(0))
                observation, reward, terminated, truncated, _ = env.step(int(q.argmax(1).item()))
                total += reward
                length += 1
                if terminated or truncated:
                    returns.append(total)
                    lengths.append(length)
                    break
    finally:
        env.close()
    return dict(episodes=episodes, seed_start=seed_start,
                mean_return=float(np.mean(returns)),
                std_return=float(np.std(returns)),
                median_return=float(np.median(returns)),
                mean_episode_length=float(np.mean(lengths)))


def evaluate_stage(model, stage, games=GAMES, episodes=20, seed_base=91000, kind="stage_end"):
    return [dict(stage=stage, game=game, kind=kind,
                 **evaluate_model(model, game, episodes, seed_base + index * 100000))
            for index, game in enumerate(games)]


def retention_metrics(schedule, rows):
    final_stage = len(schedule) - 1
    metrics = []
    for game in GAMES:
        first_stage = schedule.index(game)
        first = next(row for row in rows if row["stage"] == first_stage and
                     row["game"] == game and row["kind"] == "stage_end")
        final = next(row for row in rows if row["stage"] == final_stage and
                     row["game"] == game and row["kind"] == "final")
        bwt = final["mean_return"] - first["mean_return"]
        metrics.append(dict(game=game, first_learned_stage=first_stage,
                            first_mean_return=first["mean_return"],
                            final_mean_return=final["mean_return"],
                            bwt_return=bwt, forgetting_return=max(0.0, -bwt)))
    return metrics


def evaluate_checkpoints(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    schedule = json.loads((run_dir / "task_schedule.json").read_text())["games"]
    rows = []
    for stage in range(len(schedule)):
        checkpoint = torch.load(run_dir / f"stage_{stage}_checkpoint.pt", map_location="cpu")
        model = QNetwork()
        model.load_state_dict(checkpoint["online"])
        rows.extend(evaluate_stage(model, stage, episodes=config["evaluation_episodes"],
                                   seed_base=config["evaluation_seed_base"]))
        if stage == len(schedule) - 1:
            rows.extend(evaluate_stage(model, stage,
                       episodes=config["final_evaluation_episodes"],
                       seed_base=config["final_evaluation_seed_base"], kind="final"))
    return rows, retention_metrics(schedule, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, retention = evaluate_checkpoints(args.run_dir)
    print(json.dumps(dict(evaluations=rows, retention=retention), indent=2))


if __name__ == "__main__":
    main()
