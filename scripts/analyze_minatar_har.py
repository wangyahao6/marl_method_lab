"""Offline historical action retention on the first three MinAtar checkpoints.

The shared states come from both methods' independent greedy evaluation episodes.
No training replay or TKPR preference memory is read.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from single_agent.dqn import QNetwork
from single_agent.minatar_env import GAMES, make_env


FIRST_GAMES = ("SpaceInvaders", "Freeway", "Breakout")
METHODS = ("DQN-FT", "DQN-TKPR-K1")
TRAINING_COMMIT = "80d7ad68abf7b9069e54721d668a233a9de0f7f6"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    schedule = json.loads((run_dir / "task_schedule.json").read_text())["games"]
    commit = (run_dir / "git_revision.txt").read_text().strip()
    if config["method"] not in METHODS or config["seed"] != 3535:
        raise ValueError("Expected the seed-3535 FT or TKPR-K1 run")
    if schedule[:3] != list(FIRST_GAMES) or commit != TRAINING_COMMIT:
        raise ValueError("Training schedule or source commit differs from the published run")
    evaluations = jsonl(run_dir / "evaluations.jsonl")
    models, checkpoints = {}, {}
    for stage, game in enumerate(FIRST_GAMES):
        path = run_dir / f"stage_{stage}_checkpoint.pt"
        checkpoint = torch.load(path, map_location="cpu")
        if (checkpoint["stage"], checkpoint["game"], checkpoint["environment_steps"]) != (
                stage, game, (stage + 1) * config["interval_steps"]):
            raise ValueError(f"Checkpoint identity does not match stage {stage}: {path}")
        model = QNetwork().to(device)
        model.load_state_dict(checkpoint["online"])
        model.eval()
        models[stage] = model
        checkpoints[stage] = dict(path=str(path), sha256=sha256(path), game=game,
                                  environment_steps=checkpoint["environment_steps"])
    return dict(run_dir=run_dir, config=config, evaluations=evaluations,
                models=models, checkpoints=checkpoints, commit=commit)


def collect_greedy_states(model, game, episodes, seed_start, device):
    states, episode_indices, episode_seeds, episode_steps = [], [], [], []
    returns = []
    env = make_env(game)
    try:
        for episode in range(episodes):
            seed = seed_start + episode
            observation, _ = env.reset(seed=seed)
            total, step = 0.0, 0
            while True:
                states.append(np.asarray(observation, dtype=np.uint8).copy())
                episode_indices.append(episode)
                episode_seeds.append(seed)
                episode_steps.append(step)
                with torch.inference_mode():
                    q = model(torch.as_tensor(observation, dtype=torch.float32,
                                              device=device).unsqueeze(0))
                    action = int(q.argmax(1).item())
                observation, reward, terminated, truncated, _ = env.step(action)
                total += reward
                step += 1
                if terminated or truncated:
                    returns.append(total)
                    break
    finally:
        env.close()
    return dict(states=np.stack(states), episode_index=np.asarray(episode_indices, dtype=np.int16),
                episode_seed=np.asarray(episode_seeds, dtype=np.int32),
                episode_step=np.asarray(episode_steps, dtype=np.int16),
                returns=np.asarray(returns, dtype=np.float64))


def predict_q(model, states, device, batch_size=512):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            batch = torch.as_tensor(states[start:start + batch_size],
                                    dtype=torch.float32, device=device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs)


def action_metrics(old_q, current_q):
    """Compare greedy actions, valuing both choices with the old online Q network."""
    old_actions = old_q.argmax(axis=1)
    current_actions = current_q.argmax(axis=1)
    indices = np.arange(len(old_q))
    old_value = old_q[indices, old_actions]
    current_action_old_value = old_q[indices, current_actions]
    regret = old_value - current_action_old_value
    if np.any(regret < -1e-6):
        raise ValueError("Old-Q regret must be nonnegative")
    retained = old_actions == current_actions
    return dict(old_actions=old_actions.astype(np.uint8),
                current_actions=current_actions.astype(np.uint8),
                old_q_reference_action=old_value.astype(np.float32),
                old_q_current_action=current_action_old_value.astype(np.float32),
                old_q_regret=regret.astype(np.float32), retained=retained)


def summarize_metrics(method, game, reference_stage, current_stage, values, state_file):
    regret = values["old_q_regret"].astype(np.float64)
    retained = values["retained"]
    percentiles = np.percentile(regret, [0, 25, 50, 75, 90, 95, 99, 100])
    return dict(method=method, game=game, reference_stage=reference_stage,
                current_stage=current_stage, reference_t_env=(reference_stage + 1) * 500000,
                current_t_env=(current_stage + 1) * 500000, states=len(regret),
                har=float(retained.mean()), adr=float(1 - retained.mean()),
                mean_old_q_regret=float(regret.mean()), median_old_q_regret=float(np.median(regret)),
                regret_p0=float(percentiles[0]), regret_p25=float(percentiles[1]),
                regret_p75=float(percentiles[3]), regret_p90=float(percentiles[4]),
                regret_p95=float(percentiles[5]), regret_p99=float(percentiles[6]),
                regret_max=float(percentiles[7]), zero_regret_ratio=float(np.mean(regret == 0)),
                state_file=state_file)


def analyze(run_dirs, output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "states").mkdir()
    (output / "pointwise").mkdir()
    runs = {method: load_run(path, device) for method, path in run_dirs.items()}
    configs = [runs[method]["config"] for method in METHODS]
    if any(config["evaluation_episodes"] != configs[0]["evaluation_episodes"] or
           config["evaluation_seed_base"] != configs[0]["evaluation_seed_base"] or
           config["interval_steps"] != 500000 for config in configs):
        raise ValueError("Evaluation or stage protocol differs between methods")
    metrics, performance, state_manifest = [], [], []
    for method in METHODS:
        run = runs[method]
        for stage in range(3):
            for game in FIRST_GAMES[:stage + 1]:
                matches = [row for row in run["evaluations"] if row["kind"] == "stage_end"
                           and row["stage"] == stage and row["game"] == game]
                if len(matches) != 1:
                    raise ValueError(f"Missing stage-end evaluation: {method}, {stage}, {game}")
                row = matches[0]
                performance.append(dict(method=method, stage=stage, trained_game=FIRST_GAMES[stage],
                                        game=game, t_env=(stage + 1) * 500000,
                                        episodes=row["episodes"], seed_start=row["seed_start"],
                                        mean_return=row["mean_return"],
                                        std_return=row["std_return"]))
    for reference_stage, game in enumerate(FIRST_GAMES):
        seed_start = configs[0]["evaluation_seed_base"] + GAMES.index(game) * 100000
        episodes = configs[0]["evaluation_episodes"]
        chunks, sources = [], []
        for source_index, method in enumerate(METHODS):
            chunk = collect_greedy_states(runs[method]["models"][reference_stage],
                                           game, episodes, seed_start, device)
            matches = [row for row in runs[method]["evaluations"] if row["kind"] == "stage_end"
                       and row["stage"] == reference_stage and row["game"] == game]
            if len(matches) != 1 or not np.isclose(chunk["returns"].mean(),
                                                    matches[0]["mean_return"], atol=1e-9):
                raise ValueError(f"Collected greedy returns disagree with saved evaluation: {method}, {game}")
            chunks.append(chunk)
            sources.append(np.full(len(chunk["states"]), source_index, dtype=np.uint8))
        combined = {key: np.concatenate([chunk[key] for chunk in chunks])
                    for key in ("states", "episode_index", "episode_seed", "episode_step")}
        combined["source_method"] = np.concatenate(sources)
        state_file = output / "states" / f"{game.lower()}.npz"
        np.savez_compressed(state_file, **combined)
        unique = len(np.unique(np.packbits(combined["states"].reshape(len(combined["states"]), -1),
                                            axis=1), axis=0))
        state_manifest.append(dict(game=game, reference_stage=reference_stage,
                                   path=str(state_file.relative_to(output)), sha256=sha256(state_file),
                                   states=len(combined["states"]), unique_states=unique,
                                   source_states={method: len(chunk["states"])
                                                  for method, chunk in zip(METHODS, chunks)},
                                   episodes_per_method=episodes, seed_start=seed_start))
        if reference_stage == 2:
            continue  # Breakout has no subsequent stage in this analysis.
        for method in METHODS:
            old_q = predict_q(runs[method]["models"][reference_stage], combined["states"], device)
            for current_stage in range(reference_stage, 3):
                current_q = (old_q if current_stage == reference_stage else
                             predict_q(runs[method]["models"][current_stage],
                                       combined["states"], device))
                values = action_metrics(old_q, current_q)
                pointwise_file = ""
                if current_stage > reference_stage:
                    pointwise_file = f"pointwise/{method.lower().replace('-', '_')}_{game.lower()}_stage{current_stage}.npz"
                    np.savez_compressed(output / pointwise_file,
                                        **{key: value for key, value in values.items() if key != "retained"})
                metrics.append(summarize_metrics(method, game, reference_stage, current_stage,
                                                 values, str(state_file.relative_to(output))))
    metric_fields = list(metrics[0])
    save_csv(output / "action_retention.csv", metrics, metric_fields)
    save_csv(output / "historical_performance.csv", performance, list(performance[0]))
    by_key = {(row["method"], row["game"], row["current_stage"]): row for row in metrics}
    contrasts = []
    for game, first_stage in (("SpaceInvaders", 0), ("Freeway", 1)):
        for stage in range(first_stage + 1, 3):
            ft = by_key[("DQN-FT", game, stage)]
            tkpr = by_key[("DQN-TKPR-K1", game, stage)]
            performance_by_key = {(row["method"], row["game"], row["stage"]): row
                                  for row in performance}
            contrasts.append(dict(game=game, reference_stage=first_stage, current_stage=stage,
                                  states=ft["states"], ft_har=ft["har"], tkpr_har=tkpr["har"],
                                  tkpr_minus_ft_har=tkpr["har"] - ft["har"],
                                  ft_mean_old_q_regret=ft["mean_old_q_regret"],
                                  tkpr_mean_old_q_regret=tkpr["mean_old_q_regret"],
                                  ft_mean_return=performance_by_key[("DQN-FT", game, stage)]["mean_return"],
                                  tkpr_mean_return=performance_by_key[("DQN-TKPR-K1", game, stage)]["mean_return"]))
    save_csv(output / "method_comparison.csv", contrasts, list(contrasts[0]))
    manifest = dict(training_commit=TRAINING_COMMIT, training_seed=3535,
                    stages=[dict(stage=i, game=game, t_env=(i + 1) * 500000)
                            for i, game in enumerate(FIRST_GAMES)],
                    action_space=6, observation_shape=[7, 10, 10],
                    state_source="union of both methods' independent greedy stage-end evaluation episodes",
                    source_method_codes={str(index): method for index, method in enumerate(METHODS)},
                    evaluation_episodes_per_method=episodes,
                    runs={method: dict(run_dir=str(runs[method]["run_dir"]),
                                       checkpoints=runs[method]["checkpoints"])
                          for method in METHODS}, states=state_manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return metrics, performance, contrasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft-run", type=Path, required=True)
    parser.add_argument("--tkpr-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    _, _, contrasts = analyze({"DQN-FT": args.ft_run,
                               "DQN-TKPR-K1": args.tkpr_run}, args.output,
                              torch.device(args.device))
    print(json.dumps(contrasts, indent=2))


if __name__ == "__main__":
    main()
