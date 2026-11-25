"""Plot the historical-action metrics produced by analyze_minatar_har.py."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(metrics_file, output):
    with Path(metrics_file).open() as stream:
        rows = list(csv.DictReader(stream))
    methods = (("DQN-FT", "#36688D"), ("DQN-TKPR-K1", "#D1493F"))
    games = (("SpaceInvaders", "Space Invaders"), ("Freeway", "Freeway"))
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.3), sharex="col")
    for col, (game, label) in enumerate(games):
        for method, color in methods:
            selected = sorted((row for row in rows if row["game"] == game and
                               row["method"] == method),
                              key=lambda row: int(row["current_stage"]))
            stages = [int(row["current_stage"]) for row in selected]
            axes[0, col].plot(stages, [float(row["har"]) for row in selected],
                              marker="o", color=color, label=method, linewidth=2)
            axes[1, col].plot(stages, [float(row["mean_old_q_regret"]) for row in selected],
                              marker="o", color=color, linewidth=2)
        axes[0, col].set_title(label)
        axes[0, col].set_ylim(-0.04, 1.04)
        axes[1, col].set_ylim(bottom=0)
        for row in range(2):
            stages = [0, 1, 2] if game == "SpaceInvaders" else [1, 2]
            axes[row, col].set_xticks(stages, [f"T{stage + 1}" for stage in stages])
            axes[row, col].grid(alpha=0.25)
            axes[row, col].spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Historical action retention (HAR)")
    axes[1, 0].set_ylabel("Mean Old-Q Regret")
    axes[1, 0].set_xlabel("Model after task")
    axes[1, 1].set_xlabel("Model after task")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("MinAtar historical actions on fixed shared states (seed 3535)", y=1.07)
    fig.text(0.5, 0.005, "Old-Q Regret uses the task's reference model; Q scales are game-specific.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    if Path(output).suffix.lower() == ".svg":
        path = Path(output)
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.metrics, args.output)


if __name__ == "__main__":
    main()
