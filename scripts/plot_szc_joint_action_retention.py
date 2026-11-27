"""Plot episode-mean IAR and JAA for the fixed S trajectories."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(data_file, output):
    with Path(data_file).open() as stream:
        rows = list(csv.DictReader(stream))
    methods = (("VDN-FT", "#36688D"), ("TKPR-K1", "#D1493F"))
    measures = (("iar", "Individual action retention (IAR)"),
                ("jaa", "Joint action agreement (JAA)"))
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.3), sharex=True, sharey=True)
    for row_idx, seed in enumerate((2222, 3535)):
        for col_idx, (key, label) in enumerate(measures):
            ax = axes[row_idx, col_idx]
            for method, color in methods:
                selected = sorted((r for r in rows if int(r["seed"]) == seed and
                                   r["method"] == method), key=lambda r: int(r["phase"]))
                ax.plot([int(r["phase"]) for r in selected],
                        [float(r[key]) for r in selected], "o-", color=color,
                        linewidth=2, label=method)
            ax.set_title(f"Seed {seed} · {label}")
            ax.set_xticks([0, 1, 2], ["S", "Z", "C"])
            ax.set_ylim(-0.04, 1.04)
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel("Episode-mean retention")
            if row_idx == 1:
                ax.set_xlabel("Model after training stage")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Historical S decisions on fixed shared trajectories", y=1.07)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    if Path(output).suffix.lower() == ".svg":
        path = Path(output)
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.data, args.output)


if __name__ == "__main__":
    main()
