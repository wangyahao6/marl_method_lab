"""Verify complete rollout blocks and publish lightweight statistics and plots."""
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from run_analysis import BINS, METHODS, TASKS, read, sha, write


PANELS = (("stalker", "Z", "own_stage"), ("stalker", "C", "own_stage"),
          ("zealot", "C", "own_stage"), ("zealot", "C", "independent_z"))
TITLES = ("S after Z", "S after C", "Z after C (own Z)", "Z after C (independent Z)")
COLORS = dict(zip(METHODS, ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")))


def csv_out(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(hits, n):
    return 100 * hits / n if n else None


def summed(rows, count, hit):
    n, k = sum(count(r) for r in rows), sum(hit(r) for r in rows)
    return dict(valid_decisions=n, agreements=k, agr_pct=pct(k, n))


def panel_rows(summary, panel):
    return [r for r in summary if (r["task"], r["stage"], r["reference"]) == panel]


def plots(dest, summary):
    for scatter, filename in ((True, "win_rate_correlation.png"), (False, "agreement_by_method.png")):
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 8))
        for ax, panel, title in zip(axes.flat, PANELS, TITLES):
            rows = panel_rows(summary, panel)
            if not scatter:
                ax.set_xticks(range(len(METHODS)))
                ax.set_xticklabels(METHODS)
            for r in rows:
                agr = r["agr_pct"]
                primary = r["cohort"] == "k_ablation"
                col = COLORS[r["method"]]
                marker = {1111: "o", 2222: "^", 3535: "s"}.get(r["seed"], "D")
                if scatter:
                    x, y = agr, r["old_task_win_pct"]
                else:
                    x, y = METHODS.index(r["method"]), agr
                    x += {2222: -0.16, 3535: 0.16}.get(r["seed"], 0)
                ax.scatter(x, y, marker=marker, s=58, facecolors=col if primary else "none",
                           edgecolors=col, zorder=3)
                if scatter:
                    label = r["method"] if primary else str(r["seed"])
                    ax.annotate(label, (x, y), xytext=(5, 4 if r["seed"] != 3535 else -10),
                                textcoords="offset points", fontsize=8)
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_axisbelow(True)
            ax.grid(axis="both" if scatter else "y", alpha=0.23)
            if scatter:
                ax.set_xlabel("Greedy action agreement (%)")
                ax.set_ylabel("Archived stage-end win rate (%)")
                ax.margins(x=0.19, y=0.13)
            else:
                ax.set_ylabel("Greedy action agreement (%)")
                ax.set_ylim(0, 100)
                ax.set_xlim(-0.45, len(METHODS) - 0.48)
        fig.text(0.5, 0.025, "Filled circles: K ablation, seed 1111     |     "
                 "Open triangle/square: earlier FRPR, seeds 2222/3535",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=2.4, w_pad=2.4)
        fig.savefig(dest / filename, dpi=170)
        plt.close(fig)


def summarize(output):
    manifest = read(output / "manifest.json")
    for name, expected in manifest["analysis_sha256"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Analysis source changed since submission: {name}")
    summary, episodes, confidence, agents, audit = [], [], [], [], []
    for ref in manifest["references"]:
        dest = output / ref["id"]
        metrics = read(dest / "metrics.json")
        blocks = metrics["trajectory_blocks"]
        seed0 = 91101 + 100 * TASKS.index(ref["task"])
        per_seed = manifest["episodes_per_seed"]
        expected = {(seed, ep) for seed in range(seed0, seed0 + 4) for ep in range(per_seed)}
        if (metrics["replay_action_mismatches"]
                or [(b["env_seed"], b["episodes"]) for b in blocks] !=
                [(seed, per_seed) for seed in range(seed0, seed0 + 4)]
                or any(sha(dest / b["archive"]) != b["sha256"] for b in blocks)):
            raise ValueError(f"Incomplete replay audit for {ref['id']}")
        wins = sum(b["wins"] for b in blocks)
        recorded = ref["policy_evaluation"]
        audit.append(dict(reference_id=ref["id"], task=ref["task"], checkpoint=ref["policy"],
                          evaluated_episodes=len(expected), replay_action_mismatches=0,
                          rollout_wins=wins, historical_reference_evaluation=recorded,
                          trajectory_blocks=blocks))
        for pair in ref["comparisons"]:
            rows = [r for r in metrics["episodes"] if r["comparison_id"] == pair["id"]]
            if (len(rows) != len(expected)
                    or {(r["env_seed"], r["episode"]) for r in rows} != expected
                    or sum(r["old_won"] for r in rows) != wins):
                raise ValueError(f"Incomplete episode coverage: {pair['id']}")
            for r in rows:
                for key in ("valid", "agreed"):
                    pos = ("valid", "agreed").index(key)
                    if sum(r["agent_" + key]) != r[key] or sum(b[pos] for b in r["bins"]) != r[key]:
                        raise ValueError(f"Inconsistent denominators: {pair['id']}")
            identity = dict(comparison_id=pair["id"], **{k: pair[k] for k in
                            ("run_id", "method", "seed", "cohort", "task", "stage", "reference")})
            episode_agreements = [pct(r["agreed"], r["valid"]) for r in rows if r["valid"]]
            overall = summed(rows, lambda r: r["valid"], lambda r: r["agreed"])
            archived = pair["evaluation"]
            summary.append(dict(identity, **overall, top1_flip_pct=100 - overall["agr_pct"],
                                mean_episode_agr_pct=float(np.mean(episode_agreements)),
                                episodes=len(rows), episodes_with_valid=len(episode_agreements),
                                old_checkpoint=pair["old_checkpoint"]["path"],
                                old_commit=pair["old_checkpoint"]["commit"],
                                new_checkpoint=pair["new_checkpoint"]["path"],
                                new_commit=pair["new_checkpoint"]["commit"],
                                new_t_env=pair["new_checkpoint"]["t_env"],
                                old_task_wins=archived["wins"], old_task_win_episodes=archived["episodes"],
                                old_task_win_pct=100 * archived["win_rate"]))
            for r in rows:
                episodes.append(dict(identity, **{k: r[k] for k in
                                     ("env_seed", "episode", "length", "old_won")},
                                     valid_decisions=r["valid"], agreements=r["agreed"],
                                     agr_pct=pct(r["agreed"], r["valid"])))
            for i in range(len(rows[0]["agent_valid"])):
                agents.append(dict(identity, agent_id=i, **summed(
                    rows, lambda r: r["agent_valid"][i], lambda r: r["agent_agreed"][i])))
            for i, (lower, upper) in enumerate(zip(BINS, BINS[1:])):
                confidence.append(dict(identity, gap_lower_inclusive=lower,
                                       gap_upper_exclusive=upper, **summed(
                                           rows, lambda r: r["bins"][i][0], lambda r: r["bins"][i][1])))

    corr = []
    for panel in PANELS:
        group = panel_rows(summary, panel)
        for subset in ("k_ablation", "all_available"):
            sample = [r for r in group if subset == "all_available" or r["cohort"] == subset]
            x, y = ([r[k] for r in sample] for k in ("agr_pct", "old_task_win_pct"))
            defined = len(x) >= 3 and len(set(x)) > 1 and len(set(y)) > 1
            corr.append(dict(task=panel[0], stage=panel[1], reference=panel[2], subset=subset,
                             n=len(sample), pearson_r=float(pearsonr(x, y)[0]) if defined else None,
                             spearman_rho=float(spearmanr(x, y)[0]) if defined else None))
    dest = output / "public"
    dest.mkdir(exist_ok=False)
    for name, rows in (("summary", summary), ("per_episode", episodes),
                       ("confidence", confidence), ("per_agent", agents), ("correlation", corr)):
        csv_out(dest / f"{name}.csv", rows)
    write(dest / "rollout_audit.json", audit)
    write(dest / "manifest.json", manifest)
    plots(dest, summary)


if __name__ == "__main__":
    summarize(Path(sys.argv[1]).resolve())
