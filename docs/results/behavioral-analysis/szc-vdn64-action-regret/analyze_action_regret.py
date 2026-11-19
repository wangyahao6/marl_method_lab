"""Compare frozen VDN agents on verified greedy-action-agreement archives."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(
    Path(__file__).resolve().parent.parent / "szc-vdn64-greedy-action-agreement")]

from run_analysis import BINS, METHODS, read, sha, write
from components.episode_buffer import EpisodeBatch
from controllers import REGISTRY
from drivers.continual.common.runtime import episode_spec
from learners.action_set_preference import decision_masks, unroll
from learners.ranking_preference import reference_order
from learners.action_regret import action_regret

LOSS_BINS = (0.1, 0.5, 1.0)
LOSS_NAMES = ("lt_0_1", "0_1_to_0_5", "0_5_to_1", "ge_1")
CONF_LABELS = ("[0, .25)", "[.25, .5)", "[.5, 1)", "[1, 1.5)", "≥1.5")
PANELS = (("stalker", "Z", "own_stage"), ("stalker", "C", "own_stage"),
          ("zealot", "C", "own_stage"), ("zealot", "C", "independent_z"))
TITLES = ("S after Z", "S after C", "Z after C · own Z", "Z after C · independent Z")
COLORS = dict(zip(METHODS, ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")))
SEED_MARKERS = {1111: "o", 2222: "^", 3535: "s"}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(x):
    return float(np.mean(x, dtype=np.float64)) if x.size else None


def quantile(x, q):
    return float(np.quantile(x, q)) if x.size else None


def ratio(n, d):
    return 100 * n / d if d else None


def stats(raw, norm, flipped):
    n, k = raw.size, int(flipped.sum())
    changed_raw, changed_norm = raw[flipped], norm[flipped]
    counts = np.bincount(np.searchsorted(LOSS_BINS, changed_norm, side="right"), minlength=4)
    row = dict(valid_decisions=n, flips=k, flip_pct=ratio(k, n),
               mean_regret=mean(raw), mean_regret_norm=mean(norm),
               flip_mean_regret=mean(changed_raw), flip_mean_regret_norm=mean(changed_norm),
               flip_median_regret=quantile(changed_raw, 0.5),
               flip_p90_regret=quantile(changed_raw, 0.9),
               flip_median_regret_norm=quantile(changed_norm, 0.5),
               flip_p90_regret_norm=quantile(changed_norm, 0.9))
    for name, count in zip(LOSS_NAMES, counts):
        row[f"flips_{name}"] = int(count)
        row[f"flip_share_{name}_pct"] = ratio(count, k)
    return row


def team_stats(team, active):
    return dict(team_steps=team.size, team_steps_with_valid_agents=int(active.sum()),
                mean_team_regret=mean(team), mean_team_regret_active_steps=mean(team[active]))


def checked_inputs(source):
    public = source / "public"
    paths = [source / "manifest.json", *(public / file for file in (
        "manifest.json", "rollout_audit.json", "summary.csv", "per_episode.csv",
        "per_agent.csv", "confidence.csv"))]
    manifest = read(paths[0])
    if manifest != read(paths[1]):
        raise ValueError("The original experiment manifests differ")
    refs = {ref["id"]: ref for ref in manifest["references"]}
    audits = {audit["reference_id"]: audit for audit in read(public / "rollout_audit.json")}
    if set(refs) != set(audits):
        raise ValueError("Reference coverage differs from the original rollout audit")
    archives, points = {}, {}
    for id_, ref in refs.items():
        audit = audits[id_]
        if audit["checkpoint"] != ref["policy"] or audit["task"] != ref["task"]:
            raise ValueError(f"Reference checkpoint does not match the rollout audit: {id_}")
        for block in audit["trajectory_blocks"]:
            archives[(id_, block["env_seed"])] = block
        for pair in ref["comparisons"]:
            for point in (pair["old_checkpoint"], pair["new_checkpoint"], ref["policy"]):
                if points.setdefault(point["path"], point) != point:
                    raise ValueError(f"Conflicting checkpoint metadata: {point['path']}")
            if pair["old_checkpoint"]["sha256"] != ref["policy"]["sha256"]:
                raise ValueError(f"Reference weights differ: {pair['id']}")
    # Validate every input before loading any archived episode or running inference.
    for (ref_id, _), block in archives.items():
        path = source / ref_id / block["archive"]
        if sha(path) != block["sha256"]:
            raise ValueError(f"Trajectory hash mismatch: {path}")
    for path, point in points.items():
        if sha(ROOT / path) != point["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {path}")
    provenance = dict(agr_inputs={str(p.relative_to(source)): sha(p) for p in paths},
                      checkpoints=list(points.values()),
                      archives=[dict(reference_id=ref, **block)
                                for (ref, _), block in archives.items()])
    return manifest, archives, provenance


def controller(point, scheme, groups):
    config_path = ROOT / "runs" / point["run_id"] / "config.json"
    cfg = read(config_path)
    if (cfg["mixer"] != "vdn" or cfg["mac"] != "basic_mac"
            or cfg["agent"] != "rnn" or cfg["rnn_hidden_dim"] != 64
            or not cfg["obs_last_action"] or not cfg["obs_agent_id"]
            or cfg["agent_output_type"] != "q" or cfg.get("trhr")):
        raise ValueError(f"Expected the original unmodified VDN64 controller: {config_path}")
    args = SimpleNamespace(**cfg)
    args.device, args.use_cuda = "cpu", False
    mac = REGISTRY[args.mac](scheme, groups, args)
    mac.agent.load_state_dict(torch.load(ROOT / point["path"], map_location="cpu"))
    mac.agent.eval().requires_grad_(False)
    return mac


def episode_batch(record, scheme, groups):
    steps = record["length"]
    data = {k: v.unsqueeze(0) for k, v in record["data"].items()}
    if (any(v.shape[1] != steps + 1 for v in data.values())
            or not data["filled"].all()
            or not torch.equal(data["actions_onehot"],
                               torch.zeros_like(data["actions_onehot"]).scatter_(-1, data["actions"], 1))):
        raise ValueError(f"Malformed rollout: {record['seed']}/{record['index']}")
    return EpisodeBatch(scheme, groups, 1, steps + 1,
                        data=SimpleNamespace(transition_data=data, episode_data={}))


def legal_q(mac, batch, steps, legal):
    q = unroll(mac, batch, steps)
    if not torch.isfinite(q[legal]).all():
        raise ValueError("Non-finite legal Q values")
    return q.masked_fill(~legal, -torch.inf)


def old_values(mac, batch, steps, legal):
    old_q = legal_q(mac, batch, steps, legal)
    best, actions = old_q.max(dim=-1)
    executed = batch["actions"][:, :-1]
    if not legal.gather(-1, executed).all() or not torch.equal(actions, executed.squeeze(-1)):
        raise ValueError("Old greedy Q actions do not reproduce the executed trajectory")
    # The original experiment computes the population STD on legal actions.
    _, scale, _, confidence = reference_order(old_q, legal, q_scale_floor=0.1)
    return old_q, best, actions, scale.squeeze(-1), confidence


def contrast(mac, batch, steps, legal, valid, reference):
    old_q, best, old_actions, scale, confidence = reference
    new_actions = legal_q(mac, batch, steps, legal).argmax(dim=-1)
    raw, norm = (values[0] for values in action_regret(old_q, new_actions, best, scale))
    if not torch.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Invalid regret for a legal action")
    mask = valid[0]
    raw = raw.masked_fill(~mask, 0)
    norm = norm.masked_fill(~mask, 0)
    flipped = ((old_actions != new_actions)[0] & mask)
    return dict(raw=raw.numpy(), norm=norm.numpy(), flipped=flipped.numpy(),
                valid=mask.numpy(), confidence=confidence[0].numpy(),
                team=raw.double().sum(dim=-1).numpy(), active=mask.any(dim=-1).numpy())


def identity(pair):
    return dict(comparison_id=pair["id"], **{k: pair[k] for k in
                  ("run_id", "method", "seed", "cohort", "task", "stage", "reference")})


def expect_count(row, previous, label):
    if (row["valid_decisions"] != int(previous["valid_decisions"])
            or row["flips"] != int(previous["valid_decisions"]) - int(previous["agreements"])):
        raise ValueError(f"Decision or action-flip count disagrees with AGR: {label}")


def plot(dest, summary, confidence):
    def panel_rows(rows, panel):
        return [r for r in rows if (r["task"], r["stage"], r["reference"]) == panel]

    def base():
        fig, axs = plt.subplots(2, 2, figsize=(11.6, 7.8))
        for ax, title in zip(axs.flat, TITLES):
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.25)
        return fig, axs.flat

    handles = [Line2D([], [], marker="o", color=COLORS[m], label=m) for m in METHODS]
    handles += [Line2D([], [], marker=SEED_MARKERS[s], linestyle="none", markerfacecolor="none",
                       markeredgecolor=COLORS["FRPR"], label=f"Earlier FRPR {s}") for s in (2222, 3535)]

    def save(fig, filename, legend):
        fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=2.0)
        fig.savefig(dest / filename, dpi=165)
        plt.close(fig)

    fig, axes = base()
    for ax, panel in zip(axes, PANELS):
        for r in panel_rows(summary, panel):
            x = METHODS.index(r["method"]) + {2222: -0.19, 3535: 0.19}.get(r["seed"], 0)
            color = COLORS[r["method"]]
            main = r["cohort"] == "k_ablation"
            ax.scatter(x, r["mean_regret_norm"], s=54, marker=SEED_MARKERS.get(r["seed"], "D"),
                       facecolors=color if main else "none", edgecolors=color, zorder=3)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS)
        ax.set_ylim(0, 0.40)
        ax.set_ylabel("Mean normalized regret · all decisions")
    save(fig, "mean_regret_by_method.png", handles)

    fig, axes = base()
    for ax, panel in zip(axes, PANELS):
        for r in panel_rows(summary, panel):
            main = r["cohort"] == "k_ablation"
            color = COLORS[r["method"]]
            ax.scatter(r["flip_pct"], r["mean_regret_norm"], s=58,
                       marker=SEED_MARKERS.get(r["seed"], "D"),
                       facecolors=color if main else "none", edgecolors=color, zorder=3)
        ax.set_xlabel("Top-1 flip rate (%)")
        ax.set_ylabel("Mean normalized regret · all decisions")
        ax.margins(x=0.15, y=0.18)
    save(fig, "flip_rate_vs_regret.png", handles)

    fig, axes = base()
    for ax, panel in zip(axes, PANELS):
        by_pair = defaultdict(list)
        for r in panel_rows(confidence, panel):
            by_pair[r["comparison_id"]].append(r)
        for group in by_pair.values():
            r = group[0]
            vals = [g["flip_mean_regret_norm"] for g in group]
            main = r["cohort"] == "k_ablation"
            ax.plot(range(len(vals)), vals, color=COLORS[r["method"]],
                    marker=SEED_MARKERS.get(r["seed"], "D"), markersize=4,
                    linestyle="-" if main else "--", alpha=1 if main else 0.65,
                    fillstyle="full" if main else "none")
        ax.set_xticks(range(len(CONF_LABELS)))
        ax.set_xticklabels(CONF_LABELS)
        ax.set_xlabel("Old policy normalized top-1 margin")
        ax.set_ylabel("Mean normalized regret · flips only")
    save(fig, "regret_by_confidence.png", handles)

    fig, axes = base()
    levels = ("<0.1", "0.1–<0.5", "0.5–<1", "≥1")
    palettes = ("#86C7BD", "#CAE7A1", "#FACF76", "#C95F4D")
    bin_handles = [Line2D([], [], linestyle="none", marker="s", markerfacecolor=c,
                          markeredgecolor=c, markersize=9, label=t)
                   for t, c in zip(levels, palettes)]
    for ax, panel in zip(axes, PANELS):
        rows = sorted(panel_rows(summary, panel), key=lambda r: (METHODS.index(r["method"]), r["seed"]))
        bottom = np.zeros(len(rows))
        for name, color in zip(LOSS_NAMES, palettes):
            values = [r[f"flip_share_{name}_pct"] for r in rows]
            ax.bar(range(len(rows)), values, bottom=bottom, color=color)
            bottom += values
        labels = [r["method"] if r["seed"] == 1111 else f"FRPR\n{r['seed']}" for r in rows]
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Share of flipped decisions (%)")
    save(fig, "flip_regret_distribution.png", bin_handles)


@torch.inference_mode()
def analyze(source, output):
    torch.set_num_threads(1)
    manifest, archives, provenance = checked_inputs(source)
    agr_summary = {r["comparison_id"]: r for r in read_csv(source / "public/summary.csv")}
    agr_episodes = {(r["comparison_id"], int(r["env_seed"]), int(r["episode"])): r
                    for r in read_csv(source / "public/per_episode.csv")}
    agr_agents = {(r["comparison_id"], int(r["agent_id"])): r
                  for r in read_csv(source / "public/per_agent.csv")}
    agr_confidence = {(r["comparison_id"], float(r["gap_lower_inclusive"])): r
                      for r in read_csv(source / "public/confidence.csv")}
    metrics, episodes, agents, confidence, audit = [], [], [], [], []
    for ref in manifest["references"]:
        ref_id = ref["id"]
        pairs = {p["id"]: p for p in ref["comparisons"]}
        first = next(iter(pairs.values()))
        info = read(ROOT / "runs" / first["run_id"] / "env_info.json")
        spec, groups, preprocess = episode_spec(info)
        scheme = EpisodeBatch(spec, groups, 1, 1, preprocess=preprocess).scheme
        old = controller(ref["policy"], scheme, groups)
        models = {i: controller(p["new_checkpoint"], scheme, groups) for i, p in pairs.items()}
        chunks = defaultdict(list)
        ref_eps = ref_actions = 0
        for (key, seed), block in archives.items():
            if key != ref_id:
                continue
            entries = torch.load(source / key / block["archive"], map_location="cpu")
            if (len(entries) != block["episodes"]
                    or sum(e["won"] for e in entries) != block["wins"]
                    or sum(e["length"] for e in entries) != block["transitions"]):
                raise ValueError(f"Trajectory block metadata mismatch: {ref_id}/{seed}")
            for index, entry in enumerate(entries):
                if (entry["seed"], entry["index"]) != (seed, index):
                    raise ValueError(f"Episode order mismatch: {ref_id}/{seed}/{index}")
                steps = entry["length"]
                batch = episode_batch(entry, scheme, groups)
                legal, alive = decision_masks(batch)
                valid = alive & (legal.sum(dim=-1) > 1)
                values = old_values(old, batch, steps, legal)
                ref_eps += 1
                ref_actions += steps * info["n_agents"]
                for pair_id, mac in models.items():
                    result = contrast(mac, batch, steps, legal, valid, values)
                    chunks[pair_id].append(result)
                    eligible = result["valid"]
                    row = dict(identity(pairs[pair_id]), env_seed=seed, episode=index, length=steps,
                               old_won=entry["won"], **team_stats(result["team"], result["active"]),
                               **stats(*(result[k][eligible] for k in ("raw", "norm", "flipped"))))
                    match_key = pair_id, seed, index
                    prior = agr_episodes.pop(match_key)
                    expect_count(row, prior, match_key)
                    if steps != int(prior["length"]) or str(entry["won"]) != prior["old_won"]:
                        raise ValueError(f"Episode metadata differs from AGR: {match_key}")
                    episodes.append(row)
        for pair_id, pair in pairs.items():
            series = {k: np.concatenate([r[k] for r in chunks[pair_id]]) for k in chunks[pair_id][0]}
            eligible = series["valid"]
            raw, norm, flipped = (series[k] for k in ("raw", "norm", "flipped"))
            common = identity(pair)
            new, old_point = pair["new_checkpoint"], pair["old_checkpoint"]
            row = dict(common, episodes=len(chunks[pair_id]),
                       old_checkpoint=old_point["path"], new_checkpoint=new["path"],
                       new_t_env=new["t_env"], **team_stats(series["team"], series["active"]),
                       **stats(raw[eligible], norm[eligible], flipped[eligible]))
            row["agr_pct"] = 100 * (row["valid_decisions"] - row["flips"]) / row["valid_decisions"]
            expect_count(row, agr_summary.pop(pair_id), pair_id)
            metrics.append(row)
            for agent_id in range(info["n_agents"]):
                mask = eligible[:, agent_id]
                row = dict(common, agent_id=agent_id,
                           **stats(raw[mask, agent_id], norm[mask, agent_id], flipped[mask, agent_id]))
                expect_count(row, agr_agents.pop((pair_id, agent_id)), (pair_id, agent_id))
                agents.append(row)
            margins = series["confidence"]
            for lo, hi in zip(BINS, BINS[1:]):
                mask = eligible & (margins >= lo)
                if hi is not None:
                    mask &= margins < hi
                row = dict(common, gap_lower_inclusive=lo, gap_upper_exclusive=hi,
                           **stats(raw[mask], norm[mask], flipped[mask]))
                expect_count(row, agr_confidence.pop((pair_id, lo)), (pair_id, lo))
                confidence.append(row)
        audit.append(dict(reference_id=ref_id, episodes=ref_eps, old_actions_verified=ref_actions,
                          old_action_mismatches=0, comparisons=len(pairs)))
        print(f"{ref_id}: verified {ref_eps} archived trajectories, {len(pairs)} comparisons", flush=True)

    if any((agr_summary, agr_episodes, agr_agents, agr_confidence)):
        raise ValueError("Original AGR rows were left unmatched")
    output.mkdir(parents=True, exist_ok=False)
    plots = output / "plots"
    plots.mkdir()
    for key, rows in (("summary", metrics), ("per_episode", episodes),
                      ("per_agent", agents), ("confidence", confidence)):
        write_csv(output / f"{key}.csv", rows)
    plot(plots, metrics, confidence)
    write(output / "audit.json", dict(
        source_run=source.name, analysis_script_sha256=sha(Path(__file__)),
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        **provenance, references=audit, comparisons_verified=len(metrics),
        episode_rows_verified=len(episodes), agent_rows_verified=len(agents),
        confidence_rows_verified=len(confidence), new_trajectory_rollouts=0, new_training_steps=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original AGR run directory")
    parser.add_argument("output", type=Path, help="New directory for regret results")
    options = parser.parse_args()
    if options.output.exists():
        parser.error(f"Output already exists: {options.output}")
    analyze(options.source.resolve(), options.output.resolve())
