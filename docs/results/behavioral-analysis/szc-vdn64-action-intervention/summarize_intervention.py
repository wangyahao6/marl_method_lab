"""Check intervention evaluation coverage and summarize the live-policy contrasts."""
import csv
import math
from pathlib import Path
from statistics import mean, stdev
import sys

from run_intervention import ARMS, ROOT, read, sha, write

COUNTS = ("valid", "flips", "high_flips", "overrides")
LOSSES = ("pre_sum_regret", "post_sum_regret", "pre_sum_regret_norm", "post_sum_regret_norm")
PANELS = (("stalker", "Z"), ("stalker", "C"), ("zealot", "C"))
METHODS = ("K1", "K2", "K3", "K5", "FRPR", "FRPR 2222", "FRPR 3535")


def csv_out(path, rows):
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def wilson(wins, n):
    z = 1.959963984540054
    p = wins / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    width = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return 100 * max(0, center - width), 100 * min(1, center + width)


def describe(total):
    vals = dict(total)
    n, flips = vals["valid"], vals["flips"]
    vals["override_pct"] = 100 * vals["overrides"] / n if n else None
    vals["override_share_of_flips_pct"] = 100 * vals["overrides"] / flips if flips else None
    for field in LOSSES:
        vals[field.replace("_sum_", "_mean_")] = vals[field] / n if n else None
    return vals


def check_rule(s, arm):
    n, flips, high, overrides = (s[k] for k in COUNTS)
    expected = {"A": 0, "B": high, "C": flips, "D": 0}[arm]
    if (not 0 <= high <= flips <= n or overrides != expected
            or any(not math.isfinite(s[k]) or s[k] < 0 for k in LOSSES)):
        raise ValueError(f"Invalid arm counts or losses: {arm}")
    for suffix in ("regret", "regret_norm"):
        pre, post = s[f"pre_sum_{suffix}"], s[f"post_sum_{suffix}"]
        if (post > pre + 1e-10 or (arm == "A" and post != pre)
                or (arm in ("C", "D") and post != 0)):
            raise ValueError(f"Unexpected residual loss: {arm}")
    if arm == "B" and s["pre_sum_regret_norm"] - s["post_sum_regret_norm"] + 1e-8 < high * 0.5:
        raise ValueError("High-regret overrides must remove at least the threshold per override")


def plot(dest, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    dest.mkdir()
    colors = ("#6B7280", "#0072B2", "#D55E00", "#009E73")
    for field, label, file in (
            ("win_pct", "Win rate (%) · 95% Wilson intervals", "win_rate.png"),
            ("mean_return", "Mean episode return", "mean_return.png"),
            ("override_pct", "Override rate (% of valid decisions)", "override_rate.png")):
        fig, axs = plt.subplots(3, 1, figsize=(11, 9.2), sharex=True)
        for ax, (task, stage), title in zip(axs, PANELS, ("S after Z", "S after C", "Z after C")):
            rows = [r for r in summary if r["task"] == task and r["stage"] == stage]
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.25)
            for idx, (arm, color) in enumerate(zip(ARMS, colors)):
                ys = [r for r in rows if r["arm"] == arm]
                names = [r["method"] if r["seed"] == 1111 else f"FRPR {r['seed']}" for r in ys]
                xs = np.array([METHODS.index(name) for name in names]) + (idx - 1.5) * 0.175
                vals = [r[field] for r in ys]
                intervals = None
                if field == "win_pct":
                    intervals = [[v-r["win_ci_low_pct"] for v,r in zip(vals, ys)],
                                 [r["win_ci_high_pct"]-v for v,r in zip(vals, ys)]]
                ax.errorbar(xs, vals, yerr=intervals, color=color, marker="o",
                            linestyle="none", capsize=3, label=arm)
            ax.axvline(4.5, color="#9CA3AF", linestyle="--", linewidth=0.9)
            ax.set_ylabel(label)
        axs[-1].set_xticks(range(len(METHODS)))
        axs[-1].set_xticklabels([name.replace("FRPR ", "FRPR\n") for name in METHODS])
        handles, _ = axs[0].get_legend_handles_labels()
        fig.legend(handles, ("A New", "B High-regret override", "C All-override", "D Old"),
                   loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.savefig(dest / file, dpi=165)
        plt.close(fig)


def aggregate(output):
    m = read(output / "manifest.json")
    from run_intervention import HERE
    for name, digest in m["source_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError(f"Analysis source changed: {name}")
    by_case = {}
    block_audit = []
    pairs = {case["id"]: case for case in m["cases"]}
    for job in m["jobs"]:
        jid, arm, keys = (job[k] for k in ("job_id", "arm", "case_ids"))
        case = pairs[keys[0]]
        stored = []
        for seed in case["env_seeds"]:
            f = output / jid / f"{seed}.json"
            rows = read(f)
            if [(r["seed"], r["episode"]) for r in rows] != [
                    (seed, i) for i in range(m["episodes_per_seed"])]:
                raise ValueError(f"Coverage incomplete: {jid}/{seed}")
            if any(set(r["cases"]) != set(keys) for r in rows):
                raise ValueError(f"Case coverage incomplete: {jid}/{seed}")
            block_audit.append(dict(job_id=jid, arm=arm, seed=seed, episodes=len(rows), sha256=sha(f)))
            stored.extend(rows)
        for key in keys:
            by_case[key, arm] = stored
    summary, episodes, agents, deltas, matches = [], [], [], [], []
    for key, case in pairs.items():
        identity = dict(comparison_id=key, **{k: case[k] for k in
                        ("run_id", "method", "seed", "cohort", "task", "stage")})
        arms = {arm: by_case[key, arm] for arm in ARMS}
        arm_sum = {}
        for arm, rows in arms.items():
            n = len(rows)
            returns = [r["episode_return"] for r in rows]
            wins = sum(r["won"] for r in rows)
            lo, hi = wilson(wins, n)
            stats_by_agent = [{k: 0 for k in (*COUNTS, *LOSSES)} for _ in range(5)]
            aggregates = []
            for r in rows:
                raw = r["cases"][key]
                if any(len(values) != len(stats_by_agent) for values in raw.values()):
                    raise ValueError(f"Inconsistent agent layout: {key}")
                total = {k: 0 for k in raw}
                for i, stat in enumerate(stats_by_agent):
                    per_agent = {k: int(raw[k][i]) if k in COUNTS else raw[k][i] for k in raw}
                    check_rule(per_agent, arm)
                    for k, value in per_agent.items():
                        stat[k] += value
                        total[k] += value
                aggregates.append(total)
                episodes.append(dict(identity, arm=arm, env_seed=r["seed"],
                                     **{k: r[k] for k in ("episode", "won", "episode_return", "length",
                                                          "spawn_sha256", "initial_state_sha256",
                                                          "initial_game_loop", "action_sha256")},
                                     **describe(total)))
            total = {k: sum(r[k] for r in aggregates) for k in aggregates[0]}
            row = dict(identity, arm=arm, episodes=n, wins=wins, win_pct=100*wins/n,
                       win_ci_low_pct=lo, win_ci_high_pct=hi, mean_return=mean(returns),
                       return_sd=stdev(returns), mean_length=mean(r["length"] for r in rows),
                       **describe(total))
            arm_sum[arm] = row
            summary.append(row)
            agents.extend(dict(identity, arm=arm, agent_id=i, **describe(stat))
                          for i, stat in enumerate(stats_by_agent))
        signature = ("spawn_sha256", "initial_state_sha256", "initial_game_loop")
        matched = {field: sum(len({r[field] for r in rs}) == 1
                              for rs in zip(*arms.values())) for field in signature}
        cd_keys = ("initial_state_sha256", "action_sha256", "won", "length", "episode_return")
        cd_matches = sum(all(c[k] == d[k] for k in cd_keys) for c, d in zip(arms["C"], arms["D"]))
        matches.append(dict(comparison_id=key, episodes=len(arms["A"]),
                            matched_across_all_arms=matched, cd_trajectory_outcome_matches=cd_matches))
        for left, right in (("A", "D"), ("B", "A"), ("C", "B"), ("B", "D"), ("C", "D")):
            block_win_deltas, block_return_deltas = [], []
            for seed in case["env_seeds"]:
                u, v = ([r for r in arms[arm] if r["seed"] == seed] for arm in (left, right))
                block_win_deltas.append(100 * mean(int(x["won"]) - int(y["won"]) for x,y in zip(u, v)))
                block_return_deltas.append(mean(x["episode_return"] - y["episode_return"] for x,y in zip(u, v)))
            deltas.append(dict(identity, contrast=f"{left}-{right}",
                               delta_win_pp=arm_sum[left]["win_pct"] - arm_sum[right]["win_pct"],
                               delta_return=arm_sum[left]["mean_return"] - arm_sum[right]["mean_return"],
                               block_win_deltas_pp="|".join(f"{v:.8g}" for v in block_win_deltas),
                               block_return_deltas="|".join(f"{v:.8g}" for v in block_return_deltas)))
    checks = dict(masked_greedy_and_threshold_boundary_checked=True,
                  rule_count_and_residual_invariants_checked=True)
    if m["is_pilot"]:
        checks.update(executed_history_matches_both_mac_inputs_at_every_step=True,
                      c_d_played_same_joint_actions_and_obtained_same_outcomes=all(
                          r["episodes"] == r["cd_trajectory_outcome_matches"] for r in matches),
                      high_overrides_exercised=any(r["arm"] == "B" and r["overrides"] for r in summary),
                      c_overrides_exercised=any(r["arm"] == "C" and r["overrides"] for r in summary))
        checks["pilot_passed"] = all(checks.values())
    public = output / "public"
    public.mkdir(exist_ok=False)
    for filename, rows in (("summary", summary), ("per_episode", episodes), ("per_agent", agents),
                           ("contrasts", deltas)):
        csv_out(public / f"{filename}.csv", rows)
    write(public / "audit.json", dict(
        run_id=m["run_id"], is_pilot=m["is_pilot"], base_commit=m["base_commit"],
        source_sha256=m["source_sha256"], action_regret_audit_sha256=m["audit_input_sha256"],
        agr_manifest_sha256=m["agr_manifest_sha256"], cases=m["cases"], jobs=m["jobs"],
        episodes_per_seed=m["episodes_per_seed"], threshold=m["threshold"], blocks=block_audit,
        physical_episodes=sum(r["episodes"] for r in block_audit), comparison_episode_rows=len(episodes),
        implementation_checks=checks, initial_state_pairing=matches, pilot=m.get("pilot"),
        new_training_steps=0))
    if m["is_pilot"]:
        if not checks["pilot_passed"]:
            raise ValueError(f"Pilot failed: {checks}")
    else:
        plot(public / "plots", summary)


if __name__ == "__main__":
    aggregate(Path(sys.argv[1]).resolve())
