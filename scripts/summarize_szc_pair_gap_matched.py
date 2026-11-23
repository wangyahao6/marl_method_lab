"""Verify the completed gap-matched branch against archived K1 and Ratio-0 runs."""
import csv
import gzip
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch as th
from torch.nn.utils.rnn import pad_sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "drivers/continual"))
sys.path.insert(0, str(ROOT / "src"))
from action_set_protocol import matching_pair_reference
from common.runtime import episode_spec
from components.episode_buffer import ReplayBuffer
from controllers import REGISTRY
from learners.action_set_preference import batch_episodes, decision_masks, unroll
from summarize_szc_tkpr_k1_shared import gradient_rows, records, stage_rows


BATCH = "szc_pair_gapmatched_3535_20261123"
OLD = "szc_pair_matched_3535_20261119"
REFERENCE = ROOT / "runs/pair_structure" / (OLD + "_reference")
RUNS = {
    "TKPR-K1": ROOT / "runs/pair_structure" / (OLD + "_pair_structure_k1"),
    "Ratio-0": ROOT / "runs/pair_ratio" / (OLD + "_pair_ratio_0"),
    "Gap-Matched-NonTop1": ROOT / "runs/pair_gap_matched" / (BATCH + "_pair_gap_matched_non_top1"),
}
PREFLIGHT = ROOT / "results" / BATCH / "preflight"
OUT = ROOT / "docs/results/parameter-ablations/szc-vdn64-pair-gap-matched"
TASKS = ("stalker", "zealot", "colossus")


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("empty table " + name)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / (name + ".csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b):
    return a / b if b else None


def verify_run(path, method, reference_launch, reference_protocol, reference_restore):
    status, launcher, launch, protocol, restore = (
        read(path / name) for name in ("status.json", "launcher_status.json", "launch.json",
                                     "protocol.json", "branch_restore.json"))
    if status.get("status") != "completed" or launcher.get("exit_code") != 0:
        raise ValueError("unfinished or failed run: " + method)
    if not matching_pair_reference(reference_protocol, protocol):
        raise ValueError("shared protocol changed: " + method)
    if (restore.get("parent") != str(REFERENCE) or restore.get("model_hash") != reference_restore
            or restore.get("parent_model_hash") != reference_restore):
        raise ValueError("S fork checkpoint differs: " + method)
    return launch, protocol, restore, status


@th.no_grad()
def final_policy_audit(path, identity):
    """Rank the archived reference trajectories with the frozen C-end agent."""
    from types import SimpleNamespace

    args = SimpleNamespace(**read(path / "config.json"))
    args.device, args.use_cuda = "cpu", False
    info = read(path / "env_info.json")
    scheme, groups, preprocess = episode_spec(info)
    augmented = ReplayBuffer(scheme, groups, 1, 2, preprocess=preprocess).scheme
    mac = REGISTRY[args.mac](augmented, groups, args)
    final_stage = read(path / "retention.json")["stages"][-1]
    checkpoint = path / final_stage["checkpoint"] / "agent.th"
    mac.agent.load_state_dict(th.load(checkpoint, map_location="cpu"))
    mac.agent.eval()
    output = []
    for phase, task in enumerate(TASKS[:2]):
        source = REFERENCE / "action_set_memory" / f"phase{phase:04d}" / "success.th"
        episodes = th.load(source, map_location="cpu")["episodes"]
        batch, _ = batch_episodes(episodes, augmented, groups, "cpu",
                                  include_good=False, include_old_q=True)
        legal, _ = decision_masks(batch)
        old = batch["old_q_values"][:, :-1]
        current = unroll(mac, batch, batch.max_seq_length - 1)
        masks = th.load(path / "pair_ablation_memory" / f"phase{phase:04d}" / "selection.th",
                        map_location="cpu")
        selected = pad_sequence(masks, batch_first=True)
        valid = selected.any(dim=-1)
        n = legal.size(-1)
        left, right = th.triu_indices(n, n, 1)
        eligible = legal[..., left] & legal[..., right] & (old[..., left] != old[..., right])
        direction = (old[..., left] - old[..., right]).sign()
        new_gap = current[..., left] - current[..., right]
        pair_tau = (direction * new_gap.sign()).masked_fill(~eligible, 0)
        state_tau = pair_tau.sum(dim=-1) / eligible.sum(dim=-1).clamp_min(1)
        old_top = old.masked_fill(~legal, -th.inf).argmax(dim=-1)
        new_top = current.masked_fill(~legal, -th.inf).argmax(dim=-1)
        positions = int(valid.sum())
        output.append(dict(identity, historical_task=task, source_phase=phase,
                           final_checkpoint=str(checkpoint.relative_to(ROOT)), valid_states=positions,
                           all_candidate_kendall_tau_a=float(state_tau[valid].mean()),
                           old_top1_retention=float(((old_top == new_top) & valid).sum()) / positions,
                           selected_pair_violation_rate=float((selected & (direction * new_gap <= 0)).sum())
                           / int(selected.sum())))
    return output


def main():
    plan = read(ROOT / "results" / BATCH / "plan.json")
    if len(plan["tasks"]) != 1 or plan["tasks"][0]["dependencies"] != [REFERENCE.name]:
        raise ValueError("expected one branch depending on the archived reference")
    if Path(plan["tasks"][0]["run_dir"]) != RUNS["Gap-Matched-NonTop1"]:
        raise ValueError("plan run directory differs")
    reference_launch = read(REFERENCE / "launch.json")
    reference_protocol = read(REFERENCE / "protocol.json")
    reference_status = read(REFERENCE / "status.json")
    if reference_status.get("status") != "completed" or read(REFERENCE / "launcher_status.json").get("exit_code") != 0:
        raise ValueError("reference did not finish")
    old_evaluations = list(records(REFERENCE / "evaluations.jsonl"))
    if not old_evaluations:
        raise ValueError("missing reference evaluations")
    reference_restore = read(RUNS["TKPR-K1"] / "branch_restore.json")["model_hash"]
    if reference_launch["git_commit"] != "c6b7002560c9827061e2f543be318f32cafb5622":
        raise ValueError("unexpected archived reference commit")
    preflight = read(PREFLIGHT / "summary.json")
    if len(preflight) != 2:
        raise ValueError("missing S or Z preflight")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(exist_ok=True)
    runs, stages, curves, evaluations, pools, updates, history_rows, gradients, final_policy = ([] for _ in range(9))
    ref_identity = dict(run_id=REFERENCE.name, method="TD reference", seed=3535, role="pair-reference",
                        parent_run_id="")
    ref_stage, ref_curve = stage_rows(ref_identity, read(REFERENCE / "retention.json"),
                                      old_evaluations, old_evaluations)
    if not ref_stage or ref_stage[0]["phase"] != 0:
        raise ValueError("missing S reference stage")
    stages.append(ref_stage[0])
    curves.extend(row for row in ref_curve if row["phase"] == 0)
    evaluations.extend(dict(ref_identity, **{key: row[key] for key in
                            ("phase", "task", "label", "t_env", "wins", "episodes")})
                       for row in old_evaluations)
    runs.append(dict(ref_identity, status="completed", exit_code=0,
                     git_commit=reference_launch["git_commit"],
                     run_dir=str(REFERENCE.relative_to(ROOT)), restore_model_hash="",
                     source_pool_s_sha256=preflight[0]["source_sha256"],
                     source_pool_z_sha256=preflight[1]["source_sha256"]))
    relevant_source = ("src/learners/pair_ablation.py", "src/learners/pair_gap_matched.py",
                       "src/learners/ranking_preference.py", "src/learners/action_set_preference.py",
                       "drivers/continual/action_set_protocol.py", "drivers/continual/action_set_train.py")
    (OUT / "configs" / (REFERENCE.name + ".json")).write_text(json.dumps(
        dict(protocol=reference_protocol, launch_git_commit=reference_launch["git_commit"],
             source_sha256={key: reference_launch["source_sha256"].get(key) for key in relevant_source}),
        ensure_ascii=False, indent=2) + "\n")

    for method, path in RUNS.items():
        launch, protocol, restore, status = verify_run(
            path, method, reference_launch, reference_protocol, reference_restore)
        identity = dict(run_id=path.name, method=method, seed=3535, role="shared-branch",
                        parent_run_id=REFERENCE.name)
        own = list(records(path / "evaluations.jsonl"))
        for phase in (1, 2):
            required = set(TASKS[:phase + 1])
            stage_end = [row for row in own if row["phase"] == phase and row["label"] == "stage_end"]
            if {row["task"] for row in stage_end} != required or any(row["episodes"] != 128 for row in stage_end):
                raise ValueError("missing independent stage-end evaluation: " + method)
        joined = [row for row in old_evaluations if row["phase"] == 0] + own
        stage, curve = stage_rows(identity, read(path / "retention.json"), joined, own)
        if len(stage) != 2 or any(row["learning_auc_pct"] is None for row in stage):
            raise ValueError("incomplete stage/AUC: " + method)
        stages.extend(stage)
        curves.extend(curve)
        evaluations.extend(dict(identity, **{key: row[key] for key in
                                ("phase", "task", "label", "t_env", "wins", "episodes")})
                           for row in own)
        runs.append(dict(identity, status="completed", exit_code=0,
                         git_commit=launch["git_commit"],
                         run_dir=str(path.relative_to(ROOT)),
                         restore_model_hash=restore["model_hash"],
                         source_pool_s_sha256=preflight[0]["source_sha256"],
                         source_pool_z_sha256=preflight[1]["source_sha256"]))
        gradients.extend(gradient_rows(identity, path))
        final_policy.extend(final_policy_audit(path, identity))
        (OUT / "configs" / (path.name + ".json")).write_text(json.dumps(
            dict(protocol=protocol, launch_git_commit=launch["git_commit"],
                 source_sha256={key: launch["source_sha256"].get(key) for key in relevant_source},
                 branch_restore=restore), ensure_ascii=False, indent=2) + "\n")
        for phase in (0, 1):
            summary = read(path / "pair_ablation_memory" / f"phase{phase:04d}" / "summary.json")
            expected = preflight[phase]
            if (summary["valid_state_sha256"] != expected["valid_state_sha256"]
                    or summary["valid_pairs"] != expected["selected_pairs"]
                    or summary["source_pool"] != str(REFERENCE / "action_set_memory" /
                                                     f"phase{phase:04d}" / "success.th")):
                raise ValueError("pool/source/budget differs from preflight: " + method)
            expected_hash = {"TKPR-K1": expected["k1_mask_sha256"],
                             "Ratio-0": expected["ratio0_mask_sha256"],
                             "Gap-Matched-NonTop1": expected["matched_mask_sha256"]}[method]
            if summary["pair_mask_sha256"] != expected_hash:
                raise ValueError("fixed mask differs from preflight: " + method)
            pools.append(dict(identity, source_phase=phase,
                              historical_task=("stalker", "zealot")[phase], **summary))
        log_name = {"TKPR-K1": "pair_structure_k1_updates.jsonl",
                    "Ratio-0": "pair_ratio_0_updates.jsonl",
                    "Gap-Matched-NonTop1": "pair_gap_matched_non_top1_updates.jsonl"}[method]
        grouped = defaultdict(list)
        historical = defaultdict(list)
        for row in records(path / log_name):
            grouped[row["phase"]].append(row)
            for item in row["historical_tasks"]:
                historical[row["phase"], item["source_phase"], item["task"]].append(item)
        for phase, rows in sorted(grouped.items()):
            if phase not in (1, 2) or not rows:
                raise ValueError("unexpected preference updates: " + method)
            losses = [row["preference_loss"] for row in rows if row.get("preference_loss") is not None]
            updates.append(dict(identity, phase=phase, training_task=TASKS[phase], updates=len(rows),
                                mean_preference_loss=mean(losses) if losses else None))
        for (phase, source, historical_task), rows in sorted(historical.items()):
            pairs = sum(row["valid_pairs"] for row in rows)
            flips = sum(row["top1_flips"] for row in rows)
            flip_states = sum(row["top1_states"] for row in rows)
            violations = sum(row.get("pair_violations", 0) for row in rows)
            history_rows.append(dict(identity, phase=phase, training_task=TASKS[phase],
                                     source_phase=source, historical_task=historical_task, updates=len(rows),
                                     sampled_valid_states=sum(row["valid_states"] for row in rows),
                                     sampled_pairs=pairs, top1_flip_rate=ratio(flips, flip_states),
                                     top1_retention=1 - ratio(flips, flip_states) if flip_states else None,
                                     selected_pair_violation_rate=(ratio(violations, pairs)
                                                                   if method == "Gap-Matched-NonTop1" else None),
                                     mean_task_loss=mean(row["loss"] for row in rows if row.get("loss") is not None)))
    for phase in (0, 1):
        hashes = {row["valid_state_sha256"] for row in pools if row["source_phase"] == phase}
        if len(hashes) != 1:
            raise ValueError("methods do not share valid states")
    metric_names = ("acc_pct", "bwt_pp", "fwt_pp", "stalker_win_pct", "zealot_win_pct",
                    "colossus_win_pct", "learning_auc_pct")
    paired = []
    by = {(row["method"], row["phase"]): row for row in stages if row["role"] == "shared-branch"}
    for treatment, control in (("Gap-Matched-NonTop1", "TKPR-K1"),
                               ("Gap-Matched-NonTop1", "Ratio-0"), ("Ratio-0", "TKPR-K1")):
        for phase in (1, 2):
            a, b = by[treatment, phase], by[control, phase]
            paired.append(dict(treatment=treatment, control=control, phase=phase,
                               treatment_run_id=a["run_id"], control_run_id=b["run_id"],
                               **{name + "_delta": (a[name] - b[name] if a.get(name) is not None
                                                      and b.get(name) is not None else None)
                                  for name in metric_names}))
    for name, rows in (("runs", runs), ("evaluations", evaluations), ("stage_metrics", stages),
                       ("learning_curves", curves), ("pair_diagnostics", pools),
                       ("update_diagnostics", updates), ("preference_diagnostics", history_rows),
                       ("gradient_diagnostics", gradients), ("paired_metrics", paired)):
        write_csv(name, rows)
    write_csv("final_policy_diagnostics", final_policy)
    for name in ("gap_distribution.csv", "rank_distance_distribution.csv", "state_matching.csv.gz"):
        shutil.copy2(PREFLIGHT / name, OUT / name)
    (OUT / "source_manifest.json").write_text(json.dumps(
        dict(batch=BATCH, old_reference=REFERENCE.name, original_commit=reference_launch["git_commit"],
             new_commit=read(RUNS["Gap-Matched-NonTop1"] / "launch.json")["git_commit"],
             preflight=preflight), ensure_ascii=False, indent=2) + "\n")
    print("verified", len(runs), "runs,", len(stages), "stage rows,", len(pools), "pool rows")


if __name__ == "__main__":
    main()
