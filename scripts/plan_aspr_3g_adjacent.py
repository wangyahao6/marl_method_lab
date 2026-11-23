"""Prepare four authorized SZC 2M-stage branches sharing the completed TKPR S prefixes."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_plan(batch):
    tasks = []
    for seed in (2222, 3535):
        parent_id = f"szc_vdn64_tkpr_k1_shared_20261117_S_seed{seed}_attempt001"
        for method, config, label in (
            ("aspr_3g_adjacent_preference", "szc_vdn64_aspr_3g_adjacent_gap0.json", "ASPR-3G-Adjacent"),
            ("tkpr_k1_gap0", "szc_vdn64_tkpr_k1_gap0.json", "TKPR-K1"),
        ):
            run_id = f"{batch}_{method}_seed{seed}_attempt001"
            argv = [sys.executable, "scripts/run.py", "--suite", "vdn_action_set", "--algorithm", "vdn",
                    "--method", method, "--protocol", f"configs/continual/action_preference/szc/{config}",
                    "--seed", str(seed), "--run-id", run_id, "--preference-role", "shared-branch",
                    "--parent", str(ROOT / "runs" / parent_id)]
            preview = json.loads(subprocess.check_output([*argv, "--dry-run"], cwd=ROOT, text=True))
            tasks.append(dict(id=run_id, argv=argv, cwd=str(ROOT), type="training",
                run_dir=preview["output"], dependencies=[parent_id], queue=batch, queue_limit=4,
                resources=dict(gpu=0, gpu_mib=2000, ram_mib=6500),
                metadata=dict(experiment_group_id=batch, experiment_group_title="SZC ASPR-3G-Adjacent vs K1 gap0",
                    branch_id=method, branch_title=label, seed=seed, method=label, attempt=1,
                    parent_run_id=parent_id, results=str(ROOT / "results" / batch))))
    training = list(tasks)
    for task in training:
        for suffix, argv, resources in (
            ("independent_final", [sys.executable, "scripts/evaluate.py", "--run", task["run_dir"],
                                    "--episodes", "128"], dict(gpu=0, gpu_mib=800, ram_mib=2500)),
            ("offline_audit", [sys.executable, "scripts/audit_aspr_adjacent_offline.py", task["run_dir"]],
                               dict(gpu_mib=0, ram_mib=2000)),
        ):
            tasks.append(dict(id=task["id"] + "_" + suffix, argv=argv, cwd=str(ROOT), type="evaluation",
                dependencies=[task["id"]], queue=batch, queue_limit=4,
                resources=resources, metadata=task["metadata"]))
    tasks.append(dict(id=batch + "_final_summary", argv=[sys.executable,
        "scripts/summarize_aspr_3g_adjacent.py", str(ROOT / "results" / batch / "plan.json")],
        cwd=str(ROOT), type="summary", dependencies=[task["id"] for task in tasks],
        dependency_mode="terminal", queue=batch, resources=dict(gpu_mib=0, ram_mib=1000)))
    return dict(id=batch, tasks=tasks)


if __name__ == "__main__":
    print(json.dumps(build_plan(sys.argv[1]), indent=2))
