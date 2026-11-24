"""Prepare exactly four new branches plus evaluation/diagnostics for the ASPR 3G 2x2 gap ablation."""
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("configs/continual/action_preference/szc")
CELLS = (
    ("ASPR-3G", "aspr_3g_preference", 0., "szc_vdn64_aspr_3g_gap0_from_tkpr_s.json", True),
    ("ASPR-3G", "aspr_3g_preference", .5, "szc_vdn64_aspr_3g_from_tkpr_s.json", False),
    ("ASPR-3G-Adjacent", "aspr_3g_adjacent_preference", 0., "szc_vdn64_aspr_3g_adjacent_gap0.json", False),
    ("ASPR-3G-Adjacent", "aspr_3g_adjacent_preference", .5, "szc_vdn64_aspr_3g_adjacent_gap05.json", True),
)


def build_plan(batch):
    tasks, runs = [], []
    for seed in (2222, 3535):
        parent_id = f"szc_vdn64_tkpr_k1_shared_20261117_S_seed{seed}_attempt001"
        parent = ROOT / "runs" / parent_id
        for label, method, gap, filename, new in CELLS:
            branch = ("3g" if label == "ASPR-3G" else "adjacent") + ("_gap0" if gap == 0 else "_gap05")
            if new:
                run_id = f"{batch}_{branch}_seed{seed}_attempt001"
            elif label == "ASPR-3G":
                run_id = f"szc_vdn64_aspr_3g_shared_20261121_seed{seed}_attempt001"
            else:
                run_id = f"szc_aspr_3g_adjacent_20261123_aspr_3g_adjacent_preference_seed{seed}_attempt001"
            run = ROOT / "runs" / run_id
            metadata = dict(experiment_group_id=batch, experiment_group_title="SZC ASPR 3G min_gap 2x2",
                branch_id=branch, branch_title=f"{label} gap={gap:g}", method=label, min_gap=gap,
                seed=seed, attempt=1, parent_run_id=parent_id, results=str(ROOT / "results" / batch))
            identity = dict(run_id=run_id, run_dir=str(run), parent_run_id=parent_id,
                parent=str(parent), method=label, engine_method=method, min_gap=gap, seed=seed,
                role="shared-branch", newly_trained=new, protocol=str(CONFIG / filename))
            runs.append(identity)
            if new:
                argv = [sys.executable, "scripts/run.py", "--suite", "vdn_action_set", "--algorithm", "vdn",
                    "--method", method, "--seed", str(seed), "--run-id", run_id, "--protocol", str(CONFIG / filename),
                    "--preference-role", "shared-branch", "--parent", str(parent)]
                preview = json.loads(subprocess.check_output([*argv, "--dry-run"], cwd=ROOT, universal_newlines=True))
                if preview["output"] != str(run):
                    raise ValueError("Run output does not match the isolated ablation directory")
                tasks.append(dict(id=run_id, argv=argv, cwd=str(ROOT), type="training", run_dir=str(run),
                    dependencies=[parent_id], queue=batch, queue_limit=4,
                    resources=dict(gpu=0, gpu_mib=2000, ram_mib=6500), metadata=metadata))
                tasks.append(dict(id=run_id+"_independent_final", argv=[sys.executable,"scripts/evaluate.py",
                    "--run",str(run),"--episodes","128"], cwd=str(ROOT), type="evaluation",
                    dependencies=[run_id],queue=batch,queue_limit=4,
                    resources=dict(gpu=0,gpu_mib=800,ram_mib=2500),metadata=metadata))
            tasks.append(dict(id=f"{batch}_{branch}_seed{seed}_diagnostics", argv=[sys.executable,
                "scripts/diagnose_aspr_group_gap.py", str(run), "--output",
                str(ROOT / "results" / batch / "diagnostics" / run_id)], cwd=str(ROOT), type="evaluation",
                dependencies=[run_id] if new else [],queue=batch,queue_limit=4,
                resources=dict(gpu_mib=0,ram_mib=2000),metadata=metadata))
    tasks.append(dict(id=batch+"_summary",argv=[sys.executable,"scripts/summarize_aspr_group_gap_ablation.py",
        str(ROOT/"results"/batch/"plan.json")],cwd=str(ROOT),type="summary",
        dependencies=[t["id"] for t in tasks],dependency_mode="terminal",queue=batch,
        resources=dict(gpu_mib=0,ram_mib=1500)))
    return dict(id=batch,runs=runs,tasks=tasks)


if __name__ == "__main__":
    print(json.dumps(build_plan(sys.argv[1]),indent=2))
