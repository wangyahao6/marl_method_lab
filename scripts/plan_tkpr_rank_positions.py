"""Preview the TKPR rank-position study using the archived S prefixes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "configs/continual/action_preference/szc/szc_vdn64_tkpr_rank_position.json"
PARENTS = {seed: f"szc_vdn64_tkpr_k1_shared_20261117_S_seed{seed}_attempt001" for seed in (2222, 3535)}
METHODS = {"topk_preference": "TKPR-K1", "tkpr_r2": "TKPR-R2", "tkpr_r3": "TKPR-R3"}


def plan(batch_id):
    tasks = []
    for seed, parent_id in PARENTS.items():
        parent = ROOT / "runs" / parent_id
        for method, label in METHODS.items():
            run_id = f"{batch_id}_{label.lower().replace('-', '_')}_seed{seed}"
            argv = [sys.executable, "scripts/run.py", "--suite", "vdn_action_set", "--algorithm", "vdn",
                    "--protocol", PROTOCOL, "--seed", str(seed), "--method", method, "--run-id", run_id,
                    "--preference-role", "shared-branch", "--parent", str(parent)]
            preview = json.loads(subprocess.check_output([*argv, "--dry-run"], cwd=ROOT, text=True))
            tasks.append(dict(id=run_id, type="training", argv=argv, cwd=str(ROOT), run_dir=preview["output"],
                              dependencies=[parent_id], metadata={"method": label, "seed": seed,
                              "role": "shared-branch", "parent_run_id": parent_id,
                              "branch_id": f"{label}-seed{seed}", "branch_title": f"{label} · {seed}",
                              "attempt": 1}))
    return dict(id=batch_id, tasks=tasks, metadata={
        "experiment_group_id": batch_id, "experiment_group_title": "SZC TKPR 单排名消融"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id")
    args = parser.parse_args()
    print(json.dumps(plan(args.batch_id), indent=2))
