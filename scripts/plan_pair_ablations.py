"""Preview the shared SZC-1M reference and every registered ablation branch.

Print the resulting scheduler plan without submitting it.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = Path("configs/continual/action_preference")
sys.path.insert(0, str(ROOT / "drivers/continual"))
from action_set_protocol import PAIR_ABLATIONS


def build_plan(batch_id, experiments=("pair_structure", "pair_ratio")):
    common = [sys.executable, "scripts/run.py", "--suite", "vdn_action_set",
              "--algorithm", "vdn", "--seed", "3535"]
    reference_id = f"{batch_id}_reference"
    tasks = []

    def task(experiment, method, run_id, role, parent=None):
        protocol = CONFIGS / experiment / f"szc_vdn64_{experiment}_1m.json"
        argv = [*common, "--protocol", str(protocol), "--method", method,
                "--run-id", run_id, "--preference-role", role]
        if parent:
            argv += ["--parent", parent]
        preview = subprocess.check_output([*argv, "--dry-run"], cwd=ROOT, text=True)
        run_dir = json.loads(preview)["output"]
        tasks.append(dict(id=run_id, argv=argv, cwd=str(ROOT), type="training", run_dir=run_dir,
                          dependencies=[reference_id] if parent else []))
        return run_dir

    parent = task(experiments[0], "full_rank_preference", reference_id, "pair-reference")
    for experiment in experiments:
        methods = PAIR_ABLATIONS[experiment]
        for method in methods:
            task(experiment, method, f"{batch_id}_{method}", "shared-branch", parent)
    return dict(id=batch_id, tasks=tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id", help="Unique prefix for the scheduler task and run IDs")
    args = parser.parse_args()
    print(json.dumps(build_plan(args.batch_id), indent=2))
