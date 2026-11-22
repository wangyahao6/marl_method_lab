"""Preview two SZC rank-distance ablations with one shared reference per seed."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "drivers/continual"))
from action_set_protocol import PAIR_ABLATIONS


def build_plan(batch_id):
    tasks = []
    for seed in (2222, 3535):
        reference_id = f"{batch_id}_reference_seed{seed}"
        parent = None
        for experiment in ("pair_distance", "pair_gap_far_mid"):
            protocol = Path("configs/continual/action_preference") / experiment / f"szc_vdn64_{experiment}_1m.json"
            methods = (("full_rank_preference",) + tuple(PAIR_ABLATIONS[experiment])
                       if experiment == "pair_distance" else tuple(PAIR_ABLATIONS[experiment]))
            for method in methods:
                reference = method == "full_rank_preference"
                run_id = reference_id if reference else f"{batch_id}_{method}_seed{seed}"
                argv = [sys.executable, "scripts/run.py", "--suite", "vdn_action_set",
                        "--algorithm", "vdn", "--protocol", str(protocol),
                        "--seed", str(seed), "--method", method, "--run-id", run_id,
                        "--preference-role", "pair-reference" if reference else "shared-branch"]
                if not reference:
                    argv.extend(("--parent", parent))
                preview = subprocess.check_output([*argv, "--dry-run"], cwd=ROOT, text=True)
                run_dir = json.loads(preview)["output"]
                tasks.append(dict(id=run_id, argv=argv, cwd=str(ROOT), type="training",
                                  run_dir=run_dir, dependencies=[] if reference else [reference_id]))
                if reference:
                    parent = run_dir
    return dict(id=batch_id, tasks=tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id")
    args = parser.parse_args()
    print(json.dumps(build_plan(args.batch_id), indent=2))
