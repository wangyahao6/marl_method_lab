"""Preview one Gap-Matched-NonTop1 branch from the archived seed-3535 reference."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ID = "szc_pair_matched_3535_20261119_reference"
REFERENCE = ROOT / "runs/pair_structure" / REFERENCE_ID
PROTOCOL = Path("configs/continual/action_preference/pair_gap_matched/szc_vdn64_pair_gap_matched_1m.json")
METHOD = "pair_gap_matched_non_top1"


def build_plan(batch_id):
    if not (REFERENCE / "return_state.th").is_file():
        raise FileNotFoundError("archived S fork state is unavailable")
    run_id = batch_id + "_" + METHOD
    argv = [sys.executable, "scripts/run.py", "--suite", "vdn_action_set", "--algorithm", "vdn",
            "--protocol", str(PROTOCOL), "--seed", "3535", "--method", METHOD,
            "--run-id", run_id, "--preference-role", "shared-branch", "--parent", str(REFERENCE)]
    preview = json.loads(subprocess.check_output([*argv, "--dry-run"], cwd=ROOT, text=True))
    task = dict(id=run_id, argv=argv, cwd=str(ROOT), type="training", run_dir=preview["output"],
                dependencies=[REFERENCE_ID])
    return dict(id=batch_id, tasks=[task])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id")
    args = parser.parse_args()
    print(json.dumps(build_plan(args.batch_id), indent=2))
