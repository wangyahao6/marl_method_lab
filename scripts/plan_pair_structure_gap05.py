"""Preview the SZC pair structure gap-0.5 shared-reference batch."""
import argparse
import json

from plan_pair_ablations import build_plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id")
    args = parser.parse_args()
    print(json.dumps(build_plan(args.batch_id, ("pair_structure_gap05",)), indent=2))
