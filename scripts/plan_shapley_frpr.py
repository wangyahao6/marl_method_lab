"""Freeze a Shapley-FRPR behavioral pilot and prepare its scheduler plan."""
import argparse
from pathlib import Path
import sys

from verify_smacv2_replay import ROOT, read, write


def main(args):
    config = read(args.config)
    output = args.output.resolve()
    tasks = []
    for case, overrides in config["cases"].items():
        settings = dict(config["common"], **overrides)
        for key in ("old_checkpoint", "ft_checkpoint", "inventory_pool"):
            if not (ROOT / settings[key]).is_file():
                raise FileNotFoundError(settings[key])
        case_config = output / "configs" / f"{case}.json"
        task = dict(id=f"{output.name}_{case}", cwd=str(ROOT), type="evaluation",
                    dependencies=[args.smoke_task] if args.smoke_task else [], resources=dict(ram_mib=2000),
                    metadata=dict(results=str(output), branch_id="shapley-frpr-pilot", attempt=args.attempt),
                    argv=[sys.executable, str(ROOT / "scripts/analyze_shapley_frpr.py"),
                          "--config", str(case_config), "--output", str(output / case)])
        tasks.append((settings, case_config, task))
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    for settings, case_config, _ in tasks:
        write(case_config, settings)
    write(output / "plan.json", dict(id=output.name, tasks=[task for _, _, task in tasks]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-task", help="Registered true-environment smoke task required before dispatch")
    parser.add_argument("--attempt", type=int, default=1)
    main(parser.parse_args())
