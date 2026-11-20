"""Freeze the case configs and scheduler plan for the Shapley validation study."""
import argparse
from pathlib import Path
import sys

from verify_smacv2_replay import ROOT, read, write


def plan(args):
    protocol = read(args.protocol)
    pilot = ROOT / protocol["pilot_root"]
    original = read(ROOT / protocol["pilot_config"])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", protocol)
    tasks = []
    for name, settings in original["cases"].items():
        if args.smoke and name != "2222_s_after_z":
            continue
        env = dict(original["common"], **settings)
        original_seeds = [pair[0] for pair in env["seeds"]]
        discovery = [91101] if args.smoke else original_seeds
        held_out = [91103] if args.smoke else protocol["validation_env_seeds"][env["task"]]
        pilot_seeds = [91101, 91102] if args.smoke else original_seeds
        if args.smoke:
            cell = args.smoke_cell
            discovery_quota, validation_quota = {cell: 1}, {"early/low": 1}
        else:
            discovery_quota = protocol["discovery_per_stratum"]
            validation_quota = protocol["validation_per_stratum"]
        config = {key: protocol[key] for key in ("top_k", "max_exact_pairs", "projection_margin",
                                                "softmax_temperature", "random_seeds",
                                                "stratum_shrinkage", "monte_carlo")}
        config.update(environment=env, pilot=str(pilot / name), sampling_seed=env["sampling_seed"],
                      discovery_seeds=[[s, s] for s in discovery], validation_seeds=[[s, s] for s in held_out],
                      pilot_discovery_env_seeds=pilot_seeds, discovery_per_stratum=discovery_quota,
                      validation_per_stratum=validation_quota)
        config_path = out / f"{name}.json"
        write(config_path, config)
        tasks.append(dict(id=f"{out.name}_{name}", cwd=str(ROOT), type="evaluation",
                          dependencies=[args.after] if args.after else [],
                          resources=dict(ram_mib=2000), metadata=dict(results=str(out)),
                          argv=[sys.executable, str(ROOT / "scripts/validate_shapley_frpr.py"),
                                "--config", str(config_path), "--output", str(out / name)]))
    write(out / "plan.json", dict(id=out.name, tasks=tasks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-cell", default="early/low")
    parser.add_argument("--after")
    plan(parser.parse_args())
