"""Time matched TD-only and TD+TKPR-K1 updates without launching SC2."""
import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        parser.error("positive warmup and at least two measured pairs are required")

    root = Path(__file__).resolve().parents[1]
    plan = json.loads(args.plan.read_text())
    runs = [Path(task["run_dir"]) for task in plan["tasks"]
            if task.get("metadata", {}).get("capture_timing")]
    protocol = root / "configs/continual/szc_vdn64_tkpr_k1_shared.json"
    os.environ["MARL_PROTOCOL"] = str(protocol)
    sys.path.insert(0, str(root / "drivers/continual"))
    import action_set_train as driver
    import numpy as np
    import torch as th
    from common.preference_timing import paired_updates
    from learners.ranking_preference import RankingPreference

    if not th.cuda.is_available():
        raise RuntimeError("the timing experiment requires the training GPU")
    th.set_num_threads(1)
    rows = []
    for run in runs:
        paths = sorted((run / "timing_snapshots").glob("phase*_quarter*.th"))
        if not paths:
            logging.warning("Timing snapshots unavailable for %s", run.name)
            continue
        cfg = json.loads((run / "config.json").read_text())
        protocol = json.loads((run / "protocol.json").read_text())
        spec = driver.episode_spec(json.loads((run / "env_info.json").read_text()))
        scheme, groups = driver.augmented_scheme(spec), spec[1]
        runtime = SimpleNamespace(**dict(cfg, use_cuda=True, device="cuda:0"))
        mac = driver.base.controllers[runtime.mac](scheme, groups, runtime)
        logger = driver.base.Logger(logging.getLogger("tkpr_timing"))
        learner = driver.base.learners[runtime.learner](mac, scheme, logger, runtime)
        for network in (mac.agent, learner.target_mac.agent):
            for module in network.modules():
                module._forward_hooks.clear()
        learner.cuda()
        settings = dict(protocol["topk_preference"], batch_episodes=protocol["action_set"]["batch_episodes"])
        preference = RankingPreference("topk_preference", settings, np.random.default_rng(cfg["seed"]),
                                       scheme, groups, lambda row: None)
        parent = Path(cfg["parent"])
        preference.history.restore_success(parent, (0,), require_old_q=True)
        for path in paths:
            snapshot = th.load(path, map_location="cpu")
            if snapshot["phase"] == 2 and 1 not in preference.history:
                preference.history.restore_success(run, (1,), require_old_q=True)
            fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
            context = dict(run_id=run.name, seed=cfg["seed"], snapshot_sha256=fingerprint,
                           **{key: snapshot[key] for key in ("phase", "task", "quarter", "t_env", "update",
                                                          "batch_size", "max_seq_length")})
            rows.extend(dict(context, **row) for row in paired_updates(
                learner, snapshot, preference, scheme, groups, args.warmup, args.repeats))
    if not rows:
        raise RuntimeError("no matched learner updates could be timed")
    output = args.plan.with_name("timing_samples.csv")
    with output.with_suffix(".tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".tmp").replace(output)
    args.plan.with_name("timing_protocol.json").write_text(json.dumps(dict(
        gpu=th.cuda.get_device_name(), torch=th.__version__, warmup_pairs_per_snapshot=args.warmup,
        measured_pairs_per_snapshot=args.repeats, sample_count=len(rows)), indent=2) + "\n")


if __name__ == "__main__":
    main()
