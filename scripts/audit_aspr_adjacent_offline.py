"""Read saved policies and all historical success episodes; compute cross-group PVR without training."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "drivers/continual")]
import torch as th
from controllers import REGISTRY
from components.episode_buffer import ReplayBuffer
from common.runtime import episode_spec
from learners.action_set_preference import batch_episodes, decision_masks, unroll
from learners.ranking_preference import group_old_q, aspr_adjacent_preference, aspr_adjacent_statistics


def audit(run):
    th.set_num_threads(1)
    args = SimpleNamespace(**json.loads((run / "config.json").read_text()))
    args.device, args.use_cuda = "cpu", False
    info = json.loads((run / "env_info.json").read_text())
    scheme, groups, preprocess = episode_spec(info)
    augmented = ReplayBuffer(scheme, groups, 1, 2, preprocess=preprocess).scheme
    mac = REGISTRY[args.mac](augmented, groups, args)
    retention = json.loads((run / "retention.json").read_text())
    settings = dict(weight=1., temperature=1., min_gap=0., q_scale_floor=.1, group_count=3)
    pools = {}
    for path in sorted((run / "action_set_memory").glob("phase*/success.th")):
        pool = th.load(path, map_location="cpu")
        labeled = []
        for ep in pool["episodes"]:
            batch, _ = batch_episodes([ep], augmented, groups, "cpu", include_good=False, include_old_q=True)
            legal, decisions = decision_masks(batch)
            ids = group_old_q(batch["old_q_values"][:, :-1], legal, decisions, 3, .1)[0]
            labeled.append((ep, ids))
        pools[pool["phase"]] = (pool, labeled)
    rows = []
    with th.no_grad():
        for stage in retention["stages"]:
            phase = stage["phase"]
            if phase == 0:
                continue
            checkpoint = run / stage["checkpoint"]
            mac.agent.load_state_dict(th.load(checkpoint / "agent.th", map_location="cpu"))
            for source, (pool, episodes) in pools.items():
                if source >= phase:
                    continue
                counts = {name: Counter() for name in ("G1-G2", "G2-G3", "G1-G3")}
                for ep, ids in episodes:
                    batch, _ = batch_episodes([ep], augmented, groups, "cpu", include_good=False, include_old_q=True)
                    legal, decisions = decision_masks(batch)
                    current = unroll(mac, batch, batch.max_seq_length - 1)
                    _, _, summary = aspr_adjacent_preference(current, batch["old_q_values"][:, :-1],
                                                              legal, decisions, settings, group_ids=ids.unsqueeze(0))
                    stats = aspr_adjacent_statistics(summary.tolist())
                    for name, values in stats["group_pairs"].items():
                        counts[name].update({k: values[k] for k in
                            ("candidate_pairs", "eligible_pairs", "valid_pairs", "pair_violations")})
                for pair, values in counts.items():
                    rows.append(dict(phase=phase, training_task=stage["task"], t_env=stage["t_env"],
                        source_phase=source, historical_task=pool["task"], pair=pair,
                        episodes=len(episodes), checkpoint=str(checkpoint), **values,
                        pair_violation_rate=values["pair_violations"] / values["valid_pairs"] if values["valid_pairs"] else None))
    path = run / "aspr_cross_group_offline.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]) if rows else ["phase", "pair"])
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    audit(parser.parse_args().run)
