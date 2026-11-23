"""Build and audit fixed masks against the archived K1 and Ratio-0 histories."""
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch as th
from torch.nn.utils.rnn import pad_sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learners.pair_ablation import select_pairs
from learners.pair_gap_matched import select_gap_matched_pairs
from learners.ranking_preference import reference_order


BATCH = "szc_pair_gapmatched_3535_20261123"
OLD = "szc_pair_matched_3535_20261119"
REFERENCE = ROOT / "runs/pair_structure" / (OLD + "_reference")
K1 = ROOT / "runs/pair_structure" / (OLD + "_pair_structure_k1")
RATIO0 = ROOT / "runs/pair_ratio" / (OLD + "_pair_ratio_0")
OUT = ROOT / "results" / BATCH / "preflight"


def read(path):
    return json.loads(path.read_text())


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, *, zipped=False):
    rows = list(rows)
    if not rows:
        raise ValueError("empty table " + str(path))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    opening = (lambda: gzip.open(path, "wt", newline="")) if zipped else (lambda: path.open("w", newline=""))
    with opening() as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def decision_data(episodes):
    data = {key: pad_sequence([ep["data"][key] for ep in episodes], batch_first=True)
            for key in ("old_q_values", "avail_actions", "filled", "terminated")}
    old = data["old_q_values"][:, :-1]
    legal = data["avail_actions"][:, :-1].bool()
    filled = data["filled"].bool()
    terminal = data["terminated"][:, :-1].bool()
    after = th.cat((th.zeros_like(terminal[:, :1]), terminal[:, :-1]), dim=1)
    decisions = filled[:, :-1] & filled[:, 1:] & (after.cumsum(dim=1) == 0)
    return old, legal, decisions & legal[..., 1:].any(dim=-1)


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(count=0, mean=None, std=None, median=None, p25=None, p50=None, p75=None,
                    p90=None, min=None, max=None)
    return dict(count=len(a), mean=float(a.mean()), std=float(a.std()),
                median=float(np.median(a)), p25=float(np.quantile(a, .25)),
                p50=float(np.quantile(a, .5)), p75=float(np.quantile(a, .75)),
                p90=float(np.quantile(a, .9)), min=float(a.min()), max=float(a.max()))


def verify_archived(name, path, reconstructed, valid, episodes, expected_hash):
    archived = th.load(path, map_location="cpu")
    if len(archived) != len(episodes):
        raise ValueError("archived mask length mismatch for " + name)
    mask_hash, valid_hash = hashlib.sha256(), hashlib.sha256()
    for episode, archive, calculated, active in zip(episodes, archived, reconstructed, valid):
        length = episode["length"]
        if not th.equal(archive, calculated[:length]):
            raise ValueError("archived " + name + " mask differs from source rebuild")
        mask_hash.update(archive.numpy().tobytes())
        valid_hash.update(active[:length].numpy().tobytes())
    if valid_hash.hexdigest() != expected_hash:
        raise ValueError("valid-state SHA differs from archived " + name)
    return archived, mask_hash.hexdigest(), valid_hash.hexdigest()


def inspect_phase(phase):
    historical = ("stalker", "zealot")[phase]
    source = REFERENCE / "action_set_memory" / f"phase{phase:04d}" / "success.th"
    episodes = th.load(source, map_location="cpu")["episodes"]
    old, legal, decisions = decision_data(episodes)
    matched, valid, counts, matching = select_gap_matched_pairs(
        old, legal, decisions, "gap_matched_non_top1", 3535, min_gap=0, q_scale_floor=.1)
    k1, k1_valid, _, _ = select_pairs(old, legal, decisions, "k1", 3535, min_gap=0,
                                      q_scale_floor=.1)
    ratio0, ratio_valid, _, _ = select_pairs(old, legal, decisions, 0, 3535, min_gap=0,
                                             q_scale_floor=.1)
    if not th.equal(valid, k1_valid) or not th.equal(valid, ratio_valid):
        raise ValueError("new mask does not share K1 and Ratio-0 valid states")
    archives = {}
    for name, run, selected in (("K1", K1, k1), ("Ratio-0", RATIO0, ratio0)):
        summary = read(run / "pair_ablation_memory" / f"phase{phase:04d}" / "summary.json")
        archives[name] = verify_archived(
            name, run / "pair_ablation_memory" / f"phase{phase:04d}" / "selection.th",
            selected, valid, episodes, summary["valid_state_sha256"])
        if summary["source_pool"] != str(source.resolve()):
            raise ValueError("old history source differs")
    if archives["K1"][2] != archives["Ratio-0"][2]:
        raise ValueError("old groups have distinct valid states")
    new_hash = hashlib.sha256()
    for episode, row in zip(episodes, matched):
        new_hash.update(row[:episode["length"]].numpy().tobytes())
    OUT.mkdir(parents=True, exist_ok=True)
    th.save([row[:episode["length"]].clone() for episode, row in zip(episodes, matched)],
            OUT / f"selection_phase{phase:04d}.th")

    size = legal.size(-1)
    left, right = th.triu_indices(size, size, 1)
    legal_pairs = legal[..., left] & legal[..., right]
    distinct = old[..., left] != old[..., right]
    eligible = legal_pairs & distinct & valid[..., None]
    standardized, _, _, _ = reference_order(old, legal, .1)
    gap = (standardized[..., left] - standardized[..., right]).abs()
    order = th.sort(old.masked_fill(~legal, -th.inf), dim=-1, stable=True,
                    descending=True).indices
    ranks = order.argsort(dim=-1)
    distance = (ranks[..., left] - ranks[..., right]).abs()
    top_pairs = (left == order[..., :1]) | (right == order[..., :1])
    if ((matched & top_pairs).any() or (matched & ~eligible).any()
            or (matched.sum(dim=-1)[valid] != (legal.sum(dim=-1) - 1)[valid]).any()):
        raise ValueError("new mask violates Top-1 exclusion, eligibility, or exact budget")
    selections = {"K1": k1, "Ratio-0": ratio0, "Gap-Matched-NonTop1": matched,
                  "all_non_top1_candidates": eligible & ~top_pairs}
    gaps, rank_rows, states = [], [], []
    for name, mask in selections.items():
        values = gap[mask].numpy()
        gaps.append(dict(phase=phase, historical_task=historical, selection=name,
                         **distribution(values)))
        d = distance[mask].numpy()
        for rank in sorted(np.unique(d)):
            subset = values[d == rank]
            rank_rows.append(dict(phase=phase, historical_task=historical, selection=name,
                                  rank_distance=int(rank), count=len(subset),
                                  share=len(subset) / len(values), mean_gap=float(subset.mean()),
                                  median_gap=float(np.median(subset))))
    for episode_index, episode in enumerate(episodes):
        for time in range(episode["length"]):
            for agent in range(old.size(-2)):
                if not valid[episode_index, time, agent]:
                    continue
                target = np.sort(gap[episode_index, time, agent, k1[episode_index, time, agent]].numpy())
                chosen = np.sort(gap[episode_index, time, agent, matched[episode_index, time, agent]].numpy())
                random = np.sort(gap[episode_index, time, agent, ratio0[episode_index, time, agent]].numpy())
                all_candidates = gap[episode_index, time, agent,
                                     eligible[episode_index, time, agent]
                                     & ~top_pairs[episode_index, time, agent]].numpy()
                budget = len(target)
                w1 = float(np.abs(target - chosen).mean())
                random_w1 = float(np.abs(target - random).mean())
                target_mean = float(target.mean())
                states.append(dict(phase=phase, historical_task=historical, episode_index=episode_index,
                                   timestep=time, agent_id=agent, legal_actions=budget + 1,
                                   target_mean_gap=target_mean, matched_mean_gap=float(chosen.mean()),
                                   ratio0_mean_gap=float(random.mean()), matched_w1=w1, ratio0_w1=random_w1,
                                   target_max_gap=float(target.max()),
                                   non_top_candidate_max_gap=float(all_candidates.max()),
                                   max_gap_shortfall=max(0., float(target.max() - all_candidates.max())),
                                   matched_relative_error=w1 / target_mean,
                                   ratio0_relative_error=random_w1 / target_mean))
    if len(states) != int(valid.sum()):
        raise ValueError("state-level matching table is incomplete")
    summary = dict(phase=phase, historical_task=historical, source_pool=str(source.relative_to(ROOT)),
                   source_sha256=file_sha256(source),
                   valid_states=int(valid.sum()), decision_states=int(decisions.sum()),
                   valid_state_sha256=archives["K1"][2],
                   k1_mask_sha256=archives["K1"][1], ratio0_mask_sha256=archives["Ratio-0"][1],
                   matched_mask_sha256=new_hash.hexdigest(),
                   selected_pairs=int(matched.sum()), top1_pairs=int((matched & top_pairs).sum()),
                   matched_w1=distribution([row["matched_w1"] for row in states]),
                   ratio0_w1=distribution([row["ratio0_w1"] for row in states]),
                   matched_relative_error=distribution([row["matched_relative_error"] for row in states]),
                   max_gap_shortfall=distribution([row["max_gap_shortfall"] for row in states]),
                   states_target_exceeds_candidate_max=sum(row["max_gap_shortfall"] > 0 for row in states),
                   exact_match_states=sum(row["matched_w1"] < 1e-8 for row in states),
                   optimizer_audit=matching)
    return gaps, rank_rows, states, summary


def main():
    all_gap, all_rank, all_states, summaries = [], [], [], []
    for phase in (0, 1):
        gaps, ranks, states, summary = inspect_phase(phase)
        all_gap.extend(gaps)
        all_rank.extend(ranks)
        all_states.extend(states)
        summaries.append(summary)
    write_csv(OUT / "gap_distribution.csv", all_gap)
    write_csv(OUT / "rank_distance_distribution.csv", all_rank)
    write_csv(OUT / "state_matching.csv.gz", all_states, zipped=True)
    (OUT / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
    for row in summaries:
        print(row["historical_task"], "states", row["valid_states"], "pairs", row["selected_pairs"],
              "W1", row["matched_w1"]["mean"], "relative P90", row["matched_relative_error"]["p90"])


if __name__ == "__main__":
    main()
