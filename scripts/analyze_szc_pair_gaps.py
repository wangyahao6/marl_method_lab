"""Analyze the saved SZC-1M pair masks in the trusted launch snapshots."""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from types import ModuleType

import numpy as np
import torch as th
from torch.nn.utils.rnn import pad_sequence


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/results/parameter-ablations/szc-vdn64-pair-structure-ratio"
METHODS = ("TKPR-K1", "Adjacent", "TKPR-K1 + Adjacent", "FRPR",
           "Random-N-1", "Random-(2N-3)", "Random-(2N-3)-Matched")
MODULES = ("action_set_preference", "ranking_preference", "pair_ablation")
TASKS = ("stalker", "zealot", "colossus")
CATEGORIES = ("all", "top1", "non_top1")


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tensor_digest(tensors):
    state = hashlib.sha256()
    for tensor in tensors:
        state.update(tensor.numpy().tobytes())
    return state.hexdigest()


@contextmanager
def archived_learners(run, hashes):
    """Resolve each preference module from its launch tar, in dependency order."""
    previous, modules = {}, {}
    with tarfile.open(run / "source.tar.gz", "r:gz") as archive:
        try:
            for name in MODULES:
                relative = f"src/learners/{name}.py"
                with archive.extractfile(relative) as stream:
                    source = stream.read()
                if digest(source) != hashes[relative]:
                    raise ValueError(f"{run.name}: launch hash mismatch for {relative}")
                qualified = f"learners.{name}"
                previous[qualified] = sys.modules.get(qualified)
                module = ModuleType(qualified)
                sys.modules[qualified] = module
                exec(compile(source, f"{run.name}:{relative}", "exec"), module.__dict__)
                modules[name] = module
            yield modules
        finally:
            for name, module in previous.items():
                if module is None:
                    del sys.modules[name]
                else:
                    sys.modules[name] = module


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stages(source_phase):
    return (1, 2) if source_phase == 0 else (2,)


def gap_statistics(values):
    fields = ("mean", "std", "median", "p25", "p75", "p90", "p95", "p99", "min", "max")
    if not len(values):
        return dict.fromkeys(fields)
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, [.5, .25, .75, .9, .95, .99])
    return dict(zip(fields, (values.mean(), values.std(), *quantiles, values.min(), values.max())))


def state_values(values, mask):
    present = mask.any(axis=1)
    selected = values[present]
    kept = mask[present]
    means = np.where(kept, selected, 0).sum(axis=1) / kept.sum(axis=1)
    medians = np.nanmedian(np.where(kept, selected, np.nan), axis=1)
    return means, medians


def gap_rows(identity, gaps, mask):
    means, medians = state_values(gaps, mask)
    active = mask.any(axis=1)
    for name, values in (("pair", gaps[mask]), ("state_mean", means), ("state_median", medians)):
        loss = slope = None
        if name == "state_mean" and len(values):
            kept = mask[active]
            selected = np.where(kept, gaps[active], 0)
            count = kept.sum(axis=1)
            loss = (np.where(kept, np.logaddexp(0, -selected), 0).sum(axis=1) / count).mean()
            slope = (np.where(kept, 1 / (1 + np.exp(selected)), 0).sum(axis=1) / count).mean()
        yield dict(identity, aggregation=name, pairs=int(mask.sum()), states=len(means),
                   observations=len(values), **gap_statistics(values),
                   frozen_q_loss_mean_by_state=loss, frozen_q_logit_slope_mean_by_state=slope)


def characterize(gaps, distances, top_pairs, eligible, mask, method, phase, maximum):
    summary, by_distance, distribution = [], [], []
    selectors = (eligible, eligible & top_pairs, eligible & ~top_pairs)
    for category, candidates in zip(CATEGORIES, selectors):
        chosen = mask & candidates
        selected_per_state = chosen.sum(axis=1)
        active = selected_per_state > 0
        pool_pairs = int(selected_per_state.sum())
        common = dict(method=method, source_phase=phase, historical_task=TASKS[phase], pair_category=category)
        stats = list(gap_rows(common, gaps, chosen))
        for training_phase in stages(phase):
            identity = dict(training_phase=training_phase, training_task=TASKS[training_phase])
            summary.extend(dict(identity, **row) for row in stats)
        for distance in range(1, maximum + 1):
            at_distance = distances == distance
            subset = chosen & at_distance
            per_state = subset.sum(axis=1)
            count = int(per_state.sum())
            capacity = int((candidates & at_distance).sum())
            records = list(gap_rows(dict(common, rank_distance=distance), gaps, subset))
            for training_phase in stages(phase):
                identity = dict(training_phase=training_phase, training_task=TASKS[training_phase])
                by_distance.extend(dict(identity, **row) for row in records)
                distribution.append(dict(identity, **common, rank_distance=distance,
                                         eligible_pairs=capacity, selected_pairs=count,
                                         selected_pair_fraction=count / pool_pairs if pool_pairs else None,
                                         eligible_pair_coverage=count / capacity if capacity else None,
                                         category_active_states=int(active.sum()), states_with_distance=int((per_state > 0).sum()),
                                         mean_state_pair_fraction=(per_state[active] / selected_per_state[active]).mean()
                                         if active.any() else None))
    return summary, by_distance, distribution


def paired_rows(phase, episodes, coords, gaps, distances, top_pairs, legal_count, adjacent, matched):
    common = adjacent & matched
    union = adjacent | matched
    non_top = ~top_pairs
    a_non, m_non = adjacent & non_top, matched & non_top
    assert np.array_equal(adjacent.sum(axis=1), 2 * legal_count - 3)
    assert np.array_equal(adjacent.sum(axis=1), matched.sum(axis=1))
    assert np.array_equal(adjacent & top_pairs, matched & top_pairs)
    assert np.array_equal((matched & top_pairs).sum(axis=1), legal_count - 1)
    assert np.array_equal(a_non.sum(axis=1), legal_count - 2)
    a_mean, a_median = state_values(gaps, a_non)
    m_mean, m_median = state_values(gaps, m_non)
    a_all, _ = state_values(gaps, adjacent)
    m_all, _ = state_values(gaps, matched)
    common_all = common.sum(axis=1)
    union_all = union.sum(axis=1)
    common_non = (common & non_top).sum(axis=1)
    union_non = (union & non_top).sum(axis=1)
    counts = {f"{name}_non_top1_d{d}": (mask & (distances == d)).sum(axis=1)
              for name, mask in (("adjacent", a_non), ("matched", m_non))
              for d in range(1, int(distances.max()) + 1)}
    for i, (episode_index, t, agent) in enumerate(coords.tolist()):
        episode = episodes[episode_index]
        yield dict(source_phase=phase, historical_task=TASKS[phase],
                   training_phases="|".join(str(x) for x in stages(phase)),
                   episode_pool_index=episode_index, episode_seed=episode["seed"], episode_index=episode["index"],
                   decision_t=t, agent_index=agent, legal_actions=int(legal_count[i]),
                   selected_pairs=int(adjacent[i].sum()), selected_non_top1_pairs=int(a_non[i].sum()),
                   adjacent_gap_mean_all=a_all[i], matched_gap_mean_all=m_all[i],
                   delta_gap_mean_all=m_all[i] - a_all[i],
                   adjacent_gap_mean_non_top1=a_mean[i], matched_gap_mean_non_top1=m_mean[i],
                   delta_gap_mean_non_top1=m_mean[i] - a_mean[i],
                   adjacent_gap_median_non_top1=a_median[i], matched_gap_median_non_top1=m_median[i],
                   intersection_pairs=int(common_all[i]), union_pairs=int(union_all[i]),
                   intersection_fraction=common_all[i] / adjacent[i].sum(), jaccard=common_all[i] / union_all[i],
                   intersection_non_top1_pairs=int(common_non[i]), union_non_top1_pairs=int(union_non[i]),
                   intersection_non_top1_fraction=common_non[i] / a_non[i].sum(),
                   jaccard_non_top1=common_non[i] / union_non[i],
                   **{key: int(values[i]) for key, values in counts.items()})


def analyze(results, output):
    with (results / "runs.csv").open() as stream:
        runs = {row["method"]: row for row in csv.DictReader(stream)}
    parent = ROOT / runs["TD reference"]["run_dir"]
    summary, distances_out, distributions, paired, provenance = [], [], [], [], []
    for phase in range(2):
        path = parent / "action_set_memory" / f"phase{phase:04d}" / "success.th"
        pool_sha256 = digest(path.read_bytes())
        pool = th.load(path, map_location="cpu")
        episodes = pool["episodes"]
        if pool["task"] != TASKS[phase] or pool["phase"] != phase:
            raise ValueError(f"Unexpected history identity in {path}")
        batch = {key: pad_sequence([ep["data"][key] for ep in episodes], batch_first=True)
                 for key in ("old_q_values", "avail_actions", "filled", "terminated")}
        q = batch["old_q_values"][:, :-1]
        left, right = th.triu_indices(q.size(-1), q.size(-1), offset=1)
        common_states, shared_gaps = None, None
        selections = {}
        for method in METHODS:
            row = runs[method]
            run = ROOT / row["run_dir"]
            launch = read(run / "launch.json")
            snapshot = read(results / row["config"])
            protocol = read(run / "protocol.json")
            assert launch["git_commit"] == snapshot["launch_git_commit"] == row["commit"]
            assert not launch["git_status"] and protocol == snapshot["protocol"]
            cli = protocol["method"]
            settings = protocol[cli]
            assert settings["min_gap"] == 0
            with archived_learners(run, launch["source_sha256"]) as modules:
                legal, decisions = modules["action_set_preference"].decision_masks(batch)
                _, scale, best, _ = modules["ranking_preference"].reference_order(q, legal, settings["q_scale_floor"])
                rebuilt, valid, _, _ = modules["pair_ablation"].select_pairs(
                    q, legal, decisions, cli[len("pair_structure_"):], settings["pair_seed"])
            mask_dir = run / "pair_ablation_memory" / f"phase{phase:04d}"
            audit = read(mask_dir / "summary.json")
            assert Path(audit["source_pool"]).resolve() == path.resolve()
            assert audit["pair_seed"] == settings["pair_seed"]
            saved = th.load(mask_dir / "selection.th", map_location="cpu")
            assert len(saved) == len(episodes)
            for m, episode, recovered in zip(saved, episodes, rebuilt):
                assert m.dtype == th.bool and th.equal(m, recovered[:episode["length"]])
            assert tensor_digest(saved) == audit["pair_mask_sha256"]
            assert int(rebuilt.sum()) == audit["valid_pairs"] and int(valid.sum()) == audit["valid_states"]
            assert tensor_digest([m[:ep["length"]] for m, ep in zip(valid, episodes)]) == audit["valid_state_sha256"]
            if common_states is None:
                common_states = valid
                old = q[valid]
                available = legal[valid]
                order = old.masked_fill(~available, -th.inf).argsort(dim=-1, descending=True)
                rank = order.argsort(dim=-1)
                rank_distance = (rank[:, left] - rank[:, right]).abs().numpy()
                permitted = (available[:, left] & available[:, right]).numpy()
                assert not (permitted & (old[:, left] == old[:, right]).numpy()).any()
                legal_count = available.sum(-1).numpy()
                coords = valid.nonzero()
                maximum = q.size(-1) - 1
                top_pairs = ((left == best[valid]) | (right == best[valid])).numpy()
            assert th.equal(valid, common_states)
            gaps = ((q[valid][:, left] - q[valid][:, right]).abs() / scale[valid]).numpy()
            assert np.isfinite(gaps[permitted]).all()
            if shared_gaps is None:
                shared_gaps = gaps
            assert np.array_equal(gaps, shared_gaps)
            mask = rebuilt[valid].numpy()
            assert not (mask & ~permitted).any()
            assert int((mask & top_pairs).sum()) == audit["top1_pairs"]
            selections[method] = mask
            tables = characterize(gaps, rank_distance, top_pairs, permitted, mask, method, phase, maximum)
            for target, table in zip((summary, distances_out, distributions), tables):
                target.extend(table)
            provenance.append(dict(method=method, run_id=row["run_id"], launch_commit=row["commit"],
                                   source_phase=phase, historical_pool=str(path.relative_to(ROOT)),
                                   historical_pool_sha256=pool_sha256,
                                   source_sha256={key: launch["source_sha256"][key] for key in
                                                  [f"src/learners/{name}.py" for name in MODULES]},
                                   pair_seed=settings["pair_seed"], q_scale_floor=settings["q_scale_floor"],
                                   valid_state_sha256=audit["valid_state_sha256"],
                                   pair_mask_sha256=audit["pair_mask_sha256"], rebuilt_masks_match=True))
        # Launched environments all have 11 action slots. Retain the full distance axis in the paired CSV.
        paired.extend(paired_rows(phase, episodes, coords, shared_gaps, rank_distance, top_pairs, legal_count,
                                 selections["TKPR-K1 + Adjacent"], selections["Random-(2N-3)-Matched"]))
        print(f"{TASKS[phase]}: {len(episodes)} episodes, {len(coords)} common decisions, 7 verified masks", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("pair_gap_summary", summary), ("pair_gap_by_rank_distance", distances_out),
                       ("rank_distance_distribution", distributions), ("matched_vs_adjacent", paired)):
        write_csv(output / f"{name}.csv", rows)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    th.set_num_threads(1)
    analyze(args.results, args.output or args.results / "pair-gap-analysis")
