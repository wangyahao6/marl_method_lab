"""Context-indexed selection over the original FRPR pair mask."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import torch as th
from torch.nn.utils.rnn import pad_sequence

from learners.action_set_preference import ActionSetPreference, batch_episodes, decision_masks, unroll
from learners.ranking_preference import full_rank_pairs, full_rank_preference, ranking_statistics, reference_order
from drivers.continual.memory.episodes import preference_episode_id


def pair_inventory(reference, legal, decisions, settings):
    old, scale, _, _ = reference_order(reference, legal, settings["q_scale_floor"])
    gaps, _, _, mask = full_rank_pairs(old, legal, decisions, settings)
    return gaps, mask, scale


def oriented_pairs(gaps, mask):
    return [(i, j) if gaps[i, j] > 0 else (j, i) for i, j in mask.nonzero().tolist()]


def selection_budget(settings, count):
    k = settings.get("selection_k")
    return min(k, count) if k is not None else math.ceil(settings["selection_ratio"] * count)


def choose_pairs(context, scores, settings, *, random_control):
    k = selection_budget(settings, len(scores))
    ordered = sorted(scores)
    if random_control:
        key = json.dumps([settings["selection_seed"], *context], separators=(",", ":")).encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big")).sample(ordered, k)
    return sorted(ordered, key=lambda pair: (-scores[pair], pair))[:k]


class SelectedFRPR(ActionSetPreference):
    statistics = staticmethod(ranking_statistics)

    def __init__(self, method, settings, rng, scheme, groups, log):
        super().__init__(settings, rng, scheme, groups, log)
        self.random_control = method == "random_selected_frpr"
        self.scores = {}
        for value in settings["shapley_results"]:
            path = Path(value)
            manifest = json.loads((path / "manifest.json").read_text())
            audit = json.loads((path / "audit.json").read_text())
            if not audit["passed"] or any(manifest["frpr"][key] != settings[key]
                                         for key in ("min_gap", "q_scale_floor")):
                raise ValueError(f"Invalid Shapley audit or FRPR participant settings in {path}")
            with (path / "shapley_values.csv").open(newline="") as stream:
                for row in csv.DictReader(stream):
                    key = row["task"], row["episode_id"], int(row["timestep"]), int(row["agent_id"])
                    pair = int(row["preferred_action"]), int(row["dispreferred_action"])
                    score = float(row["phi_return"])
                    scores = self.scores.setdefault(key, {})
                    if pair in scores or not math.isfinite(score):
                        raise ValueError(f"Duplicate or invalid Shapley score for {key}, {pair}")
                    scores[pair] = score

    def prepare_pool(self, pool):
        """Check every eligible decision in the pool before it can be sampled."""
        prepared, covered, required, examples = [], 0, 0, []
        seen = set()
        task = pool["task"]
        for episode in pool["episodes"]:
            batch, _ = batch_episodes([episode], self.scheme, self.groups, "cpu",
                                      include_good=False, include_old_q=True)
            episode_id = preference_episode_id(task, episode)
            if episode_id in seen or episode.get("episode_id", episode_id) != episode_id:
                raise ValueError(f"Duplicate or mismatched preference episode ID: {task}, {episode_id}")
            seen.add(episode_id)
            legal, decisions = decision_masks(batch)
            gaps, pairs, _ = pair_inventory(batch["old_q_values"][:, :-1], legal, decisions, self.settings)
            pairs, gaps = pairs[0], gaps[0]
            selected = th.zeros_like(pairs)
            for step, agent in pairs.flatten(-2).any(dim=-1).nonzero().tolist():
                context = task, episode_id, step, agent
                expected = set(oriented_pairs(gaps[step, agent], pairs[step, agent]))
                scores = self.scores.get(context, {})
                required += 1
                if scores.keys() != expected:
                    if len(examples) < 10:
                        examples.append(dict(context=context, expected=len(expected), observed=len(scores),
                                             missing=sorted(expected - scores.keys()),
                                             unexpected=sorted(scores.keys() - expected)))
                    continue
                covered += 1
                for pair in choose_pairs(context, scores, self.settings, random_control=self.random_control):
                    left, right = sorted(pair)
                    selected[step, agent, left, right] = True
            prepared.append(dict(episode, episode_id=episode_id, shapley_pair_mask=selected))
        audit = dict(task=task, phase=pool["phase"], required=required, covered=covered,
                     coverage=covered / required if required else None, examples=examples)
        return dict(pool, episodes=prepared), audit

    def task_loss(self, mac, pool, episodes):
        device = mac.args.device
        batch, _ = batch_episodes(episodes, self.scheme, self.groups, device,
                                  include_good=False, include_old_q=True)
        legal, decisions = decision_masks(batch)
        mask = pad_sequence([ep["shapley_pair_mask"] for ep in episodes], batch_first=True).to(device)
        current_q = unroll(mac, batch, batch.max_seq_length - 1)
        return full_rank_preference(current_q, batch["old_q_values"][:, :-1], legal, decisions,
                                    self.settings, pair_selection=mask)
