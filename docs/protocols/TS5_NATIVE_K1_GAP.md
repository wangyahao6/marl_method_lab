# TS5-1M normal TKPR-K1 gap-only ablation (2026-11-26)

Authorized: exactly four new training tasks, `topk_preference(k=1)` × min_gap 0/0.5 × seeds2222/3535. This experiment does **not** use `pair_tc_k1`, TC state eligibility, Near/Mid/Far quotas, selected-pair masks or extra non-Top-1 edges. The native loss and QLearner implementations are unchanged.

## Shared experiment settings

Reuse each seed's original `ts5_tc_coverage_20261123_reference_seed*` S checkpoint and `return_state.th`: online model, target Agent, optimizer, counters, RNG and environment distribution state. Reuse the entire fixed S/Z/C/D successful trajectory pools with their stored rollout old-Q tensors. No reference training, prior TC rerun or extra seed is scheduled.

S→Z→C→2S3Z→2Z3C, 1M cumulative whole-episode stage boundaries; VDN single-FF64, FT TD replay, per-task epsilon, RMSprop, lr0.0005, preference weight1, temperature1, old legal-Q std floor0.1, historical batch8, original evaluation blocks and25k interval. Each new branch resumes at S end and trains Z/C/D/E (4M nominal new steps); four branches add16M nominal steps. Shared S learning metrics are reused and are not independent training repetitions.

## Native implementation and fixed history

`RankingPreference` calls the existing `learners.ranking_preference.topk_preference`, with k=1. Its existing `reference_order` normalizes legal historical Q using per-state population std with floor0.1. Existing legal, nonterminal/live decisions with at least two legal actions enter the normal pair construction; equal-old-Q pairs remain excluded by the unchanged native definition. Gap0 keeps every such native Top-1 pair, while gap0.5 removes only pairs with absolute normalized old-Q difference **<0.5**. Neither configuration filters whole states or complete trajectories.

States with no surviving pair remain in the stored history and recurrent unroll, contribute zero corresponding preference loss, and do not enter the native valid-loss-state average. No confidence weighting, hinge/ReLU substitution, per-task batch change or denominator change is introduced.

The opt-in `fixed_reference_history` driver path handles compatibility with the original FT `pair-reference` snapshot, then attaches complete reference pools one boundary at a time: S for Z, S/Z for C, S/Z/C for D, S/Z/C/D for E. Future-task pools are held separately and never exposed to the preference sampler early. At each attachment the source `success.th` is copied byte-for-byte, all original episodes/states are kept, and source/local SHA-256, class `RankingPreference` and loss function `learners.ranking_preference.topk_preference` are logged in `fixed_reference_memory.jsonl`.

The change is limited to checkpoint/protocol compatibility and fixed-history loading. It does not change `src/learners/ranking_preference.py`, `src/learners/q_learner.py`, native sampling or native loss logic. Existing protocols that do not opt in retain their original behavior.

## Offline inventory (unique fixed historical data)

Both gaps use the same400 successful episodes per seed across four original pools. Counts below are diagnostic reconstruction verified directly against the native `topk_preference` return counts; no diagnostic mask is supplied to training.

| Seed | Gap | Live decision states | States with pairs | Zero-pair states retained | Selected pairs |
|---|---:|---:|---:|---:|---:|
| 2222 | 0 | 102823 | 102823 | 0 | 545725 |
| 2222 | 0.5 | 102823 | 94852 | 7971 | 372660 |
| 3535 | 0 | 104675 | 104675 | 0 | 554418 |
| 3535 | 0.5 | 104675 | 92785 | 11890 | 358036 |

Gap0 includes634/674 native pairs from states with two or three legal actions; gap0.5 includes130/85. They are not excluded by the earlier TC >=4-action rule. The complete tensor/trajectory inputs are unchanged between gaps. Q-gap means/medians/std and per-pool checkpoint/input hashes are in `historical_pair_inventory.csv`; the zero-loss-state counts here are not a removal of those states from history.

## Execution and outputs

- Prepare only configurations, pair-count audit and manifest with `scripts/prepare_native_k1_gap.py`; no SC2 training is used for validation.
- Submit `results/ts5_native_tkpr_k1_gap_20261126/plan.json` through `scheduler/service.py submit`. Four training jobs, four independent final evaluations, terminal report and publication are the entire plan. Batch limit4; actual admission follows the shared scheduler resources.
- Each training goes through `scripts/run.py --suite vdn_action_set --algorithm vdn_1ff --method topk_preference --preference-role shared-branch --parent <original FT reference>` with its own named protocol and unique run ID.
- At terminal completion `scripts/report_native_k1_gap.py` reconstructs ACC/BWT/FWT, per-task win rates and current-task learning AUC for every stage from raw evaluation counts, checks trainer retention values, and exports same-seed gap0.5−gap0 differences. AUC is trapezoidal greedy win rate over actual stage training steps divided by actual duration (percent); BWT/FWT differences are percentage points.
- Actual sampled historical pair counts are aggregated from `topk_preference_updates.jsonl`. They include repetitions and are separate from fixed-pool inventory counts. Future-pool participation, all four memory-copy hashes, native class/function, restoration identity, configuration snapshots and final evaluation files are checked before publication.

Results: `results/ts5_native_tkpr_k1_gap_20261126/`. Runs: `runs/ts5_native_tkpr_k1_gap_20261126_*`. Configuration files: `configs/continual/action_preference/ts5/native_k1_gap/`.

Only after all four trainings and20 independent model/task final evaluations (128 episodes each) are verified will the publication job package a standalone result under `docs/results/parameter-ablations/ts5-native-tkpr-k1-gap/`, update result indices and commit scoped reports/data to Ubuntu main. Existing TC datasets and other experiment results are not edited. GitHub main synchronization follows the existing deployment workflow after terminal verification. Failures retain partial evidence and do not authorize reruns, extra seeds or extra budgets.
