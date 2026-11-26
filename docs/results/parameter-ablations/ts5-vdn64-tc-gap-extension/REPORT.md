# TS5-1M TC: K1-only and min_gap=0.5 extension

Only training seeds 2222/3535 enter both main comparison cohorts. Seed 4646 remains in the original dataset. No reference retraining and no completed TC gap0 reruns. Original task sequence, 1M per stage, RMSprop/lr, auxiliary weight and evaluation seeds/counts are preserved.

New training completed: 10/10. References and six completed gap0 TC branches are reused. Shared-prefix S metrics are reused, not independent repeats.
[Offline feasibility](OFFLINE_REPORT.md) and offline_statistics.csv specify common states, exact base masks, budgets, distance quotas and Q-gap statistics.
Gap0 K1 has N-1 Top-1 pairs on exactly the old common valid states. Gap05 K1 and TC arms share threshold-filtered Top-1 pairs and the same quota-feasible state set. TC extras retain N-2 total and the original Near/Mid/Far per-state quotas; K1-only has zero extras.
Normalization and filtering directly reuse reference_order/full_rank_pairs. Filtering applies to K1 as well as extras. Some original K1 edges therefore do not survive gap05. Threshold is never lowered and budgets/quotas are never shrunk.

| Gap | Method | n | ACC % | BWT pp | FWT pp | E AUC % |
|---:|---|---:|---:|---:|---:|---:|
| 0 | VDN-FT-reference | 2 | 26.484 | -36.035 | 18.750 | 57.117 |
| 0 | K1-only | 2 | 51.328 | 6.445 | 20.996 | 39.369 |
| 0 | low-TC | 2 | 53.594 | 10.156 | 20.898 | 43.097 |
| 0 | medium-TC | 2 | 52.344 | 11.621 | 18.652 | 38.225 |
| 0 | high-TC | 2 | 53.438 | 11.914 | 19.434 | 41.176 |
| 0.5 | VDN-FT-reference | 2 | 26.484 | -36.035 | 18.750 | 57.117 |
| 0.5 | K1-only | 2 | 44.922 | -7.812 | 15.918 | 40.371 |
| 0.5 | low-TC | 2 | 47.656 | -0.781 | 16.211 | 37.892 |
| 0.5 | medium-TC | 2 | 46.719 | -2.441 | 16.504 | 31.010 |
| 0.5 | high-TC | 2 | 47.969 | -2.637 | 17.188 | 31.312 |

comparison_stage_metrics.csv saves all stages and per-task win rates. paired_vs_k1.csv compares each TC arm against same-seed K1 within each gap. paired_gap05_minus_gap0.csv records cross-gap contrasts, which combine filtering and a change of eligible-state population and do not isolate the threshold effect.
Direct coverage, Micro/Macro TC and Closure Gain are in offline_statistics.csv; exact per-state TC frequencies are in tc_state_histogram.csv; raw per-state selected/base/extra gap and TC distributions are in masks/seed*/gap*/{arm}/phase*_distributions.npz on the experiment host.
Actual configurations and restoration identities are in gap0/configs/ and gap05/configs/. Each terminal formal history mask is checked against its frozen offline hash by the shared summarizer. Incomplete/failed paths are retained without implicit retries or added budgets.
Pair-violation and preference-loss diagnostics count repeated sampled historical episodes; Top-1 flip denominator follows the existing implementation. Fixed-pool TC is not sampling-weighted. Two training seeds cannot establish a general optimal TC interval or a causal effect of scalar TC independent of graph topology and Q-gap differences.
