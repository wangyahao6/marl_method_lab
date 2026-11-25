# TS5-1M Transitive Closure Coverage Ablation

S → Z → C → 2S3Z → 2Z3C, five 1M stages, seeds 2222/3535/4646. VDN single-FF64, FT replay, per-task epsilon, RMSprop. Three fresh full VDN-FT references and nine branches sharing their seed-specific S boundary and four fixed FT historical success pools.

Completed training runs: 12/12. Full terminal evidence requires completed training, five-stage greedy evaluation matrix, and independent final evaluations.

Offline selection, feasibility and controls: [OFFLINE_REPORT.md](OFFLINE_REPORT.md). Actual configs and source commits: configs/ and runs.csv.
Performance: stage_metrics.csv (ACC percent, BWT/FWT percentage points, normalized per-stage current-task learning AUC percent), seed_summary.csv, paired_final.csv, evaluations.csv.
Mechanisms: offline_statistics.csv, offline_seed_summary.csv and preference_diagnostics.csv. Offline counts are unique fixed states; training mechanism counts include repeated historical sampling. Top-1 flips use the existing unique-old-top1 decision denominator, which also includes decisions outside the >=4-action TC subset.

| Method | Seed | ACC % | BWT pp | FWT pp | E AUC % |
|---|---:|---:|---:|---:|---:|
| VDN-FT-reference | 2222 | 25.000 | -34.961 | 15.039 | 57.439 |
| VDN-FT-reference | 3535 | 27.969 | -37.109 | 22.461 | 56.795 |
| VDN-FT-reference | 4646 | 26.094 | -38.086 | 16.406 | 57.185 |
| low-TC | 2222 | 51.719 | 11.133 | 22.070 | 43.567 |
| medium-TC | 2222 | 47.969 | 9.570 | 19.531 | 34.372 |
| high-TC | 2222 | 47.500 | 9.570 | 18.750 | 36.201 |
| low-TC | 3535 | 55.469 | 9.180 | 19.727 | 42.627 |
| medium-TC | 3535 | 56.719 | 13.672 | 17.773 | 42.078 |
| high-TC | 3535 | 59.375 | 14.258 | 20.117 | 46.151 |
| low-TC | 4646 | 54.062 | 15.430 | 17.383 | 32.996 |
| medium-TC | 4646 | 52.188 | 15.625 | 14.648 | 37.823 |
| high-TC | 4646 | 50.156 | 13.867 | 14.258 | 34.618 |

acc_pct final-stage High−Low: seed2222 -4.219, seed3535 +3.906, seed4646 -3.906; mean -1.406. Monotonic patterns: seed2222 decreasing, seed3535 increasing, seed4646 decreasing.

bwt_pp final-stage High−Low: seed2222 -1.562, seed3535 +5.078, seed4646 -1.562; mean 0.651. Monotonic patterns: seed2222 decreasing, seed3535 increasing, seed4646 nonmonotonic.

fwt_pp final-stage High−Low: seed2222 -3.320, seed3535 +0.391, seed4646 -3.125; mean -2.018. Monotonic patterns: seed2222 decreasing, seed3535 nonmonotonic, seed4646 decreasing.

learning_auc_pct final-stage High−Low: seed2222 -7.366, seed3535 +3.525, seed4646 +1.621; mean -0.740. Monotonic patterns: seed2222 nonmonotonic, seed3535 nonmonotonic, seed4646 nonmonotonic.

Highest observed three-seed final ACC: low-TC (53.750%). This identifies an observed arm, not a general optimal TC interval.

tc_performance.csv joins performance to the cumulative unique-state fixed-pool TC and Q-gap statistics available at each stage; these are not exposure-weighted training TC. within_seed_trends.csv retains per-stage paired trends and residual gap differences.

Interpretation limits: three training seeds are the independent replicates; states/pairs are not extra replicates. The arms preserve K1, edge budgets and per-state distance quotas, while graph topology and TC change together. TC-performance correlation alone does not isolate a causal role for the scalar TC score. Residual Q-gap differences remain possible competing explanations. The experiment contains no separately budgeted full-order training arm; if complete-order coverage is zero throughout, it cannot determine full versus partial ordering superiority. A best observed three-arm setting is not an estimated general optimum.

Shared-prefix AUC and stage-S metrics reuse the reference S trajectory once. References continue FT through all five stages; formal methods replay masks selected on FT reference states, including C/D reference states rather than each branch's own history. This is an intentional fixed-reference intervention.
