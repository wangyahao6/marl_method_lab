# TS5-1M TC offline feasibility

For each seed, pooled across four history pools: adjacent arms >=1 percentage point and High-Low >=5 points in both Micro and Macro TC. No Q-gap gate.

Strict controls: identical within-seed source pools, valid states, all K1 edges, per-state N-2 extra edges, and Near/Mid/Far quotas. Near=distance 1; Mid=2; Far>=3.
Q-gap is only a secondary tie break among equal closure gains. Actual distributions are retained; no gap filtering or exclusion.
Direct edge coverage = |E| / choose(N,2). TC = |closure(E)| / choose(N,2). Micro pools numerators/denominators; Macro averages states. Complete order coverage = fraction of states with TC=1. Closure gain = TC minus direct coverage.
States with fewer than four legal actions or exact old-Q ties are excluded identically. They are not removed to improve separation. Low action counts may have identical attainable TC.
Because distance quotas reserve Mid/Far edges, complete-order coverage may be zero in all arms; that outcome cannot answer whether full ordering improves learning.

| Seed | Arm | Micro TC | Macro TC |
|---|---|---:|---:|
| 2222 | low | 0.5515 | 0.5947 |
| 2222 | medium | 0.6092 | 0.6377 |
| 2222 | high | 0.7523 | 0.7491 |
| 3535 | low | 0.5517 | 0.5956 |
| 3535 | medium | 0.6099 | 0.6388 |
| 3535 | high | 0.7526 | 0.7499 |
| 4646 | low | 0.5573 | 0.6045 |
| 4646 | medium | 0.6136 | 0.6449 |
| 4646 | high | 0.7514 | 0.7486 |

Verdict: PASS; dependent training may start.

See offline_statistics.csv for poolwise controls, Q-gap means/medians/std, complete-order coverage, and TC quantiles. Raw per-state distributions, masks, hashes and pool provenance remain in each reference run/offline_tc/.
