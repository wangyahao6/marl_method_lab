# TC K1 / min_gap extension offline report

Standard filter reuses reference_order + full_rank_pairs: population std of legal old Q (floor 0.1), non-tied upper-triangle legal pairs, abs(standardized difference) >= min_gap. All selected edges, including K1, use the same standard. This leaves fewer than N-1 K1 pairs at gap05 in some states; no filtered pair is restored.
Gap0 K1 uses the same old valid states and exact N-1 old Top-1 edges. Gap05 common states are the old valid population intersected with at least one surviving K1 edge and sufficient threshold-passing non-Top-1 capacity in EACH original distance bucket. K1-only uses that same common set although it needs no extras. No silent threshold or budget change.
Counts and Q-gap distributions: offline_statistics.csv. TC = directed transitive closure / choose(N,2); Micro pools pairs, Macro averages states. Closure Gain = Micro TC minus Direct Coverage.

| Seed | Phase | Old states | Gap05 common | Retained % |
|---|---:|---:|---:|---:|
| 2222 | 0 | 21881 | 3084 | 14.09 |
| 2222 | 1 | 21468 | 4447 | 20.71 |
| 2222 | 2 | 33356 | 4565 | 13.69 |
| 2222 | 3 | 25797 | 4943 | 19.16 |
| 3535 | 0 | 22286 | 3339 | 14.98 |
| 3535 | 1 | 22794 | 4037 | 17.71 |
| 3535 | 2 | 32104 | 2931 | 9.13 |
| 3535 | 3 | 27152 | 5177 | 19.07 |

| Seed | Gap | Arm | Micro TC | Macro TC |
|---|---:|---|---:|---:|
| 2222 | 0 | k1 | 0.30128 | 0.32978 |
| 2222 | 0.5 | k1 | 0.25714 | 0.26354 |
| 2222 | 0.5 | low | 0.54033 | 0.55559 |
| 2222 | 0.5 | medium | 0.54788 | 0.55998 |
| 2222 | 0.5 | high | 0.59087 | 0.58820 |
| 3535 | 0 | k1 | 0.30159 | 0.33053 |
| 3535 | 0.5 | k1 | 0.25891 | 0.26808 |
| 3535 | 0.5 | low | 0.54260 | 0.56017 |
| 3535 | 0.5 | medium | 0.54961 | 0.56427 |
| 3535 | 0.5 | high | 0.59146 | 0.59203 |

PASS: training may start.
Gap0 and gap05 use different common populations; cross-gap performance differences combine gap screening and state selection. Seed 4646 is preserved only in the original dataset and excluded from this two-seed comparison. No TC gap-size or retained-state-fraction threshold is added.
