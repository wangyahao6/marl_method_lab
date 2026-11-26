# TS5 TC K1 / min_gap extension (2026-11-25)

Authorized scope: seeds 2222 and 3535 only. Reuse the existing fresh FT references, their S-boundary state (online/target/optimizer/RNG) and fixed S/Z/C/D history pools from `ts5_tc_coverage_20261123`. No reference retraining, no completed Low/Medium/High gap0 reruns, no seed4646 training. Original five-stage TS5 sequence, 1M/stage cumulative whole-episode boundaries, VDN single-FF64, FT, per-task epsilon, RMSprop, lr=0.0005, auxiliary weight=1, temperature=1, historical batch=8 and evaluation settings are unchanged.

## New training and reuse

- Gap0: add K1-only for each seed, on exactly the original TC valid state set. Every selected state has the N−1 original Top-1 pairs and zero extra pairs.
- Gap0.5: add K1-only/Low/Medium/High for each seed (eight branches). All four use the same filtered base and common effective states.
- Ten new branches each restore the original S checkpoint and train Z/C/D/E: 40M nominal new steps. References are not retrained. Six old two-seed gap0 TC branches are reused for comparison. Seed4646 is preserved in the original dataset but excluded from the two-seed primary comparison.

## Standard threshold and feasibility

The selector now supports min_gap=0 or 0.5. It calls the existing `reference_order` and `full_rank_pairs` helpers: legal old Q is centered and divided by its population standard deviation floored at 0.1; legal, distinct pairs pass when the absolute standardized difference is **>= min_gap**. The threshold applies to K1 and extra edges alike. No alternate gap definition or unfiltered K1 exception is introduced.

At gap0.5, the old valid state population (live decisions, >=4 legal actions, strict old-Q ordering) is intersected with: (a) at least one surviving K1 edge, and (b) enough threshold-passing non-Top-1 edges in every original distance bucket. K1-only uses this same intersection although it does not add edges. K1 base pairs can be fewer than N−1 after filtering, exactly as the existing K1 threshold rule implies. All four arms retain exactly the same filtered K1 base.

Each TC arm still adds N−2 extras. The original quota allocation is unchanged: Near distance1, Mid2, Far>=3, balanced subject to original unfiltered bucket capacity, ties Near→Mid→Far. Feasibility never reduces these quotas, shrinks the extra budget or lowers the threshold. Infeasible states are excluded identically from all four gap0.5 methods and counted explicitly. Root closure initialization uses only surviving K1 edges; candidate gains are computed only among threshold-passing, quota-feasible edges. Equal-gain Q-gap tie breaking targets the mean of passing candidates in the same bucket. The default gap0 path retains its original tie-breaking behavior.

Offline verification calls an independent Boolean Floyd-Warshall closure audit. It checks common valid states, exact K1 base equality, passing thresholds, fixed extras and per-bucket quotas, all masks and the source-pool hashes. Gap0 K1 state and base hashes are compared against the existing saved TC masks. Forty mask artifacts cover 2 seeds × 4 historical pools × (1 gap0 arm + 4 gap0.5 arms).

## Observed offline outcome

All eight gap0.5 historical pools have a nonempty common feasible set. Seed2222 retains 17,039 of 102,502 original states, seed3535 15,484 of 104,336. Across individual pools the retained fraction is about 9%–21%; this selection effect is material and must accompany all cross-gap comparisons.

| Seed | Gap | K1 Micro/Macro | Low Micro/Macro | Medium Micro/Macro | High Micro/Macro |
|---|---:|---|---|---|---|
| 2222 | 0.5 | 25.714% / 26.354% | 54.033% / 55.559% | 54.788% / 55.998% | 59.087% / 58.820% |
| 3535 | 0.5 | 25.891% / 26.808% | 54.260% / 56.017% | 54.961% / 56.427% | 59.146% / 59.203% |

Low/Medium are close after filtering; High−Low Macro TC is approximately 3.2 percentage points. These actual differences are reported, not enlarged by changing the requested threshold, budget or quotas. No new retained-fraction, TC-separation-size or performance gate is introduced. The feasibility requirement for this extension is a nonempty common state set per history pool satisfying all specified structural controls.

## Execution and artifacts

1. `scripts/offline_tc_gap_extension.py` generated masks and the offline feasibility report without training.
2. `scripts/plan_tc_gap_extension.py` constructs only ten new training tasks, ten independent final evaluations and a terminal summary/publication, plus a frozen-input verification task.
3. Submit the complete plan through `scheduler/service.py submit`. Queue limit5 inherits the prior batch; the global resource scheduler governs actual concurrency. Every training task depends on the frozen-input verification, which checks the already passed feasibility verdict and source/mask hashes.
4. `scripts/summarize_tc_gap_extension.py` reuses the common metric/mask verifier, forms separate two-seed gap0 and gap0.5 cohorts and exports same-seed TC−K1 and gap0.5−gap0 contrasts. A single failed branch is retained and does not cancel independent branches; no retries or extra budgets are implicit.

Configuration snapshots: `configs/continual/action_preference/tc_coverage/gap_extension/`.
Actual artifact paths: `results/ts5_tc_gap_extension_20261125/` and `runs/tc_coverage/ts5_tc_gap_extension_20261125_*`.
Masks: `results/ts5_tc_gap_extension_20261125/masks/seed*/gap*/{arm}/phase*.th` with frozen pair-count tensors, selection audit and source hashes. Raw selected/base/extra Q-gap and TC arrays are alongside masks in compressed NPZ. Statistics and exact TC state-frequency distributions are CSV files; original pools/masks/results are preserved.

## Reporting

Report Direct Coverage, Micro/Macro TC, Closure Gain, complete-order fraction, valid-state counts, base/extra edge counts and Q-gap distributions. Report ACC/BWT/FWT, normalized current-task AUC for every stage, per-task endpoint win rates and independent final evaluation counts. Same-seed comparisons are only seeds2222/3535; original4646 evidence remains separate.

Strict controls hold **within** each gap cohort. Across gaps, the effective state population and number of surviving K1 base pairs change by design, so a cross-gap performance difference cannot be attributed to the scalar threshold alone. TC varies with graph topology; residual Q-gap differences remain possible explanations. Two seeds and a best observed arm do not establish a general optimal TC interval. Training diagnostic counts contain repeated samples, whereas offline statistics count fixed states once.

Only after terminal artifact verification should the resulting report and lightweight data be committed as the extension result. The original three-seed result remains intact and should link to the extension rather than be overwritten.

## Terminal packaging

`scripts/publish_tc_gap_extension.py` is a scheduler task after the terminal summary. It requires all ten new training runs and their five-task independent final evaluations to be complete, rechecks the forty frozen masks, configurations and source pools, and packages the two-seed cohorts under `docs/results/parameter-ablations/ts5-vdn64-tc-gap-extension/`. It preserves the original dataset and appends only a link there. It commits the scoped report/data and result indices on the Ubuntu repository; GitHub synchronization uses the existing Mac deployment workflow after that commit exists. If other tasks have pending edits to shared indices, packaging stops for review without touching those edits or restarting training.

## 2026-11-26 terminal verification

All 23 scheduler tasks completed, including ten new training runs, ten independent evaluation jobs, frozen-input verification, summary and publication. No reference or completed gap0 TC branch was retrained. The [two-seed result](../results/parameter-ablations/ts5-vdn64-tc-gap-extension/README.md) preserves both gap cohorts, all paired metrics and the original three-seed result. New branches provide 50 model/task independent evaluations; with reused original artifacts the comparison contains 18 unique runs and 90 unique final model/task evaluations. Low−K1 final ACC and BWT are positive for both seeds at both gaps, while the TC arms show no monotonic mean-ACC trend. Cross-gap differences remain confounded with the common-state selection and filtered K1 pair counts.
