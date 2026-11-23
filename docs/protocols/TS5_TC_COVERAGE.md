# TS5-1M Transitive Closure Coverage Ablation

User-authorized 2026-11-23. Sequence explicitly confirmed as the repository TS5-1M:
S → Z → C → 2S3Z → 2Z3C, one million environment steps per stage (cumulative whole-episode boundaries).
Seeds: 2222, 3535, 4646. Single-FF VDN64, inherited TS5 VDN-FT configuration, RMSprop,
learning rate 0.0005 from default.yaml, per-task epsilon. No historical artifacts from older environments are reused.

## Reference and branch execution

Three fresh unprotected full-sequence references train from scratch. `full_rank_preference`
with `pair-reference` is a collection interface: its preference module captures old Q, but
`training_preference=None` throughout the reference. Its config records effective preference weight 0.
Save the S-boundary online network, optimizer, target network, RNG, environment distribution RNG
and training counters. Continue FT through Z/C/D/E, collecting the standard successful greedy
history pool and rollout-time old Q after S/Z/C/D. The final reference E stage provides the
full five-stage VDN-FT baseline. No auxiliary preference loss affects any reference update.

Every formal arm restores its seed's same S snapshot (including optimizer and target), then trains
Z/C/D/E for 4M new nominal steps. All arms consume the same fixed S/Z/C/D reference pools.
No branch collects replacement histories. Three references (15M) plus nine 4M branches (36M)
are 51M nominal new environment steps; shared S prefixes are not independent replicates.

## Masks and controls

A state is one valid live agent decision. Common eligibility: at least four legal actions and
strict old-Q ordering among legal actions. Preserve every K1 Top-1-to-other legal action pair.
Add N−2 non-Top-1 pairs, giving 2N−3 direct pairs per eligible state. Near = rank distance 1,
Mid = 2, Far >=3. Allocate the N−2 extras by repeatedly incrementing the least-filled bucket
with remaining capacity, ties Near→Mid→Far. This gives identical per-state budgets and quotas
across arms, including states where graph differences cannot produce different TC.

Starting from K1, recompute exact marginal transitive closure count for every remaining
quota-feasible edge. Low selects the smallest gain, High the largest, Medium the median
of distinct feasible gain values. TC count and normalized TC gain give the same per-state
ranking. Within an equal gain class, minimize absolute deviation from the common mean
candidate Q-gap in that distance bucket; final tie break is stable action-rank pair ID.
This is a greedy topology intervention, not a proof of globally extremal attainable TC.
The secondary Q-gap target does not filter candidates or override closure gain.

`min_gap=0`, weight=1, temperature=1, old legal-Q population std floor=0.1, batch=8
historical episodes. All training loss/optimizer/init/budget parameters are shared.
Q-gap matching is approximate and explicitly not a gate. Report selected pair and extra-pair
gap distributions; residual differences are competing explanations.

Direct coverage = |E|/choose(N,2). TC = |closure(E)|/choose(N,2).
Complete Order Coverage = fraction of eligible states with TC=1.
Closure Gain = TC−Direct coverage. Report unique-state distributions, Micro TC (pooled pair
counts), Macro TC (equal state weights), per-pool and per-seed aggregates. Independent
Floyd-Warshall audit checks selector closure counts, all K1 edges, eligible sets, budgets and quotas.

## Offline gate and queue

`python scripts/plan_tc_coverage.py --output results/ts5_tc_coverage_20261123/plan.json`
creates the full dependency plan. Submit through `scheduler/service.py submit`.
Batch queue limit 5; the existing global scheduler/resource allocation still governs admission.
Other experiment settings and the server-wide limit are unchanged.

References → per-reference offline mask/audit → all-seed feasibility → nine formal branches
→ independent final 128-episode/task evaluations → terminal summary.
Before any formal branch can run, all three references and offline audits must complete and
the all-seed feasibility task must pass. An explicit operational definition of clear separation:
for each seed, pooled over its four reference history pools, High−Low >=5 percentage points
and both adjacent-arm differences >=1 point for both Micro and Macro TC.
No Q-gap threshold. Failure stops dependent formal training and preserves the report/masks;
no retries, seed additions or budget changes are implicit. Ordinary diagnostics during training
are records, not new stopping conditions. A single unrelated branch failure does not cancel peers.

## Artifacts and analysis

Runs: `runs/tc_coverage/ts5_tc_coverage_20261123_*`.
Reference `offline_tc/`: frozen masks/counts, tensor and file SHA-256, source pool/checkpoint
hashes, provenance, per-state CSV.gz, Q-gap/TC distribution NPZ, statistics CSV and audit JSON.
Batch results: `results/ts5_tc_coverage_20261123/` (plan, feasibility, reports, actual configs,
per-seed and aggregate ACC/BWT/FWT/AUC, raw evaluation counts and training mechanisms).
Training source archive and environment dimensions are inherited from the run launcher.

Stage AUC is trapezoidal current-task greedy win rate integrated over actual stage training steps,
normalized by actual duration (percent). ACC is equal-task endpoint win rate. BWT/FWT follow
repository definitions; FWT uses the initial untrained references and next-unseen evaluations.
Top-1 flips, preference loss and pair violations come from repeated sampled history updates;
offline coverage counts each fixed state once. Summaries retain this denominator distinction.

Analyze same-seed contrasts, seed consistency, stage-specific retention and learning, and Q-gap
imbalances. Three seeds do not establish a general optimal TC interval. Graph structure changes
with TC, so scalar TC correlations alone do not establish causation. Fixed distance quotas can
make complete order impossible: absent full-order states, this experiment cannot decide whether
full ordering is better than partial ordering. Do not add a fourth training method to answer it.

After actual training/evaluation and results verification, publish the lightweight evidence under
`docs/results/parameter-ablations/ts5-vdn64-tc-coverage/`, update result indices, and synchronize
GitHub main using the existing repository workflow. Raw masks, pools, models and full logs remain remote.
