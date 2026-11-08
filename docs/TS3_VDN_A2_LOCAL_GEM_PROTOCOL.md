# TS3 VDN: equal-size A2 high/middle/low local GEM

2026-11-04 user authorized remote execution of method-added counterparts to existing VDN seeds; queue when resources are insufficient, permit parallelism as old experiments finish. No new baseline training or A-prefix training.

## Fixed comparison and reuse

Seeds1234/4567/3535 all have completed matched VDN baselines with five2M stages. For each seed reuse its original full A-prefix snapshot (online, target, RMSprop, replay, exploration/counters, random/distribution states) and archived last32A episodes. Baselines1234/4567: `ts3_vdn_top3_N_seed<seed>_20261102_attempt001`; baseline3535: `ts3_vdn_typeunion_N_seed3535_20261102_attempt001`. Prefix/source paths are explicit in `configs/continual/ts3_vdn_A2_local_gem_20261104.json` and result `reuse_audit.json`. Current core learner/agent/controller/runner/config/environment code hashes match all three original baselines. New SC2 processes restore recoverable distribution state; engine bit-level continuity is not claimed.

Nine new methods only: high/middle/low for each seed. Each continues B-E from ~2M to10M cumulative steps (8M nominal new). Total72M nominal added steps,90M path-equivalent; actual increment is endpoint minus each restored prefix, with whole-episode overshoot. Same TS3: friendly2S3Z; enemy2S3Z->4S1Z->1S4Z->3S2Z->5S0Z. No additional seeds, training stages, full-network GEM, 0.1 soft protection or combinations.

## Screening and parameter masks

At A/B/C/D boundaries use last32complete current-task training episodes, same online endpoint model and exact VDN team action-sensitivity weight `omega=c/mean(c)`. Rank all64A2 units by state-weighted mean absolute activation over all agents, descending; ties ascending index. Each arm selects21units: high ranks1-21, middle22-42, low44-64. Rank43 stays ordinary in all arms at that boundary. Zero activation with defined positive sensitivity is resolved by ties; undefined sensitivity preserves previous active mask (empty if none), records phenomenon and does not terminate training. At B/C/D use the arm's own scores, not future/baseline scores.

A2=`agent.net[4]`. Each selected unit maps to net.3 incoming weight row/bias plus all three GRU weight_ih input columns,257unique elements/unit,5397each arm. Unselected entries, recurrent weight_hh, GRU biases and action output parameters update normally. No replacement of the ordinary replay policy or exploration reset. Grouping is latest-boundary replacement, NOT neuronal history union. Task memories do accumulate separately from unit identities.

## Local GEM adaptation

GEM reference: Lopez-Paz & Ranzato, Gradient Episodic Memory for Continual Learning, NeurIPS2017, section3 eq7-8, https://arxiv.org/html/1706.08840v6 . Author code https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py was statically inspected; no dependency or source code copied into the runtime.

For each completed task retain32complete training episodes, recompute GRU histories from episode0 using current reference parameters; team TD mean-square loss, same Double-Q, padding/termination and normal target-network policy as the existing learner. Target action/values are detached within the loss evaluation. Per old task get the gradient restricted to the current21-unit mask, g_k,S. All old tasks remain separate constraints. At E there are4such gradients.

Let d0 be the selected part of the ordinary clipped-RMSprop actual increment. Solve `min_d .5||d-d0||^2` s.t. `g_k,S dot d <= 0` for every available old-task reference gradient. Write this local increment back, leaving other elements exactly ordinary. RMSprop state accumulates as before; target synchronization occurs after the projected update. This is local actual-update GEM, NOT original SGD GEM, not a guarantee that the complete-model TD loss or team win rate does not deteriorate. Unselected parameters can still increase old losses, and TD targets change over time.

The <=4-dimensional dual is implemented via exact small active-set enumeration, float64 homogeneous normalization, zero RHS/zero margin, no positive dual bias or diagonal epsilon. Zero reference rows are omitted. Numerical fallback is zero local increment, always feasible; log occurrences. Reference calculation exceptions are recorded and skipped for that reference/update, without stopping the base learner or other queued runs. No new performance/diagnostic stop conditions. QP/TD-loss implementation checks are offline on existing data; no environment validation training.

## Evaluation

Inherited32greedy current-task episodes at stage starts and ~25k steps;128per seen task at each stage end, same fixed seed blocks. PrefixA evaluation is referenced, not counted as another trained replicate. Also128episodes on next unseen task before any training there.

For FWT, evaluate a fresh untrained same-seed architecture reference constructed by the current initialization path (save/hash the actual reference model). It is explicitly a reconstructed reference, NOT an archived historical initial model. Cache once per seed with model provenance; differing reconstructed hashes are reported, not training gates. Three evaluation-only jobs load old baseline A/B/C/D checkpoints to collect next-unseen results. They use the public launcher, have zero training steps, and do not write into original baseline runs. No training or replay insertion from evaluation. Training RNG restored after every evaluation. Each task uses the same focus seed blocks as old-task evaluation. Report ACC/BWT/FWT and five-task matrices per seed, current-task curves/AUC and per-seed method-minus-baseline contrasts. FWT missing points remain N/A; always disclose its reconstructed-reference scope.

Aggregate local gradient/increment cosine and conflicts, QP trigger/fallback counts, float32 actual-update constraint residuals, update/correction norms, reference time, membership and parameter count. Save compact per1000-update summaries plus boundary states/inputs; no per-step full gradients. Archived model+optimizer, targets and gem_state with task memories/masks provide method provenance, not a complete mid-phase trainer-resume snapshot. The final aggregate includes auxiliary errors as interpretive limitations, never as automatic cancellation criteria.

## Dispatch and artifacts

Public launcher `scripts/run.py --suite ts3 --algorithm vdn --method gem --ts3-role branch --ts3-group GEM_<high|middle|low> --parent <original prefix> --protocol configs/continual/ts3_vdn_A2_local_gem_20261104.json ...`.

Supervisor `scripts/run_ts3_vdn_local_gem.py`, result `results/ts3_vdn_A2_local_gem_20261104/`. Max5new jobs; globalGPUcompute processes<8before launch, freeGPU>=3500MiB, availableRAM>=12000MiB,90seconds between launches to allow allocations to settle. These are dispatch-resource rules, not training early-stop thresholds. Poll20seconds. No other queue/process is modified or stopped. A failed job is recorded independently, no automatic extra attempt/budget. Queue ends by running `scripts/summarize_ts3_vdn_local_gem.py`, producing curves/matrices/metrics and REPORT for subsequent terminal/completeness review. No desktop wakeup or notification automation was configured.

## 2026-11-08 overall A2 score > 3 supplement

The user requested a separate local GEM branch selected by the strict A2 threshold. The protocol is `configs/continual/ts3_vdn_A2_local_gem_gt3_20261108.json`, with launcher group `GEM_gt3`. Use the same normalized, state-weighted all-agent A2 score. At the A, B, C and D boundaries, replace the active mask with all A2 units whose score exceeds 3. The count follows the score distribution. An empty or undefined selection disables local projection for the next stage. Keep the task-wise memories and RMSprop increment projection of the original local GEM experiment. The A masks use the archived overall scores and initially contain [16,23], [39,43] and [24,61] for seeds 1234, 4567 and 3535, respectively.

Reuse the matching A prefixes, VDN baselines and initial FWT references. Train three B–E branches, 8M nominal new steps each, 24M total. Stage and FWT evaluation follow the original local GEM protocol. The complete scheduler plan and separate summary output live under `results/ts3_vdn_A2_local_gem_gt3_20261108/`.
