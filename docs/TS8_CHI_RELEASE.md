# TS8 CHI release-duration comparison

Run seed 3535 on each single-feedforward VDN width (64 and 128) using
the original TS8 roster sequence, thresholds and per-task exploration reset.
Use latest-selection CHI, with fixed weighted probes and a strict score >3.
Reuse the 64-unit TS8 CHI A1 prefix. Start the 128-unit shared trajectory
from initialization because the archived TS8 baseline lacks a branch checkpoint.

For each width, the shared trajectory ends at the first boundary with an exit
and a subsequent training stage. Save the continuation before changing the
mask, plus the prior and current masks and probe paths. Restore it independently
in five arms: immediate, 50k, 100k, 200k, and keep the exiting units protected
for the next stage. Each arm trains through that stage only. If a width has no
exit before the last stage, record that outcome and skip its five arms.

The intervention changes only the exited units. Use paired 128-episode
evaluation blocks for the next task at the boundary and every 25k steps
through 200k, and for old tasks at 50k, 100k, 200k and stage end. Log the
actual whole-episode evaluation step, and record nominal window labels
for comparing arms. Preserve stage-end ACC, BWT, FWT and model artifacts.
