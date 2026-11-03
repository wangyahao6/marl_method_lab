# TS3 VDN A2 overall screening and historical union

Authorized 2026-11-03. Add one seed3535 branch from the completed common A prefix (2,000,023 steps). Use the saved overall threshold set [24,61] at the A/B boundary. At B/C/D use the branch own latest 32 complete episodes, VDN exact sensitivity and all-agent weighted activity with layer-normalized score strictly >3. Apply historical union and .1 actual RMSprop increments only to A2 slices. Compare against existing baseline and type-union paths. Coverage can differ with the screening rule. New interaction budget approximately 8M, cumulative thresholds 4/6/8/10M. Same evaluation protocol, full episode boundaries, and recoverable training-state restore as the type-union run.

Entry scripts/run_ts3_vdn_overallunion.py uses scripts/run.py. Output results/ts3_vdn_A2_overallunion_3535_20261103 includes three-path summary. Original prefix and results are shared read-only inputs.
