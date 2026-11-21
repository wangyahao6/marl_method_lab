# MountainCar single-agent TKPR comparison, seed 3535

## Protocol

- Methods: DQN-FT, DQN-TKPR-K1, DQN-TKPR-K1-W, DQN-TKPR-K2. All four use the same DQN and task configuration in `../mountaincar_tkpr.json`; only the preference method changes.
- Training seed: 3535 for every method. Task order: T1 (goal 0.45) → T2 (0.50) → T3 (0.55), 300,000 environment steps per task. Total planned training budget: 3,600,000 steps across four independent runs.
- Training uses Gymnasium MountainCar-v0's original reward, 200-step episode limit and shared three-dimensional observation. Epsilon resets at each task switch. The TD replay clears; online network, target network, optimizer and historical preference memory persist.
- After T1 and T2, each task-end frozen greedy model attempts 100 reference episodes. Only successful trajectories enter preference memory, with full old Q vectors. Empty history does not stop training; it means the corresponding TKPR term cannot be exercised. This is reported separately from task performance.
- Independent evaluation runs after training: each of the three stage-end checkpoints is evaluated greedily on T1, T2 and T3 using 100 episodes per evaluated task, with seeds `91000 + 1000 * task_index + episode_index`. Evaluation seeds are identical across methods and distinct from training and reference-collection seeds.

## Readout

The primary comparison is the T3 checkpoint's success rate on T1, T2 and T3 versus DQN-FT. Report the nine stage/task success rates, mean returns and episode lengths per method. The T1 and T2 checkpoint evaluations show acquisition and later retention. Training episode records and 10,000-step progress records show learning behavior; `reference_states` and `preference_updates` show whether TKPR was actually active. A zero-memory or zero-update run remains in the report and is not treated as evidence of TKPR benefit or failure.

This is one paired training seed. Differences are descriptive and cannot establish seed-robust effects. No convergence threshold or diagnostic gate changes the planned training budget. Each run and evaluation is an independent scheduler task; a failed task is recorded without cancelling unrelated methods.
