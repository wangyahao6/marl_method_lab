# Single-agent MountainCar TKPR

`PYTHONPATH=src python -m single_agent.train --method DQN-TKPR-K1 --seed 0`

The default schedule is T1 (goal 0.45) → T2 (0.50) → T3 (0.55), 300,000 environment steps each. The other methods are `DQN-FT`, `DQN-TKPR-K1-W`, and `DQN-TKPR-K2`. Each method and seed needs a separate output directory. `--steps-per-task N` is available for short smoke runs.

The TD replay is cleared at each switch. Network, target network, optimizer, and frozen-reference preference records persist. Reference collection uses greedy frozen task-end policies and stores states only from successful episodes. If a task has no such episodes, its preference memory is empty; training continues without a preference term until successful history is available. Each run saves task checkpoints, full stored old Q vectors, effective configuration, source snapshot, Git revision, and a summary in its output directory.

K1-W uses the existing confidence-weighted TKPR loss. K2 covers every pair of the three MountainCar actions and therefore equals FRPR Softplus under the same settings. The environment retains Gymnasium's original reward and 200-step episode limit; its observation appends `(goal_position - 0.50) / 0.05`.
