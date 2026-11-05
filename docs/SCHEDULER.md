# Remote experiment scheduler

The server-wide default is six active tasks across registered experiment queues. Training, evaluation and summary attempts each consume one slot. A queue can retain a lower local limit. The service fills available slots with eligible tasks ordered by priority, then enqueue order. Resource availability and reservations for processes still allocating memory are both considered. Legacy Local GEM retains its five-slot queue limit, GPU/RAM availability settings and 90-second launch spacing. Selection Matched retains its serial limit.

## Operation

The user service is `marl-scheduler.service`. User lingering is enabled so dispatch survives SSH logout and can start after reboot. Run the following in the remote repository.

```sh
.venv/bin/python scheduler/service.py status
.venv/bin/python scheduler/service.py limit 6
systemctl --user status marl-scheduler
journalctl --user -u marl-scheduler -n 30
```

Changing the limit affects new dispatch. Running training continues until its normal exit. Restarting the scheduler leaves workers alive. Startup reconciles process boot ID, PID and start ticks with stored attempts. A lost or ambiguous launch becomes unknown and requires inspection. Failures do not cause automatic retries. Failed dependencies block dependent training while independent tasks continue. Summary tasks can depend on terminal states to produce reports even after failures.

SQLite in `runtime/scheduler/state.sqlite3` is the primary dispatch record. Tasks, immutable specifications, dependencies, attempts, exit codes and changes of state are retained. Original training artifacts provide progress, metrics and training completion. Experimental result review remains recorded in the experiment report.

## Submit a complete authorized plan

```sh
.venv/bin/python scheduler/service.py submit /absolute/path/plan.json
```

A plan has an `id` and a `tasks` list. Each task includes a stable `id`, string-list `argv`, absolute `cwd`, `type`, `dependencies`, `resources`, and optional absolute `run_dir`. Other supported fields are `queue`, `queue_limit`, `cooldown_seconds`, `priority`, `mutex` and `metadata`. Dependencies name tasks in the same submission or tasks already registered. Cycles and unknown dependencies are rejected atomically. Repeated identical submission is idempotent. Changing a submitted specification requires a new task ID. The output directory defaults to a mutex key. Commands execute without a shell.

```json
{
  "id": "authorized-batch",
  "tasks": [{
    "id": "unique-run-id",
    "argv": ["/home/wyh/marl_method_lab/.venv/bin/python", "scripts/run.py", "--suite", "ts3", "--algorithm", "vdn", "--method", "baseline", "--seed", "1234", "--run-id", "unique-run-id", "--protocol", "configs/continual/example.json", "--ts3-role", "prefix"],
    "cwd": "/home/wyh/marl_method_lab",
    "run_dir": "/home/wyh/marl_method_lab/runs/unique-run-id",
    "type": "training",
    "dependencies": [],
    "resources": {"gpu": 0, "gpu_mib": 1300, "ram_mib": 2000},
    "metadata": {"results": "/home/wyh/marl_method_lab/results/authorized-batch"}
  }]
}
```

The values above illustrate the manifest structure. A real plan must use the user's authorized settings and measured resource requirements. This release targets one server and GPU index 0. It shares available CPU/RAM and records conservative memory reservations. Exact CPU affinity allocation is outside this release.

With `runtime/scheduler/enabled` present, normal `scripts/run.py` calls submit and wait on the global scheduler before creating training artifacts. Its dry-run path remains a configuration preview. The two migrated queue scripts submit complete plans. Older scripts that launch through `run.py` also share the global limit, although their not-yet-submitted future steps are visible only when their script publishes them. Arbitrary training started directly outside these entry points is discovered conservatively and occupies visible slots. All new project entry points should submit a complete plan to obtain full queue visibility.

Training parameters and methods are defined in each task's protocol. The scheduler controls admission and records failures. It does not add training assertions, modify budgets, or stop training based on resource readings. Closing the local read-only monitor has no effect on dispatch.

## Scheduler retention

The runtime policy is `runtime/scheduler/retention.json`. The main SQLite database has a 512 MiB page limit, applied on each connection. Event history expires after 90 days. Task specifications, dependencies and exit summaries remain available for scheduling, duplicate detection and provenance. At 80% used capacity, the maintenance status raises a warning. Reaching the hard database limit can reject new writes, so this warning requires operator attention.

The user timer `marl-scheduler-maintenance.timer` runs daily at 03:30 server local time and catches up after downtime. It removes expired events and checkpoints available WAL pages. Only inactive `runtime/scheduler/logs/*.log` files are eligible for deletion, at 14 days or to keep inactive/active combined log use within 50 MiB where possible. Active logs are protected, so log usage may temporarily exceed this target. Experiment run directories, models and original logs are excluded.

The SQLite WAL has an 8 MiB post-checkpoint retention target. WAL, SHM, journal storage and active logs are additional to the main database limit. Maintenance writes a single replaceable `maintenance.json` status file. View results with `journalctl --user -u marl-scheduler-maintenance -n 10`.
