"""Durable single-host scheduler and isolated execution wrapper."""
import argparse
import fcntl
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (ROOT, DEFAULT_DB, ACTIVE, TERMINAL, availability, connect,
                  descendants, identity, is_alive, now, process_tree, read_json,
                  read_tasks, reconcile, resources, submit, transaction, transition)


CHILDREN = []

def worker(database, attempt_id):
    db = connect(database)
    attempt = db.execute('SELECT * FROM attempts WHERE id=?', (attempt_id,)).fetchone()
    if not attempt:
        raise ValueError('Unknown attempt')
    path = Path(database).parent/'locks'
    path.mkdir(exist_ok=True)
    with (path/(attempt_id+'.lock')).open('a') as exclusive:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = identity(os.getpid())
        with transaction(db):
            row = db.execute('SELECT * FROM tasks WHERE id=?', (attempt['task'],)).fetchone()
            current = db.execute('SELECT * FROM attempts WHERE id=?', (attempt_id,)).fetchone()
            if row['state'] not in ACTIVE or current['ended'] is not None:
                return
            if current['pid'] and is_alive(current) and current['pid'] != os.getpid():
                raise ValueError('Attempt is already running')
            db.execute('UPDATE attempts SET pid=?,boot=?,start=? WHERE id=?',
                       (process['pid'], process['boot'], process['start'], attempt_id))
            transition(db, row['id'], 'running')
        spec = json.loads(row['spec'])
        env = dict(os.environ, MARL_SCHEDULER_ATTEMPT=attempt_id, MARL_SCHEDULER_DB=str(database))
        code = None
        try:
            with open(attempt['log'], 'ab', buffering=0) as stream:
                child = subprocess.Popen(spec['argv'], cwd=spec['cwd'], env=env,
                                         stdin=subprocess.DEVNULL, stdout=stream, stderr=stream)
                code = child.wait()
            state = 'completed' if code == 0 else 'failed'
            if state == 'completed' and spec['type'] == 'training' and spec.get('run_dir'):
                status = read_json(Path(spec['run_dir'])/'status.json')
                if status.get('status') not in (None, 'completed'):
                    state = 'unknown'
            with transaction(db):
                db.execute('UPDATE attempts SET ended=?,exit_code=? WHERE id=?', (now(), code, attempt_id))
                transition(db, row['id'], state, 'exit code {}'.format(code))
        except Exception:
            with transaction(db):
                transition(db, row['id'], 'unknown', traceback.format_exc())
                db.execute('UPDATE attempts SET ended=?,exit_code=? WHERE id=?', (now(), code, attempt_id))
            raise


def external_occupancy(processes, owned):
    """Count visible unmanaged train launchers/drivers once per process subtree."""
    candidates = set()
    for pid, proc in processes.items():
        if pid in owned:
            continue
        argv = proc['argv']
        scripts = [Path(x).name for x in argv[1:4]]
        if not scripts or not any('python' in Path(x).name for x in argv[:1]):
            continue
        if 'run.py' in scripts:
            # run.py is a submission client when the scheduler is enabled.
            continue
        if any(name.endswith('_train.py') or name in ('train.py', 'smac_entry.py', 'main.py') for name in scripts):
            candidates.add(pid)
    return {pid for pid in candidates if not any(
        other != pid and pid in descendants(other, processes) for other in candidates)}


def tick(db, database, launch=True):
    CHILDREN[:] = [child for child in CHILDREN if child.poll() is None]
    reconcile(db)
    tasks = read_tasks(db)
    attempts = {row['task']: row for row in db.execute('SELECT * FROM attempts')}
    active = [row for row in tasks if row['state'] in ACTIVE]
    processes = process_tree()
    owned = set()
    for row in active:
        attempt = attempts.get(row['id'])
        if attempt and is_alive(attempt):
            owned.update(descendants(attempt['pid'], processes))
    measured = resources()
    external = external_occupancy(processes, owned)
    external_covered = set(owned)
    for pid in external:
        external_covered.update(descendants(pid, processes))
    external.update(pid for pid in measured['gpu_processes'] if pid not in external_covered)
    limit = json.loads(db.execute("SELECT value FROM settings WHERE key='max_parallel'").fetchone()[0])
    states = {row['id']: row['state'] for row in tasks}
    for row in tasks:
        if row['state'] != 'pending':
            continue
        item = row['spec']
        parents = [states[parent] for parent in item['dependencies']]
        reason = ''
        failed_parents = any(state in ('failed', 'blocked') for state in parents)
        if failed_parents and item['dependency_mode'] == 'success':
            with transaction(db):
                transition(db, row['id'], 'blocked', 'dependency_failed')
            states[row['id']] = 'blocked'
            continue
        if item['dependency_mode'] == 'terminal':
            ready = all(state in TERMINAL for state in parents)
        else:
            ready = all(state == 'completed' for state in parents)
        if not ready:
            reason = 'dependencies'
        elif len(active)+len(external) >= limit:
            reason = 'global_limit'
        else:
            latest = max((attempt['created'] for task, attempt in attempts.items()
                          if any(t['id'] == task and t['spec']['queue'] == item['queue'] for t in tasks)), default=0)
            if now()-latest < item['cooldown_seconds']:
                reason = 'launch_cooldown'
            else:
                reason = availability(item, active, attempts, measured, processes)
        if not reason and item.get('run_dir') and Path(item['run_dir']).exists():
            reason = 'existing_output_requires_import'
        if reason:
            with transaction(db):
                transition(db, row['id'], 'pending', reason)
            continue
        if not launch:
            continue
        attempt_id = uuid.uuid4().hex
        log = str(Path(database).parent/'logs'/(attempt_id+'.log'))
        Path(log).parent.mkdir(exist_ok=True)
        with transaction(db):
            # Claims also serialize with external submitters.
            current = db.execute('SELECT state FROM tasks WHERE id=?', (row['id'],)).fetchone()
            if current['state'] != 'pending':
                continue
            db.execute('INSERT INTO attempts(id,task,created,log) VALUES (?,?,?,?)',
                       (attempt_id, row['id'], now(), log))
            transition(db, row['id'], 'starting', 'slot reserved')
        try:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--db', str(database),
                              'worker', attempt_id], cwd=str(ROOT), stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            CHILDREN.append(child)
        except OSError as exc:
            with transaction(db):
                transition(db, row['id'], 'failed', 'worker launch failed: '+str(exc))
                db.execute('UPDATE attempts SET ended=? WHERE id=?', (now(), attempt_id))
            continue
        row['state'] = 'starting'
        active.append(row)
        attempts[row['id']] = db.execute('SELECT * FROM attempts WHERE id=?', (attempt_id,)).fetchone()
    return dict(max_parallel=limit, active=len(active), unmanaged=len(external),
                unmanaged_pids=sorted(external), resources=measured)


def serve(database):
    directory = Path(database).parent
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'scheduler.lock').open('a') as exclusive:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = connect(database)
        stopping = []
        signal.signal(signal.SIGTERM, lambda *args: stopping.append(True))
        while not stopping:
            try:
                summary = tick(db, database)
                with transaction(db):
                    db.execute("INSERT OR REPLACE INTO settings VALUES ('heartbeat',?)",
                               (json.dumps(dict(summary, pid=os.getpid(), time=now())),))
            except Exception:
                traceback.print_exc()
            for _ in range(10):
                if stopping:
                    break
                time.sleep(.5)


def forward_run(run_id, argv, cwd, run_dir, task_type, metadata=None):
    """Called by the normal launcher. Wait for the central attempt's exit."""
    db = connect(Path(os.environ.get('MARL_SCHEDULER_DB', str(DEFAULT_DB))))
    token = os.environ.get('MARL_SCHEDULER_ATTEMPT')
    if token:
        attempt = db.execute('SELECT * FROM attempts WHERE id=?', (token,)).fetchone()
        if attempt and attempt['pid'] == os.getppid() and is_alive(attempt):
            row = db.execute('SELECT spec FROM tasks WHERE id=?', (attempt['task'],)).fetchone()
            if row and json.loads(row['spec'])['run_dir'] == str(run_dir):
                if metadata is None:
                    return False
                return json.loads(row['spec']).get('metadata', {})
    raw = dict(id=run_id, argv=argv, cwd=str(cwd), run_dir=str(run_dir), type=task_type)
    if metadata:
        raw['metadata'] = metadata
    existing = db.execute('SELECT spec FROM tasks WHERE id=?', (run_id,)).fetchone()
    if existing:
        spec = json.loads(existing['spec'])
        if spec['argv'] != argv or spec.get('run_dir') != str(run_dir):
            raise ValueError('Run ID already queued with different arguments')
    else:
        if run_dir.exists():
            raise FileExistsError('Existing run directory: '+str(run_dir))
        submit(db, dict(id='launcher', tasks=[raw]))
    while True:
        row = db.execute('SELECT state FROM tasks WHERE id=?', (run_id,)).fetchone()
        if row['state'] in TERMINAL:
            attempt = db.execute('SELECT exit_code FROM attempts WHERE task=?', (run_id,)).fetchone()
            code = attempt['exit_code'] if attempt else None
            raise SystemExit(code if isinstance(code, int) and code >= 0 else 1)
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('serve')
    worker_parser = commands.add_parser('worker')
    worker_parser.add_argument('attempt')
    submit_parser = commands.add_parser('submit')
    submit_parser.add_argument('plan', type=Path)
    limit_parser = commands.add_parser('limit')
    limit_parser.add_argument('value', type=int)
    commands.add_parser('status')
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.db, args.attempt)
    elif args.command == 'serve':
        serve(args.db)
    else:
        db = connect(args.db)
        if args.command == 'submit':
            submit(db, json.loads(args.plan.read_text()))
        elif args.command == 'limit':
            if args.value < 1:
                parser.error('positive slot count required')
            with transaction(db):
                db.execute("UPDATE settings SET value=? WHERE key='max_parallel'", (json.dumps(args.value),))
        else:
            rows = read_tasks(db)
            print(json.dumps(dict(settings=dict(db.execute('SELECT key,value FROM settings')),
                                  tasks=rows), indent=2))


if __name__ == '__main__':
    main()
