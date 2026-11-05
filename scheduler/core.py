"""Single-host read/write scheduling state. Training artifacts remain authoritative."""
import contextlib
import datetime
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / 'runtime/scheduler/state.sqlite3'
ACTIVE = ('starting', 'running')
TERMINAL = ('completed', 'failed', 'blocked', 'unknown')


def now():
    return time.time()


def connect(path=DEFAULT_DB):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), timeout=20)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript("""
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, spec TEXT NOT NULL,
            state TEXT NOT NULL, reason TEXT, enqueued REAL NOT NULL,
            updated REAL NOT NULL, priority INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS dependencies (
            task TEXT NOT NULL REFERENCES tasks(id), parent TEXT NOT NULL REFERENCES tasks(id),
            PRIMARY KEY(task,parent));
        CREATE TABLE IF NOT EXISTS attempts (
            id TEXT PRIMARY KEY, task TEXT NOT NULL UNIQUE REFERENCES tasks(id),
            pid INTEGER, boot TEXT, start TEXT, created REAL NOT NULL, ended REAL,
            exit_code INTEGER, log TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (
            seq INTEGER PRIMARY KEY, task TEXT, time REAL NOT NULL, state TEXT, detail TEXT);
    """)
    with db:
        db.execute("INSERT OR IGNORE INTO settings VALUES ('max_parallel','6')")
    return db


@contextlib.contextmanager
def transaction(db):
    db.execute('BEGIN IMMEDIATE')
    try:
        yield
        db.commit()
    except BaseException:
        db.rollback()
        raise


def transition(db, task, state, detail=''):
    row = db.execute('SELECT state,reason FROM tasks WHERE id=?', (task,)).fetchone()
    if row['state'] == state and (row['reason'] or '') == detail:
        return
    db.execute('UPDATE tasks SET state=?, reason=?, updated=? WHERE id=?', (state, detail, now(), task))
    db.execute('INSERT INTO events(task,time,state,detail) VALUES (?,?,?,?)', (task, now(), state, detail))


def identity(pid):
    try:
        data = Path('/proc/{}/stat'.format(pid)).read_text().rsplit(')', 1)[1].split()
        if data[0] == 'Z':
            return None
        return dict(pid=pid, boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), start=data[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def is_alive(attempt):
    if not attempt or not attempt['pid']:
        return False
    current = identity(attempt['pid'])
    return current and all(current[key] == attempt[key] for key in ('pid', 'boot', 'start'))


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_tasks(db):
    return [dict(row, spec=json.loads(row['spec'])) for row in db.execute(
        'SELECT * FROM tasks ORDER BY priority DESC,enqueued,rowid')]


def validate_plan(plan):
    tasks = plan['tasks']
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('tasks must be a nonempty list')
    ids = [item['id'] for item in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate task IDs')
    normalized = []
    for raw in tasks:
        item = dict(raw)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', item['id']):
            raise ValueError('invalid task ID')
        if not isinstance(item.get('argv'), list) or not item['argv'] or not all(
                isinstance(arg, str) and '\0' not in arg for arg in item['argv']):
            raise ValueError('argv must be a nonempty string list')
        if not Path(item['cwd']).is_absolute() or not Path(item['cwd']).is_dir():
            raise ValueError('cwd must be an existing absolute directory')
        if item.get('run_dir') and not Path(item['run_dir']).is_absolute():
            raise ValueError('run_dir must be absolute')
        item.setdefault('dependencies', [])
        if not isinstance(item['dependencies'], list) or not all(isinstance(x, str) for x in item['dependencies']):
            raise ValueError('dependencies must be a list of IDs')
        item.setdefault('queue', plan.get('id', 'default'))
        item.setdefault('queue_limit', 6)
        item.setdefault('cooldown_seconds', 0)
        item.setdefault('priority', 0)
        item.setdefault('type', 'training')
        item.setdefault('dependency_mode', 'success')
        if item['dependency_mode'] not in ('success', 'terminal'):
            raise ValueError('invalid dependency_mode')
        for name in ('queue_limit',):
            if type(item[name]) is not int or item[name] < 1:
                raise ValueError(name + ' must be positive')
        if type(item['priority']) is not int:
            raise ValueError('priority must be integer')
        if not isinstance(item['cooldown_seconds'], (int, float)) or item['cooldown_seconds'] < 0:
            raise ValueError('invalid cooldown')
        item.setdefault('resources', {'gpu_mib': 1300, 'ram_mib': 2000})
        resources = item['resources']
        for key in ('gpu_mib', 'ram_mib', 'min_free_gpu_mib', 'min_available_ram_mib'):
            if type(resources.get(key, 0)) is not int or resources.get(key, 0) < 0:
                raise ValueError('resource values must be nonnegative integers')
        if resources.get('gpu', 0) != 0:
            raise ValueError('first release supports GPU index 0')
        item.setdefault('mutex', str(Path(item['run_dir']).resolve()) if item.get('run_dir') else None)
        normalized.append(item)
    return normalized


def submit(db, plan, imports=None):
    items = validate_plan(plan)
    imports = imports or {}
    with transaction(db):
        existing = {row['id']: json.loads(row['spec']) for row in db.execute('SELECT id,spec FROM tasks')}
        graph = dict(existing)
        graph.update({item['id']: item for item in items})
        visiting, visited = set(), set()

        def visit(key):
            if key in visiting:
                raise ValueError('dependency cycle')
            if key in visited:
                return
            if key not in graph:
                raise ValueError('unknown dependency: ' + key)
            visiting.add(key)
            for parent in graph[key]['dependencies']:
                visit(parent)
            visiting.remove(key)
            visited.add(key)
        for key in graph:
            visit(key)
        for index, item in enumerate(items):
            raw = json.dumps(item, sort_keys=True, ensure_ascii=True)
            fingerprint = hashlib.sha256(raw.encode()).hexdigest()
            row = db.execute('SELECT fingerprint FROM tasks WHERE id=?', (item['id'],)).fetchone()
            if row:
                if row['fingerprint'] != fingerprint:
                    raise ValueError('task ID already has another plan: ' + item['id'])
                continue
            imported = imports.get(item['id'], {})
            state = imported.get('state', 'pending')
            if state not in ('pending', *ACTIVE, *TERMINAL):
                raise ValueError('invalid imported state')
            db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?)',
                       (item['id'], fingerprint, raw, state, '', now()+index*0.000001, now(), item['priority']))
            db.execute('INSERT INTO events(task,time,state,detail) VALUES (?,?,?,?)',
                       (item['id'], now(), state, 'imported' if imported else 'submitted'))
            if imported.get('pid'):
                process = identity(imported['pid'])
                if not process:
                    raise ValueError('imported process disappeared: ' + item['id'])
                log = str(Path(item['run_dir'])/'console.log')
                db.execute('INSERT INTO attempts(id,task,pid,boot,start,created,log) VALUES (?,?,?,?,?,?,?)',
                           ('adopted-'+item['id'], item['id'], process['pid'], process['boot'], process['start'], now(), log))
        for item in items:
            for parent in item['dependencies']:
                db.execute('INSERT OR IGNORE INTO dependencies VALUES (?,?)', (item['id'], parent))


def resources():
    memory = {line.split(':')[0]: int(line.split()[1])//1024
              for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
    result = dict(ram_mib=memory['MemAvailable'], gpu_mib=None, gpu_processes={})
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                          '--format=csv,noheader,nounits'], text=True, timeout=5)
        result['gpu_mib'] = int(output.splitlines()[0])
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                          '--format=csv,noheader,nounits'], text=True, timeout=5)
        for line in output.splitlines():
            pid, used = line.split(',')
            if pid.strip().isdigit() and used.strip().isdigit():
                result['gpu_processes'][int(pid)] = int(used)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result['gpu_error'] = str(exc)
    return result


def process_tree():
    rows = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = (directory/'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z' or directory.stat().st_uid != os.getuid():
                continue
            argv = (directory/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            rows[int(directory.name)] = dict(parent=int(stat[1]), rss=int(stat[21])*os.sysconf('SC_PAGE_SIZE')//1048576, argv=argv)
        except (OSError, ValueError):
            continue
    return rows


def descendants(pid, processes):
    found = {pid}
    while True:
        extra = {key for key, value in processes.items() if value['parent'] in found}
        if extra <= found:
            return found
        found |= extra


def availability(item, active, attempts, measured, processes):
    queue = [row for row in active if row['spec']['queue'] == item['queue']]
    if len(queue) >= item['queue_limit']:
        return 'queue_limit'
    if item.get('mutex') and any(row['spec'].get('mutex') == item['mutex'] for row in active):
        return 'mutex'
    unallocated_gpu = unallocated_ram = 0
    for row in active:
        spec = row['spec']
        attempt = attempts.get(row['id'])
        pids = descendants(attempt['pid'], processes) if attempt and attempt['pid'] else set()
        allocation = spec['resources']
        actual_gpu = sum(measured['gpu_processes'].get(pid, 0) for pid in pids)
        actual_ram = sum(processes.get(pid, {}).get('rss', 0) for pid in pids)
        unallocated_gpu += max(0, allocation.get('gpu_mib', 0)-actual_gpu)
        unallocated_ram += max(0, allocation.get('ram_mib', 0)-actual_ram)
    required = item['resources']
    if measured['ram_mib']-unallocated_ram < max(required.get('ram_mib', 0)+2048,
                                                required.get('min_available_ram_mib', 0)):
        return 'ram'
    if required.get('gpu_mib', 0):
        if measured['gpu_mib'] is None:
            return 'gpu_probe_unavailable'
        if measured['gpu_mib']-unallocated_gpu < max(required['gpu_mib']+512, required.get('min_free_gpu_mib', 0)):
            return 'gpu_memory'
    return ''


def reconcile(db):
    processes = process_tree()
    for row in read_tasks(db):
        if row['state'] not in ACTIVE:
            continue
        attempt = db.execute('SELECT * FROM attempts WHERE task=?', (row['id'],)).fetchone()
        if attempt and attempt['ended'] is not None:
            continue
        if attempt and is_alive(attempt):
            continue
        if attempt and not attempt['pid']:
            matches = [pid for pid, p in processes.items()
                       if 'worker' in p['argv'] and attempt['id'] in p['argv']]
            if matches:
                process = identity(matches[0])
                if process:
                    with transaction(db):
                        db.execute('UPDATE attempts SET pid=?,boot=?,start=? WHERE id=?',
                                   (process['pid'], process['boot'], process['start'], attempt['id']))
                continue
            if now()-attempt['created'] < 30:
                continue
        artifact = Path(row['spec']['run_dir']) if row['spec'].get('run_dir') else None
        launcher = read_json(artifact/'launcher_status.json') if artifact else {}
        status = read_json(artifact/'status.json') if artifact else {}
        code = launcher.get('exit_code')
        if code == 0 and (row['spec']['type'] != 'training' or status.get('status') == 'completed'):
            state = 'completed'
        elif code is not None or status.get('status') == 'failed':
            state = 'failed'
        else:
            state = 'unknown'
        with transaction(db):
            transition(db, row['id'], state, 'reconciled execution records')
            db.execute('UPDATE attempts SET ended=?,exit_code=? WHERE task=?', (now(), code, row['id']))
