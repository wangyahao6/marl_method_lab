"""Convert authorized experiment plans into complete scheduler manifests."""
import importlib.util
import sys
import time
from pathlib import Path
from core import ROOT, TERMINAL, connect, read_json, submit
from service import process_tree

QUEUES = {
    'local_gem': ('ts3_vdn_A2_local_gem_20261104', 'run_ts3_vdn_local_gem.py', 'summarize_ts3_vdn_local_gem.py'),
    'selection_matched': ('ts3_vdn_selection_matched_20261104', 'run_ts3_vdn_selection_matched.py', 'summarize_ts3_vdn_selection_matched.py'),
}


def build(kind):
    folder, script, summary = QUEUES[kind]
    out = ROOT/'results'/folder
    plan = read_json(out/'plan.json')
    loader = importlib.util.spec_from_file_location('legacy_'+kind, ROOT/'scripts'/script)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    tasks, parents = [], set()
    for item in plan['items']:
        parent = item['parent']
        parents.add(parent)
        required = dict(gpu_mib=1300, ram_mib=2000)
        if kind == 'local_gem':
            required.update(min_free_gpu_mib=plan['min_free_gpu_mib'],
                            min_available_ram_mib=plan['min_available_ram_mib'])
        tasks.append(dict(id=item['run'], argv=module.command(item), cwd=str(ROOT),
                          run_dir=str(ROOT/'runs'/item['run']), dependencies=[parent],
                          queue=folder, queue_limit=plan['max_parallel'],
                          cooldown_seconds=90 if kind == 'local_gem' else 0,
                          type='evaluation' if item.get('role') == 'eval_fwt' else 'training',
                          metadata=dict(item, results=str(out)), resources=required))
    all_runs = [t['id'] for t in tasks]
    tasks.append(dict(id=folder+'_summary', argv=[str(ROOT/'.venv/bin/python'), 'scripts/'+summary],
                      cwd=str(ROOT), dependencies=all_runs, dependency_mode='terminal',
                      queue=folder, queue_limit=1, type='summary', mutex=str(out),
                      resources=dict(gpu_mib=0, ram_mib=512), metadata=dict(results=str(out))))
    imports = {}
    for parent in sorted(parents):
        directory = ROOT/'runs'/parent
        if read_json(directory/'status.json').get('status') != 'completed':
            raise ValueError('Parent has no completed training record: '+parent)
        tasks.insert(0, dict(id=parent, argv=['/usr/bin/true'], cwd=str(ROOT), run_dir=str(directory),
                             type='training', queue='imported_prefixes',
                             resources=dict(gpu_mib=0, ram_mib=0)))
        imports[parent] = dict(state='completed')
    processes = process_tree()
    for item in tasks:
        if not item.get('run_dir') or item['id'] in imports:
            continue
        matches = [pid for pid, proc in processes.items()
                   if item['id'] in proc['argv'] and any(Path(arg).name == 'run.py' for arg in proc['argv'][:3])]
        if len(matches) > 1:
            raise ValueError('Duplicate live run ID: '+item['id'])
        if matches:
            imports[item['id']] = dict(state='running', pid=matches[0])
        elif Path(item['run_dir']).exists():
            launcher = read_json(Path(item['run_dir'])/'launcher_status.json')
            status = read_json(Path(item['run_dir'])/'status.json')
            if launcher.get('exit_code') == 0 and status.get('status') == 'completed':
                imports[item['id']] = dict(state='completed')
            elif status.get('status') == 'failed' or launcher.get('exit_code') not in (None, 0):
                imports[item['id']] = dict(state='failed')
            else:
                imports[item['id']] = dict(state='unknown')
    return dict(id=folder, tasks=tasks), imports


def submit_queue(kind):
    db = connect()
    plan, imports = build(kind)
    submit(db, plan, imports)
    summary = plan['id']+'_summary'
    print('Queue submitted to central scheduler: '+plan['id'], flush=True)
    while True:
        state = db.execute('SELECT state FROM tasks WHERE id=?', (summary,)).fetchone()['state']
        if state in TERMINAL:
            return 0 if state == 'completed' else 1
        time.sleep(5)


if __name__ == '__main__':
    raise SystemExit(submit_queue(sys.argv[1]))
