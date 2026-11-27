"""Export metadata read-only for the global index; never import training code."""
import argparse
import datetime
import json
import sqlite3
from pathlib import Path
from sanitize_experiment_metadata import redact


STATUS_FILES = ('status.json', 'launcher_status.json', 'training_complete.json')
LAUNCH_FIELDS = ('request', 'details', 'git_commit', 'started', 'output', 'metadata',
                 'command', 'python', 'sc2path', 'git_status')
ARCHITECTURE_FILES = ('src/modules/agents/rnn_agent.py',
                      'src/modules/agents/single_ff_rnn_agent.py',
                      'src/modules/agents/__init__.py', 'src/modules/mixers/vdn.py',
                      'src/modules/layers/act_layer.py')


def read(path):
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def emit(record):
    print(json.dumps(redact(record), sort_keys=True))


def collect(roots, host):
    emit(dict(kind='collection', host=host,
              captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              roots=[str(root) for root in roots]))
    seen_runs, seen_dbs = set(), set()
    for root in roots:
        if not root.exists():
            continue
        paths = set(root.glob('runs/**/launch.json')) | set(root.glob('runs/**/status.json'))
        for directory in sorted({path.parent for path in paths}):
            real = directory.resolve()
            if real in seen_runs:
                continue
            seen_runs.add(real)
            original = read(directory / 'launch.json')
            launch = {key: original[key] for key in LAUNCH_FIELDS if key in original}
            if 'source_sha256' in original:
                launch['source_sha256'] = {key: value for key, value in original['source_sha256'].items()
                                          if key in ARCHITECTURE_FILES}
            if 'details' in launch:
                launch['details'] = {key: value for key, value in launch['details'].items()
                                     if key in ('protocol', 'steps', 'parent', 'method', 'seed')}
            statuses = {name: read(directory / name) for name in STATUS_FILES
                        if (directory / name).is_file()}
            emit(dict(kind='runtime', host=host, root=str(root), run_dir=str(directory),
                      run_id=directory.name, launch=launch,
                      config=read(directory / 'config.json'),
                      protocol=read(directory / 'protocol.json'), status_documents=statuses))
        database = root / 'runtime/scheduler/state.sqlite3'
        if not database.is_file() or database.resolve() in seen_dbs:
            continue
        seen_dbs.add(database.resolve())
        connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            query = ('SELECT t.id,t.state,t.spec,a.created,a.ended,a.exit_code '
                     'FROM tasks t LEFT JOIN attempts a ON t.id=a.task ORDER BY t.id,a.created')
            for task, state, spec, started, ended, code in connection.execute(query):
                if started is None:
                    continue
                spec = json.loads(spec)
                kind = 'training' if spec.get('type') == 'training' else 'analysis'
                emit(dict(kind='scheduler_' + kind, host=host, root=str(root),
                          analysis_id=task, state=state, spec=spec,
                          started=started, ended=ended, exit_code=code))
        finally:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', type=Path, required=True)
    parser.add_argument('--host', required=True)
    args = parser.parse_args()
    collect(args.root, args.host)


if __name__ == '__main__':
    main()
