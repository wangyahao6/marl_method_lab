"""Bounded retention for scheduler metadata and scheduler-owned inactive logs."""
import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

DEFAULTS = dict(database_mib=512, event_days=90, log_days=14, log_mib=50)


def maintain(database, policy, moment=None):
    moment = time.time() if moment is None else moment
    directory = Path(database).parent
    db = sqlite3.connect(str(database), timeout=20)
    try:
        db.execute('PRAGMA busy_timeout=20000')
        with db:
            removed = db.execute('DELETE FROM events WHERE time < ?',
                                 (moment-policy['event_days']*86400,)).rowcount
            db.execute('PRAGMA incremental_vacuum(1024)')
        # Checkpoint only available pages without waiting for live readers.
        checkpoint = db.execute('PRAGMA wal_checkpoint(PASSIVE)').fetchone()
        protected = {Path(row[0]).resolve() for row in db.execute(
            "SELECT a.log FROM attempts a JOIN tasks t ON a.task=t.id "
            "WHERE a.ended IS NULL OR t.state IN ('starting','running')")}
        logs = directory/'logs'
        files = sorted((p for p in logs.glob('*.log') if p.is_file() and not p.is_symlink()),
                       key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        log_removed = 0
        for path in files:
            size = path.stat().st_size
            if path.resolve() in protected:
                continue
            if path.stat().st_mtime < moment-policy['log_days']*86400 or total > policy['log_mib']*1048576:
                path.unlink()
                total -= size
                log_removed += 1
        page = db.execute('PRAGMA page_size').fetchone()[0]
        used = (db.execute('PRAGMA page_count').fetchone()[0]
                - db.execute('PRAGMA freelist_count').fetchone()[0])*page
        result = dict(time=moment, events_removed=removed, logs_removed=log_removed,
                      scheduler_log_bytes=total, database_used_bytes=used,
                      database_mib=policy['database_mib'], checkpoint=list(checkpoint),
                      warning=used >= policy['database_mib']*1048576*.8
                      or total > policy['log_mib']*1048576)
        temporary = directory/('maintenance.'+str(os.getpid())+'.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(directory/'maintenance.json')
        return result
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path,
                        default=Path(__file__).resolve().parents[1]/'runtime/scheduler/state.sqlite3')
    args = parser.parse_args()
    config = args.db.parent/'retention.json'
    policy = dict(DEFAULTS)
    if config.exists():
        policy.update(json.loads(config.read_text()))
    if any(type(v) is not int or v < 1 for v in policy.values()):
        parser.error('retention values must be positive integers')
    print(json.dumps(maintain(args.db, policy)))


if __name__ == '__main__':
    main()
