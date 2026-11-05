"""One-time handover of the two active authorized queues."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from core import ROOT, DEFAULT_DB, ACTIVE, connect, identity, read_json, read_tasks, submit
from legacy import QUEUES, build
from service import process_tree

def main():
    db=connect()
    supervisors=[]
    for kind,(folder,script,_) in QUEUES.items():
        old=read_json(ROOT/'results'/folder/'queue_status.json')
        pid=old['pid']
        argv=process_tree().get(pid,{}).get('argv',[])
        if not any(Path(arg).name==script for arg in argv):
            raise ValueError('Supervisor identity changed: '+script)
        supervisors.append(dict(kind=kind,pid=pid,folder=folder,identity=identity(pid)))
    frozen=[]
    committed=False
    try:
        for old in supervisors:
            os.kill(old['pid'],signal.SIGSTOP)
            frozen.append(old['pid'])
        for pid in frozen:
            for _ in range(40):
                if Path('/proc/{}/stat'.format(pid)).read_text().rsplit(')',1)[1].split()[0]=='T':
                    break
                time.sleep(.05)
            else:
                raise ValueError('Supervisor did not stop')
        all_items={}
        imported={}
        expected=set()
        for old in supervisors:
            plan,states=build(old['kind'])
            for item in plan['tasks']:
                if item['id'] in all_items and all_items[item['id']]!=item:
                    raise ValueError('Inconsistent shared parent')
                all_items[item['id']]=item
            imported.update(states)
            expected.update(n['pid'] for n in states.values() if n.get('pid'))
        children=set()
        for pid in frozen:
            children.update(int(x) for x in Path('/proc/{}/task/{}/children'.format(pid,pid)).read_text().split())
        if expected!=children:
            raise ValueError('Queue changed during handover: {} {}'.format(expected,children))
        audit=DEFAULT_DB.parent/'migration_20261105.json'
        backups=[]
        for old in supervisors:
            source=ROOT/'results'/old['folder']/'queue_status.json'
            target=source.with_name('queue_status.pre_unified_20261105.json')
            with target.open('xb') as output:
                data=source.read_bytes()
                output.write(data)
            backups.append(dict(path=str(source),backup=str(target),sha256=hashlib.sha256(data).hexdigest()))
        submit(db,dict(id='migration_20261105',tasks=list(all_items.values())),imported)
        with db:
            db.execute("UPDATE settings SET value='6' WHERE key='max_parallel'")
        active=[r for r in read_tasks(db) if r['state'] in ACTIVE]
        assert len(active)==len(expected)
        audit.write_text(json.dumps(dict(time=time.time(),supervisors=supervisors,adopted=sorted(expected),backups=backups),indent=2))
        (DEFAULT_DB.parent/'enabled').touch(exist_ok=False)
        committed=True
        for pid in frozen:
            os.kill(pid,signal.SIGKILL)
        print(json.dumps(dict(adopted=sorted(expected),tasks=len(all_items),limit=6)))
    finally:
        if not committed:
            for pid in frozen:
                os.kill(pid,signal.SIGCONT)


if __name__=='__main__':
    main()
