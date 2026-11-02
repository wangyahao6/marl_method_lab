"""Authorized TS4 seed4242 queue, max three concurrent, adopting existing runs."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ts4_vdn_agent_4242_20261102'
def alive(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[0]!='Z'
    except FileNotFoundError:return False
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'queue_status.json'
    state=json.loads(path.read_text()) if path.exists() else dict(runs={})
    state.update(status='running',pid=os.getpid(),max_parallel=3)
    def save():
        tmp=OUT/'queue_status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
    active={};todo=[]
    for group in ['N','A1_linear','A2_linear']:
        run=f'ts4_vdn_{group}_seed4242_20261102_attempt001'
        existing=state['runs'].get(run);runpath=ROOT/'runs'/run
        if existing and existing.get('state')=='running':
            active[run]=None
        elif not existing and not runpath.exists():todo.append((group,run))
        elif not existing:
            # Preserve existing artifacts; never overwrite or silently retry.
            state['runs'][run]=dict(group=group,state='failed',error='existing run without queue ownership',exit_code=1)
    save()
    while todo or active:
        while todo and len(active)<3:
            group,run=todo.pop(0)
            command=[str(ROOT/'.venv/bin/python'),'scripts/run.py','--suite','ts4','--algorithm','vdn','--method','baseline' if group=='N' else 'softprotect','--seed','4242','--run-id',run,'--protocol','configs/continual/ts4_vdn_4242.json','--ts4-group',group]
            proc=subprocess.Popen(command,cwd=ROOT)
            state['runs'][run]=dict(group=group,state='running',pid=proc.pid);active[run]=proc;save()
        for run,proc in list(active.items()):
            if proc is not None:
                code=proc.poll()
                if code is None:continue
            else:
                p=ROOT/'runs'/run/'launcher_status.json'
                info=json.loads(p.read_text()) if p.exists() else {}
                if info.get('state') in ['exited','failed','interrupted']:code=info.get('exit_code',1)
                elif alive(state['runs'][run]['pid']):continue
                else:code=1
            state['runs'][run].update(state='exited' if code==0 else 'failed',exit_code=code);del active[run];save()
        if active:time.sleep(5)
    code=subprocess.call([str(ROOT/'.venv/bin/python'),'scripts/summarize_ts4_vdn.py'],cwd=ROOT)
    state.update(status='finished_with_failures' if code or any(v.get('exit_code') for v in state['runs'].values()) else 'finished_pending_review',summary_exit_code=code);save()
if __name__=='__main__':main()
