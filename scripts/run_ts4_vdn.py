"""Authorized TS4 seed4242 queue; one process at a time; independent failures."""
import json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ts4_vdn_agent_4242_20261102'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',pid=os.getpid(),max_parallel=1,runs={})
    def save():
        tmp=OUT/'queue_status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(OUT/'queue_status.json')
    save()
    for group in ['N','A1_linear','A2_linear']:
        run=f'ts4_vdn_{group}_seed4242_20261102_attempt001'
        command=[str(ROOT/'.venv/bin/python'),'scripts/run.py','--suite','ts4','--algorithm','vdn','--method','baseline' if group=='N' else 'softprotect','--seed','4242','--run-id',run,'--protocol','configs/continual/ts4_vdn_4242.json','--ts4-group',group]
        proc=subprocess.Popen(command,cwd=ROOT)
        state['runs'][run]=dict(group=group,state='running',pid=proc.pid);save()
        code=proc.wait()
        state['runs'][run].update(state='exited' if code==0 else 'failed',exit_code=code);save()
    code=subprocess.call([str(ROOT/'.venv/bin/python'),'scripts/summarize_ts4_vdn.py'],cwd=ROOT)
    state.update(status='finished_with_failures' if code or any(v.get('exit_code') for v in state['runs'].values()) else 'finished_pending_review',summary_exit_code=code);save()
if __name__=='__main__':main()
