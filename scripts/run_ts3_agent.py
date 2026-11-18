"""Authorized TS3 shared prefix then five branches, max five concurrent."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ts3_agent_3535_20261102';PARENT='ts3_Aprefix_seed3535_20261102_attempt001'
GROUPS=['N','A1_linear','A1_union','A2_linear','A2_union']
def cmd(run,role,group='N'):
    result=[str(ROOT/'.venv/bin/python'),'scripts/run.py','--suite','ts3','--algorithm','qmix','--method','baseline' if group=='N' else 'softprotect','--seed','3535','--run-id',run,'--protocol','configs/continual/neuron_protection/softprotect/ts3_agent_3535.json','--ts3-role',role,'--ts3-group',group]
    if role=='branch':result+=['--parent',str(ROOT/'runs'/PARENT)]
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    status=dict(status='running',pid=os.getpid(),max_concurrent_branches=5,prefix=dict(run_id=PARENT,state='running'),branches={})
    def save():
        tmp=OUT/'queue_status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(OUT/'queue_status.json')
    save();code=subprocess.call(cmd(PARENT,'prefix'),cwd=ROOT)
    status['prefix'].update(state='exited' if code==0 else 'failed',exit_code=code);save()
    if code!=0:
        status['status']='prefix_failed';status['branches']={g:dict(state='blocked_missing_shared_prefix') for g in GROUPS};save();return
    todo=list(GROUPS);active={}
    while todo or active:
        while todo and len(active)<5:
            g=todo.pop(0);run=f'ts3_{g}_seed3535_20261102_attempt001';p=subprocess.Popen(cmd(run,'branch',g),cwd=ROOT);active[run]=p;status['branches'][run]=dict(group=g,state='running',pid=p.pid);save()
        for run,p in list(active.items()):
            code=p.poll()
            if code is not None:status['branches'][run].update(state='exited' if code==0 else 'failed',exit_code=code);del active[run];save()
        if active:time.sleep(5)
    status['status']='training_processes_exited';save()
    code=subprocess.call([str(ROOT/'.venv/bin/python'),'scripts/summarize_ts3_agent.py'],cwd=ROOT)
    status['summary_exit_code']=code;status['status']='finished_with_failures' if code or any(v.get('exit_code') for v in status['branches'].values()) else 'finished_pending_review';save()
if __name__=='__main__':main()
