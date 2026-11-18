"""Seed3535 shared prefix then VDN baseline/type-history-union branches; max two processes."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ts3_vdn_A2_typeunion_3535_20261102'
SEEDS=[3535];GROUPS=['N','A2_union']
def parent(seed):return f'ts3_vdn_typeunion_Aprefix_seed{seed}_20261102_attempt001'
def branch(seed,g):return f'ts3_vdn_typeunion_{g}_seed{seed}_20261102_attempt001'
def cmd(seed,role,g='N'):
    run=parent(seed) if role=='prefix' else branch(seed,g)
    result=[str(ROOT/'.venv/bin/python'),'scripts/run.py','--suite','ts3','--algorithm','vdn','--method','baseline' if g=='N' else 'softprotect','--seed',str(seed),'--run-id',run,'--protocol','configs/continual/neuron_protection/softprotect/ts3_vdn_A2_typeunion_3535.json','--ts3-role',role,'--ts3-group',g]
    if role=='branch':result+=['--parent',str(ROOT/'runs'/parent(seed))]
    return run,result

def main():
    OUT.mkdir(parents=True,exist_ok=True);status=dict(status='running',pid=os.getpid(),max_parallel=2,runs={});todo=[(s,'prefix','N') for s in SEEDS];active={}
    def save():
        temp=OUT/'queue_status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(OUT/'queue_status.json')
    save()
    while todo or active:
        while todo and len(active)<2:
            seed,role,g=todo.pop(0);run,command=cmd(seed,role,g);p=subprocess.Popen(command,cwd=ROOT);active[run]=(p,seed,role);status['runs'][run]=dict(state='running',seed=seed,role=role,group=g,pid=p.pid);save()
        for run,(p,seed,role) in list(active.items()):
            code=p.poll()
            if code is not None:
                status['runs'][run].update(state='exited' if code==0 else 'failed',exit_code=code);del active[run]
                if role=='prefix':
                    if code==0:todo.extend((seed,'branch',g) for g in GROUPS)
                    else:
                        for g in GROUPS:status['runs'][branch(seed,g)]=dict(state='blocked_missing_prefix',seed=seed,role='branch',group=g)
                save()
        if active:time.sleep(5)
    code=subprocess.call([str(ROOT/'.venv/bin/python'),'scripts/summarize_ts3_vdn_typeunion.py'],cwd=ROOT)
    status.update(status='finished_pending_review' if code==0 and all(x.get('exit_code')==0 for x in status['runs'].values()) else 'finished_with_failures',summary_exit_code=code);save()
if __name__=='__main__':main()
