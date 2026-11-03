"""Authorized three-seed VDN S/Z/C prefix tree; at most two new workers."""
import itertools,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/vdn_task_history_20261103'
def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    master=json.loads((ROOT/'configs/continual/history.json').read_text());write(OUT/'protocol.json',master)
    tasks={'S':'stalker','Z':'zealot','C':'colossus'};nodes={}
    for length in (1,2,3):
        for seed in master['seeds']:
            for seq in itertools.permutations('SZC',length):
                path=''.join(seq);run=f'vdnh_{path}_seed{seed}_20261103_attempt001'
                parent=f'vdnh_{path[:-1]}_seed{seed}_20261103_attempt001' if length>1 else None
                cfg=dict(master,seed=seed,path=path,order=[tasks[x] for x in seq],thresholds=[2000000*i for i in range(1,length+1)])
                cp=OUT/f'{run}.json';write(cp,cfg)
                nodes[run]=dict(seed=seed,path=path,parent=parent,protocol=str(cp),state='pending')
    state=dict(status='running',pid=os.getpid(),max_parallel=master['max_parallel'],runs=nodes);active={}
    def save():write(OUT/'queue_status.json',state)
    save()
    while True:
        for run,proc in list(active.items()):
            code=proc.poll()
            if code is None:continue
            status_path=ROOT/'runs'/run/'status.json'
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            nodes[run].update(state='completed' if code==0 and status.get('status')=='completed' else 'failed',exit_code=code,t_env=status.get('t_env'));del active[run];save()
        for run,n in nodes.items():
            if n['state']!='pending':continue
            parent=n['parent']
            if parent and nodes[parent]['state'] in ['failed','dependency_failed']:
                n.update(state='dependency_failed',reason='parent training unavailable');save();continue
            if len(active)>=state['max_parallel']:continue
            if parent and nodes[parent]['state']!='completed':continue
            cmd=[str(ROOT/'.venv/bin/python'),'scripts/run.py','--suite','history','--algorithm','vdn','--method','baseline','--seed',str(n['seed']),'--run-id',run,'--protocol',n['protocol']]
            if parent:cmd+=['--parent',str(ROOT/'runs'/parent)]
            proc=subprocess.Popen(cmd,cwd=ROOT);active[run]=proc;n.update(state='running',pid=proc.pid);save()
        if not active and all(n['state']!='pending' for n in nodes.values()):break
        time.sleep(5)
    code=subprocess.call([str(ROOT/'.venv/bin/python'),'scripts/summarize_history.py'],cwd=ROOT)
    state.update(status='finished_pending_review' if all(n['state']=='completed' for n in nodes.values()) and code==0 else 'finished_with_failures',summary_exit_code=code);save()
if __name__=='__main__':main()
