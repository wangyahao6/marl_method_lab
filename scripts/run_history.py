"""Authorized three-seed VDN S/Z/C prefix tree; at most two new workers."""
import argparse,itertools,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/vdn_task_history_20261103'
def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
def initialize():
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
    return dict(status='running',pid=os.getpid(),max_parallel=master['max_parallel'],runs=nodes)

class AdoptedLauncher:
    def __init__(self, run, pid):
        self.run, self.pid = run, pid

    def poll(self):
        directory=ROOT/'runs'/self.run
        try:
            launcher=json.loads((directory/'launcher_status.json').read_text())
        except (OSError, ValueError):
            launcher={}
        if launcher.get('exit_code') is not None:
            return launcher['exit_code']
        try:
            stat=Path(f'/proc/{self.pid}/stat').read_text().rsplit(')',1)[1].split()
            argv=Path(f'/proc/{self.pid}/cmdline').read_bytes().split(b'\0')
            if stat[0]!='Z' and self.run.encode() in argv:
                return None
        except FileNotFoundError:
            pass
        # The launcher writes its exit status before exiting.
        try:
            return json.loads((directory/'launcher_status.json').read_text()).get('exit_code',1)
        except (OSError, ValueError):
            return 1

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-parallel', type=int)
    options=parser.parse_args()
    if options.max_parallel is not None and options.max_parallel < 1:
        parser.error('--max-parallel must be positive')
    if options.resume:
        state=json.loads((OUT/'queue_status.json').read_text())
        old=Path(f"/proc/{state['pid']}/stat")
        if old.exists() and old.read_text().rsplit(')',1)[1].split()[0] not in ('T','Z'):
            parser.error('previous scheduler must be stopped before handover')
    else:
        state=initialize()
    state.update(pid=os.getpid(),status='running')
    if options.max_parallel is not None:
        state['max_parallel']=options.max_parallel
    nodes=state['runs']
    active={run:AdoptedLauncher(run,node['pid']) for run,node in nodes.items() if node['state']=='running'}
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
