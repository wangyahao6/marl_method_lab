"""Authorized local-GEM method branches only; resource-aware max5 supervisor."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ts3_vdn_A2_local_gem_20261104'

def command(item):
    return [str(ROOT/'.venv/bin/python'),'scripts/run.py','--suite','ts3','--algorithm','vdn','--method','gem' if item['role']=='branch' else 'baseline','--seed',str(item['seed']),'--run-id',item['run'],'--protocol',item['protocol'],'--ts3-role',item['role'],'--ts3-group',item['group'],'--parent',str(ROOT/'runs'/item['parent'])]

def resources():
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    processes=sum(x.strip().isdigit() for x in lines)
    ram=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])//1024
    return dict(free_gpu_mib=free,gpu_compute_processes=processes,available_ram_mib=ram,time=time.time())

def main():
    if (ROOT/'runtime/scheduler/enabled').exists() and '--dry-run' not in sys.argv:
        sys.path.insert(0,str(ROOT/'scheduler'))
        from legacy import submit_queue
        return submit_queue('local_gem')
    plan=json.loads((OUT/'plan.json').read_text())
    if '--dry-run' in sys.argv:
        for item in plan['items']:
            if subprocess.call(command(item)+['--dry-run'],cwd=ROOT):return 1
        return 0
    # An OS lock prevents two copies of this new supervisor dispatching duplicates.
    import fcntl
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(state='running',pid=os.getpid(),started=time.time(),max_parallel=plan['max_parallel'],runs={})
    pending=[];active={}
    for item in plan['items']:
        existing=ROOT/'runs'/item['run']
        if existing.exists():
            state=json.loads((existing/'launcher_status.json').read_text()) if (existing/'launcher_status.json').exists() else {}
            status['runs'][item['run']]=dict(state='existing_not_relaunched',launcher=state,seed=item['seed'],band=item['band'],role=item['role'])
        else:
            pending.append(item);status['runs'][item['run']]=dict(state='queued_resources',seed=item['seed'],band=item['band'],role=item['role'])
    def save():
        temp=OUT/'queue_status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(OUT/'queue_status.json')
    save();last_launch=0
    while pending or active:
        for run,p in list(active.items()):
            code=p.poll()
            if code is not None:
                actual=ROOT/'runs'/run/'status.json';training=json.loads(actual.read_text()) if actual.exists() else {}
                status['runs'][run].update(state='completed_pending_review' if code==0 and training.get('status')=='completed' else 'failed',exit_code=code,training_status=training,ended=time.time())
                del active[run];save()
        try:
            available=resources();status['resources']=available
            ready=(available['free_gpu_mib']>=plan['min_free_gpu_mib'] and available['available_ram_mib']>=plan['min_available_ram_mib'] and available['gpu_compute_processes']<plan['max_gpu_compute_processes'])
            if pending and len(active)<plan['max_parallel'] and ready and time.time()-last_launch>=90:
                item=pending.pop(0);run=item['run']
                try:
                    p=subprocess.Popen(command(item),cwd=ROOT);active[run]=p
                    status['runs'][run].update(state='running',launcher_pid=p.pid,started=time.time());last_launch=time.time()
                except Exception:status['runs'][run].update(state='launch_failed',error=traceback.format_exc())
            status['state']='running' if active else 'queued_resources' if pending else 'summarizing'
            status['pending']=len(pending);status['active']=len(active);save()
        except Exception:
            status['resource_probe_error']=traceback.format_exc();save()
        if pending or active:time.sleep(20)
    code=subprocess.call([str(ROOT/'.venv/bin/python'),'scripts/summarize_ts3_vdn_local_gem.py'],cwd=ROOT)
    status.update(state='finished_pending_review' if code==0 and all(x.get('state')=='completed_pending_review' for x in status['runs'].values()) else 'finished_with_failures',summary_exit_code=code,ended=time.time());save()
    return code
if __name__=='__main__':sys.exit(main())
