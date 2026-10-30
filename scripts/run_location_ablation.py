"""Authorized four-location, single-seed queue; no validation training or retries."""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/location_ablation_seed1111'
LOCATIONS=['A','W','V','AWV']
RUNS={l:'qmix_switch_softprotect_'+l+'_seed1111_20261030_attempt002' for l in LOCATIONS}

def write(value):
    dest=OUT/'status.json';tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(dest)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',SDL_VIDEODRIVER='dummy',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
    pending=list(LOCATIONS);active={};done=[];failed=[];started=time.time()
    def state(phase):
        return dict(status='running',phase=phase,pid=os.getpid(),started=started,planned=4,
                    training_steps=16000000,completed=done,failed=failed,pending=pending,
                    active=[dict(location=l,run_id=RUNS[l],pid=p.pid) for l,p in active.items()])
    while pending or active:
        while pending and len(active)<2:
            location=pending.pop(0);run=ROOT/'runs'/RUNS[location]
            if run.exists():
                failed.append(dict(location=location,reason='run directory already exists; preserved without automatic retry'));continue
            command=[sys.executable,'scripts/run.py','--suite','continual','--algorithm','qmix','--method','softprotect','--scenario','task_switch','--seed','1111','--run-id',RUNS[location],'--protocol','configs/continual/location_'+location+'_switch_1111.json']
            with (OUT/(location+'_launcher.log')).open('w') as log:
                active[location]=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        write(state('training'));time.sleep(15)
        for location,process in list(active.items()):
            code=process.poll()
            if code is None:continue
            path=ROOT/'runs'/RUNS[location]/'status.json'
            result=json.loads(path.read_text()) if path.exists() else {}
            if code==0 and result.get('status')=='completed':done.append(location)
            else:failed.append(dict(location=location,exit_code=code,state=result))
            del active[location]
    eval_failures=[]
    for location in done:
        value=state('independent_final_evaluation');value['evaluating']=location;write(value)
        with (OUT/(location+'_evaluation.log')).open('w') as log:
            code=subprocess.call([sys.executable,'scripts/evaluate.py','--run',str(ROOT/'runs'/RUNS[location]),'--episodes','32'],stdout=log,stderr=subprocess.STDOUT)
        if code:eval_failures.append(dict(location=location,exit_code=code))
    code=subprocess.call([sys.executable,'scripts/summarize_location_ablation.py'])
    value=state('finished');value.update(status='completed' if not failed and not eval_failures and code==0 else 'finished_with_failures',finished=time.time(),evaluation_failures=eval_failures,summary_exit_code=code)
    write(value)
if __name__=='__main__':main()
