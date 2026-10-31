"""User-authorized 2.5M-step ABA comparison, seeds 3535 and 2222."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
import run_aba as summary
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/aba_WV_A2_1M_seeds3535_2222_20261031'
PROTOCOL='configs/continual/aba_WV_A2_1M_3535_2222.json'
SEEDS=[3535,2222]
JOBS=[dict(seed=seed,method=m,run_id=f'qmix_ABA_A2_1M_{m}_seed{seed}_20261031') for seed in SEEDS for m in ['baseline','softprotect']]

def write(v):
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(OUT/'status.json')

def summarize():
    errors=[]
    for seed in SEEDS:
        try:
            summary.OUT=OUT/f'seed{seed}'
            summary.RUNS={j['method']:j['run_id'] for j in JOBS if j['seed']==seed}
            summary.summarize()
        except Exception:
            errors.append(dict(seed=seed,traceback=traceback.format_exc()))
    lines=['# ABA: return A for 1M steps','',
           'QMIX baseline versus W+V soft protection; seeds 3535, 2222. Each independently initialized run trains Stalker 1M → Zealot 500k → Stalker 1M (2.5M each, 10M total). Protection and screening settings inherit the previous ABA experiment; B only, immediate release at return A. Retain replay, target, optimizer and cumulative exploration.','',
           '| Seed | Method | Training state | Steps |','| ---: | --- | --- | ---: |']
    for j in JOBS:
        p=ROOT/'runs'/j['run_id']/'status.json';s=json.loads(p.read_text()) if p.exists() else {}
        lines.append(f"| {j['seed']} | {j['method']} | {s.get('status','not_started')} | {s.get('t_env',0)} |")
    for seed in SEEDS:
        lines+=['',f'## Seed {seed}','',f'[Full results and 32-episode evaluation data](seed{seed}/REPORT.md)','',f'![Learning and return-A recovery, seed {seed}](seed{seed}/learning_and_recovery.png)']
    lines+=['','Curves are shown separately for each training seed. The previous seed1111 2M-step experiment is a different schedule and is not pooled into these results. Raw artifacts remain in the remote runs directories.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    if errors:(OUT/'summary_errors.json').write_text(json.dumps(errors,indent=2))
    return errors

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',SDL_VIDEODRIVER='dummy',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
    pending=list(JOBS);active={};done=[];failed=[];eval_failures=[];started=time.time()
    def state(phase):return dict(status='running',phase=phase,pid=os.getpid(),started=started,planned=4,total_training_steps=10000000,pending=pending,active=[dict(j,pid=p.pid) for j,p in active.values()],completed=done,failed=failed)
    while pending or active:
        while pending and len(active)<2:
            j=pending.pop(0)
            if (ROOT/'runs'/j['run_id']).exists():
                failed.append(dict(j,reason='existing run retained; no automatic retry'));continue
            cmd=[sys.executable,'scripts/run.py','--suite','continual','--algorithm','qmix','--method',j['method'],'--scenario','task_switch','--seed',str(j['seed']),'--run-id',j['run_id'],'--protocol',PROTOCOL]
            with (OUT/(j['run_id']+'_launcher.log')).open('w') as stream:p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
            active[j['run_id']]=(j,p)
        write(state('training'))
        time.sleep(15)
        for key,(j,p) in list(active.items()):
            code=p.poll()
            if code is None:continue
            path=ROOT/'runs'/key/'status.json';s=json.loads(path.read_text()) if path.exists() else {}
            if code==0 and s.get('status')=='completed':done.append(j)
            else:failed.append(dict(j,exit_code=code,state=s))
            del active[key]
    for j in done:
        v=state('independent_final_A_evaluation');v['evaluating']=j;write(v)
        with (OUT/(j['run_id']+'_evaluation.log')).open('w') as stream:code=subprocess.call([sys.executable,'scripts/evaluate.py','--run',str(ROOT/'runs'/j['run_id']),'--episodes','32'],stdout=stream,stderr=subprocess.STDOUT)
        if code:eval_failures.append(dict(j,exit_code=code))
    errors=summarize()
    v=state('finished');v.update(status='completed' if not failed and not eval_failures and not errors else 'finished_with_failures',finished=time.time(),evaluation_failures=eval_failures,summary_errors=errors);write(v)
if __name__=='__main__':
    if '--summarize' in sys.argv:summarize()
    else:main()
