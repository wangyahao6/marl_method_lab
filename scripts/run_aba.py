"""Authorized QMIX versus W+V soft protection, ABA, seed 1111."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/aba_WV_seed1111_20261031'
RUNS={m:'qmix_ABA_'+m+'_seed1111_20261031' for m in ['baseline','softprotect']}

def write(value):
    dest=OUT/'status.json';tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(dest)

def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];states=[];final=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    colors={'baseline':'#2878b5','softprotect':'#df7833'}
    lines=['# ABA: QMIX vs QMIX + W/V soft protection','',
        'Seed 1111; 5v5 homogeneous mirror teams. Stalker 1M → Zealot 500k → Stalker 500k. Each method independently initializes and trains 2M environment steps. Only the B phase scales selected W/V post-RMSprop increments by 0.1; returning A immediately restores all scales to 1. Screening uses only the last 32 training episodes of the first A phase, threshold strictly >3. Other settings inherit the existing task-switch driver.','',
        'Replay, target networks, optimizer state and cumulative exploration persist across transitions. Evaluation: 32 greedy episodes before training each phase and about every 10k steps, including exact phase endpoints. The return-A initial value is phase=2, t_env=1500000, evaluation=phase_start, before any A2 update. Phase boundaries truncate exactly; periodic evaluation follows whole episodes. Logical 5gen_protoss uses the inherited 10gen_protoss registration with n_units=n_enemies=5.','',
        '| Method | Training state | Steps | A1 end | Return A initial | A2 end |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    def rate(records,phase,label):
        selected=[r for r in records if r['phase']==phase and r['evaluation']==label]
        return '{:.2%}'.format(selected[-1]['win_rate']) if selected else 'pending'
    for method,runid in RUNS.items():
        run=ROOT/'runs'/runid
        path=run/'evaluations.jsonl'
        records=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        states.append(dict(method=method,run_id=runid,status=status.get('status','not_started'),t_env=status.get('t_env',0)))
        for r in records: rows.append(dict(method=method,**{k:v for k,v in r.items() if k!='wall_seconds'},return_A_steps=r['t_env']-1500000 if r['phase']==2 else ''))
        for phase in range(3):
            group=[r for r in records if r['phase']==phase]
            if group: axes[0].plot([r['t_env']/1e6 for r in group],[r['win_rate'] for r in group],color=colors[method],label=method if phase==0 else None)
        group=[r for r in records if r['phase']==2]
        if group:axes[1].plot([(r['t_env']-1500000)/1000 for r in group],[r['win_rate'] for r in group],color=colors[method],label=method)
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(method,status.get('status','not_started'),status.get('t_env',0),rate(records,0,'phase_end'),rate(records,2,'phase_start'),rate(records,2,'phase_end')))
        for p in sorted((run/'evaluation').glob('*/results.json')):
            for r in json.loads(p.read_text()):final.append(dict(method=method,task=r['task'],t_env=r['t_env'],wins=r['wins'],episodes=r['episodes'],win_rate=r['win_rate'],evaluation_seed=r['evaluation_seed'],result=str(p)))
    for x in [1,1.5]:axes[0].axvline(x,color='gray',ls='--',lw=.8)
    axes[0].set(title='ABA learning curves',xlabel='Total environment steps (million)',xlim=(0,2))
    axes[1].set(title='Return to A: initial retention and recovery',xlabel='A2 environment steps (thousand)',xlim=(0,500))
    for ax in axes:
        ax.set(ylabel='Win rate (32 episodes)',ylim=(-.02,1.02));ax.grid(alpha=.2)
        if ax.lines:ax.legend()
    fig.tight_layout();fig.savefig(OUT/'learning_and_recovery.png',dpi=180);fig.savefig(OUT/'learning_and_recovery.svg');plt.close(fig)
    for name,data in [('evaluations.csv',rows),('return_A_recovery.csv',[r for r in rows if r['phase']==2]),('run_summary.csv',states),('independent_final.csv',final)]:
        if data:
            with (OUT/name).open('w',newline='') as stream:
                w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines+=['','## Independent final A evaluation','','| Method | Wins / episodes | Win rate |','| --- | ---: | ---: |']
    for r in final:lines.append('| {} | {} / {} | {:.2%} |'.format(r['method'],r['wins'],r['episodes'],r['win_rate']))
    lines+=['','![Learning and recovery](learning_and_recovery.png)','','A single training seed is a case comparison; evaluation episodes are not independent training replicates. No across-seed significance claim. Incomplete runs provide partial evidence only. Raw artifacts: '+str(ROOT/'runs')+'. Configuration: configs/continual/aba_WV_switch_1111.json.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',SDL_VIDEODRIVER='dummy',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
    active={};done=[];failed=[];eval_failures=[];started=time.time()
    def state(phase):
        return dict(status='running',phase=phase,pid=os.getpid(),started=started,planned=2,total_training_steps=4000000,completed=done,failed=failed,active=[dict(method=m,run_id=RUNS[m],pid=p.pid) for m,p in active.items()])
    for method,runid in RUNS.items():
        if (ROOT/'runs'/runid).exists():
            failed.append(dict(method=method,reason='run directory exists; retained without retry'));continue
        cmd=[sys.executable,'scripts/run.py','--suite','continual','--algorithm','qmix','--method',method,'--scenario','task_switch','--seed','1111','--run-id',runid,'--protocol','configs/continual/aba_WV_switch_1111.json']
        with (OUT/(method+'_launcher.log')).open('w') as stream:active[method]=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
    write(state('training'))
    while active:
        time.sleep(15)
        for method,proc in list(active.items()):
            code=proc.poll()
            if code is None:continue
            p=ROOT/'runs'/RUNS[method]/'status.json'
            s=json.loads(p.read_text()) if p.exists() else {}
            if code==0 and s.get('status')=='completed':done.append(method)
            else:failed.append(dict(method=method,exit_code=code,state=s))
            del active[method]
        write(state('training'))
    for method in done:
        v=state('independent_final_A_evaluation');v['evaluating']=method;write(v)
        with (OUT/(method+'_evaluation.log')).open('w') as stream:
            code=subprocess.call([sys.executable,'scripts/evaluate.py','--run',str(ROOT/'runs'/RUNS[method]),'--episodes','32'],stdout=stream,stderr=subprocess.STDOUT)
        if code:eval_failures.append(dict(method=method,exit_code=code))
    try:
        summarize();summary_error=None
    except Exception as e:
        import traceback
        summary_error=traceback.format_exc();(OUT/'summary_error.log').write_text(summary_error)
    v=state('finished');v.update(status='completed' if not failed and not eval_failures and not summary_error else 'finished_with_failures',finished=time.time(),evaluation_failures=eval_failures,summary_error=summary_error);write(v)
if __name__=='__main__':
    if '--summarize' in sys.argv:summarize()
    else:main()
