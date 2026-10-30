"""Compare only the four requested soft-protection locations, one training seed."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'drivers/continual'))
from neuron_records import export_experiment
OUT=ROOT/'results/location_ablation_seed1111'
COLORS={'A':'#1f77b4','W':'#e1812c','V':'#3b9b71','AWV':'#9c62bc'}

def save_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True);curves=[];stages=[];final=[];summary=[]
    fig,ax=plt.subplots(figsize=(11,5))
    for location in COLORS:
        run=ROOT/'runs'/('qmix_switch_softprotect_'+location+'_seed1111_20261030')
        path=run/'evaluations.jsonl';records=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        for r in records:
            row={k:v for k,v in r.items() if k!='wall_seconds'}
            curves.append(dict(location=location,**row))
            if r['evaluation']=='phase_end':stages.append(dict(location=location,**row))
        # Connect points only within a task stage: switches have both old-task end
        # and new-task start evaluations at the same cumulative training step.
        for phase in range(8):
            group=[r for r in records if r['phase']==phase]
            if group:ax.plot([r['t_env']/1e6 for r in group],[r['win_rate'] for r in group],color=COLORS[location],label=location.replace('AWV','A+W+V') if phase==0 else None,alpha=.85,lw=1.3)
        for f in sorted((run/'evaluation').glob('*/results.json')):
            for r in json.loads(f.read_text()):
                final.append(dict(location=location,task=r['task'],t_env=r['t_env'],wins=r['wins'],episodes=r['episodes'],win_rate=r['win_rate'],evaluation_seed=r['evaluation_seed'],result=str(f)))
        status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        ends=[r['win_rate'] for r in records if r['evaluation']=='phase_end']
        summary.append(dict(location=location,status=status.get('status','not_started'),t_env=status.get('t_env',0),completed_phases=len(ends),mean_phase_end_win_rate=float(np.mean(ends)) if ends else None,run=str(run)))
    for k in range(1,8):ax.axvline(k*.5,color='grey',lw=.5,alpha=.5)
    ax.set(xlabel='Cumulative environment steps (million)',ylabel='Win rate (32 episodes)',ylim=(-.02,1.02),xlim=(0,4),title='5-agent Protoss task-switch | QMIX soft protection | seed 1111')
    ax.legend();fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=180);fig.savefig(OUT/'learning_curves.svg');plt.close(fig)
    save_csv('evaluations.csv',curves);save_csv('phase_end.csv',stages);save_csv('independent_final.csv',final);save_csv('run_summary.csv',summary)
    export_experiment(ROOT)
    lines=['# Soft-protection location ablation','', 'Four independently initialized runs (same seed 1111), 4M steps each. A = shared agent first ReLU output; W/V = mixer hyper_w_final/V ReLU output. All other training settings inherit the task-switch protocol.','', '## Completion','', '| Location | State | Steps | Completed phases | Mean phase-end win rate |','| --- | --- | ---: | ---: | ---: |']
    for r in summary:
        val=r['mean_phase_end_win_rate'];lines.append('| {} | {} | {} | {} | {} |'.format(r['location'],r['status'],r['t_env'],r['completed_phases'],'n/a' if val is None else '{:.2%}'.format(val)))
    lines+=['','## Phase-end comparisons','','| End step | Task | A | W | V | A+W+V |','| ---: | --- | ---: | ---: | ---: | ---: |']
    for phase in range(8):
        rows=[r for r in stages if r['phase']==phase];vals={r['location']:r['win_rate'] for r in rows}
        lines.append('| {} | {} | {} |'.format((phase+1)*500000,'Stalker' if phase%2==0 else 'Zealot',' | '.join('{:.2%}'.format(vals[l]) if l in vals else 'n/a' for l in COLORS)))
    lines+=['','## Independent final evaluation','','| Location | Task | Wins / episodes | Win rate |','| --- | --- | ---: | ---: |']
    for r in final:lines.append('| {} | {} | {} / {} | {:.2%} |'.format(r['location'],r['task'],r['wins'],r['episodes'],r['win_rate']))
    lines+=['','![Learning curves](learning_curves.png)','','This is a single-training-seed comparison. Evaluation episodes and task stages are not independent training replicates; no cross-seed significance or general method ranking is claimed. The number of selected units may differ across layers: the experiment changes eligible protection locations, not a matched parameter budget. Training metrics and 32-episode win rates are from explicit evaluations.jsonl, not the upstream mixed-batch test log. Per-neuron scores, identities, set changes, and actual update audits remain in each run.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
