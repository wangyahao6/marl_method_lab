"""Summarize TS3 actual completion, all stage-end wins and plasticity/retention."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ts3_agent_3535_20261102'
GROUPS=['N','A1_linear','A1_union','A2_linear','A2_union'];PARENT=ROOT/'runs/ts3_Aprefix_seed3535_20261102_attempt001'
def read(p):
    if not p.exists():return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def dump_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True);prefix=read(PARENT/'evaluations.jsonl');rows=[];summaries=[];curves=[];fig,axes=plt.subplots(1,5,figsize=(19,4),sharey=True)
    for ax,group in zip(axes,GROUPS):
        run=ROOT/'runs'/f'ts3_{group}_seed3535_20261102_attempt001';ev=prefix+read(run/'evaluations.jsonl');matrix=np.full((5,5),np.nan)
        for x in ev:
            if x['label']=='stage_end' and x['focus']:
                i=x['phase'];j='ABCDE'.index(x['task']);matrix[i,j]=100*x['win_rate'];rows.append(dict(group=group,trained_task='ABCDE'[i],evaluated_task=x['task'],t_env=x['t_env'],wins=x['wins'],episodes=x['episodes'],win_rate_percent=100*x['win_rate']))
        for x in ev:
            if x['task']==x['trained_task']:curves.append(dict(group=group,**x))
        state=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'status':'not_started'}
        finite=int(np.isfinite(matrix).sum());diag=np.diag(matrix);final=matrix[4]
        bwt=float(np.mean(final[:4]-diag[:4])) if np.isfinite(final[:4]).all() and np.isfinite(diag[:4]).all() else None
        summaries.append(dict(group=group,status=state['status'],t_env=state.get('t_env'),matrix_points=finite,expected_matrix_points=15,final_old_mean_percent=float(np.mean(final[:4])) if np.isfinite(final[:4]).all() else None,BWT_E_pp=bwt,final_all_mean_percent=float(np.mean(final)) if np.isfinite(final).all() else None))
        im=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis');ax.set_title(group);ax.set_xticks(range(5));ax.set_xticklabels(list('ABCDE'));ax.set_yticks(range(5));ax.set_yticklabels(list('ABCDE'));ax.set_xlabel('evaluated task')
        for i in range(5):
            for j in range(i+1):
                if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]<50 else 'black',fontsize=8)
    axes[0].set_ylabel('completed training stage');fig.colorbar(im,ax=axes,shrink=.8,label='win rate (%)');fig.savefig(OUT/'retention_matrices.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'retention_matrices.svg',bbox_inches='tight');plt.close(fig)
    dump_csv(OUT/'stage_end_matrix.csv',rows);dump_csv(OUT/'summary.csv',summaries)
    fields=['group','t_env','phase','trained_task','task','label','focus','wins','episodes','win_rate']
    dump_csv(OUT/'current_task_curves.csv',[{k:x[k] for k in fields} for x in curves])
    fig,axes=plt.subplots(1,5,figsize=(20,4),sharey=True)
    for ax,task in zip(axes,'ABCDE'):
        for g in GROUPS:
            series=[x for x in curves if x['group']==g and x['task']==task]
            ax.plot([(x['t_env']-'ABCDE'.index(task)*1000000)/1e6 for x in series],[100*x['win_rate'] for x in series],label=g)
        ax.set_title(task);ax.set_xlabel('stage training steps (M)');ax.set_ylim(0,100)
    axes[0].set_ylabel('current task win rate (%)');axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'current_task_learning.png',dpi=180);fig.savefig(OUT/'current_task_learning.svg');plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2))
    lines=['# TS3 single-layer protection comparison','', 'Seed 3535 only. Five paths share the A prefix. Evaluation episodes and tasks are not independent training replicates.','', '| Group | Status | Steps | Matrix points / 15 | Final old mean % | BWT pp | Final all mean % |','|---|---|---:|---:|---:|---:|---:|']
    for x in summaries:lines.append('| '+' | '.join(str(x[k]) for k in ['group','status','t_env','matrix_points','final_old_mean_percent','BWT_E_pp','final_all_mean_percent'])+' |')
    lines+=['','Read absolute first-learned wins and current-task curves alongside retention. Different coverage and protection duration are part of the compared schedules; results do not isolate ramp shape or prove screening causality. Raw runs remain in /home/wyh/marl_method_lab/runs. Incomplete paths are not complete results.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
