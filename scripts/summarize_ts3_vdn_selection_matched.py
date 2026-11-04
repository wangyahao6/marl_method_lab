"""Nine paths: three existing VDN baselines and both screening rules per seed."""
import csv,json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ts3_vdn_selection_matched_20261104'
def read(path):
    if not path.exists():return []
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def dump(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    plan=json.loads((OUT/'plan.json').read_text());items={i['seed']:i for i in plan['items']}
    summaries=[];matrix_rows=[];curves=[];audits=[]
    fig,axes=plt.subplots(3,3,figsize=(12,11))
    for row,seed in enumerate([3535,1234,4567]):
        item=items[seed];parent=ROOT/'runs'/item['parent']
        old=f'ts3_vdn_top3_A2_union_seed{seed}_20261102_attempt001'
        base=f'ts3_vdn_top3_N_seed{seed}_20261102_attempt001'
        if seed==3535:
            old='ts3_vdn_overallunion_A2_union_seed3535_20261103_attempt001'
            base='ts3_vdn_typeunion_N_seed3535_20261102_attempt001'
        specs=[('baseline',base),('top3',item['run'] if seed==3535 else old),('threshold',old if seed==3535 else item['run'])]
        for col,(method,run_id) in enumerate(specs):
            run=ROOT/'runs'/run_id;group=f'{seed}_{method}';ev=read(parent/'evaluations.jsonl')+read(run/'evaluations.jsonl');matrix=np.full((5,5),np.nan)
            for x in ev:
                if x['focus'] and x['label']=='stage_end':
                    i=x['phase'];j='ABCDE'.index(x['task']);matrix[i,j]=100*x['win_rate']
                    matrix_rows.append(dict(group=group,run_id=run_id,trained_task='ABCDE'[i],evaluated_task=x['task'],t_env=x['t_env'],wins=x['wins'],episodes=x['episodes'],win_rate_percent=100*x['win_rate']))
                if x['task']==x['trained_task']:curves.append(dict(group=group,**x))
            state=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'status':'not_started'}
            complete=state['status']=='completed' and int(np.isfinite(matrix).sum())==15
            summaries.append(dict(seed=seed,method=method,run_id=run_id,status=state['status'],t_env=state.get('t_env'),matrix_points=int(np.isfinite(matrix).sum()),final_old_mean_percent=float(matrix[4,:4].mean()) if complete else None,BWT_E_pp=float((matrix[4,:4]-np.diag(matrix)[:4]).mean()) if complete else None,final_all_mean_percent=float(matrix[4].mean()) if complete else None))
            if run_id==item['run']:
                raw=Counter();wins=Counter()
                key=lambda x:(x['t_env'],x['phase'],x['task'],x['label'],x['focus'])
                for x in read(run/'test_episodes.jsonl'):raw[key(x)]+=1;wins[key(x)]+=int(x['won'])
                mismatches=[dict(key=list(key(x)),episodes=x['episodes'],raw=raw[key(x)],wins=x['wins'],raw_wins=wins[key(x)]) for x in read(run/'evaluations.jsonl') if raw[key(x)]!=x['episodes'] or wins[key(x)]!=x['wins']]
                audits.append(dict(seed=seed,run_id=run_id,status=state,restore=json.loads((run/'branch_restore.json').read_text()) if (run/'branch_restore.json').exists() else None,evaluation_mismatches=mismatches,detection_errors=[str(p.relative_to(ROOT)) for p in (run/'detections').glob('*/error.json')],matrix_complete=complete))
            ax=axes[row,col];im=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis');ax.set_title(group);ax.set_xticks(range(5));ax.set_xticklabels(list('ABCDE'));ax.set_yticks(range(5));ax.set_yticklabels(list('ABCDE'))
            for i in range(5):
                for j in range(i+1):
                    if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]<50 else 'black')
    fig.tight_layout();fig.savefig(OUT/'retention_matrices.png',dpi=170);plt.close(fig)
    dump(OUT/'summary.csv',summaries);dump(OUT/'stage_end_matrix.csv',matrix_rows)
    dump(OUT/'current_task_curves.csv',[{k:x[k] for k in ['group','t_env','phase','trained_task','task','label','focus','wins','episodes','win_rate']} for x in curves])
    fig,axes=plt.subplots(3,5,figsize=(18,10),sharey=True)
    for row,seed in enumerate([3535,1234,4567]):
        for col,task in enumerate('ABCDE'):
            ax=axes[row,col]
            for method in ['baseline','top3','threshold']:
                series=[x for x in curves if x['group']==f'{seed}_{method}' and x['task']==task]
                ax.plot([(x['t_env']-col*2000000)/1e6 for x in series],[x['win_rate']*100 for x in series],label=method)
            ax.set_title(f'{seed} task {task}');ax.set_ylim(0,100);ax.set_xlabel('stage steps (M)')
        axes[row,0].set_ylabel('current task win rate (%)');axes[row,-1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'current_task_learning.png',dpi=170);plt.close(fig)
    contrasts=[]
    keys=['final_old_mean_percent','BWT_E_pp','final_all_mean_percent']
    for seed in items:
        entries={x['method']:x for x in summaries if x['seed']==seed}
        for a,b in [('top3','baseline'),('threshold','baseline'),('top3','threshold')]:
            contrasts.append(dict(seed=seed,comparison=a+' minus '+b,**{k:None if entries[a][k] is None or entries[b][k] is None else entries[a][k]-entries[b][k] for k in keys}))
    (OUT/'paired_contrasts.json').write_text(json.dumps(contrasts,indent=2));(OUT/'completion_review.json').write_text(json.dumps(audits,indent=2))
    lines=['# TS3 VDN A2: matched top-3 versus strict score >3','', 'Overall state-weighted screening, history union, seeds3535/1234/4567. Same seed-specific completed A prefix; baselines and existing opposite-rule paths reused. Three new B-E branches only, nominal24M new training steps.','', '| Seed | Method | Status | Matrix points | Final old % | BWT pp | Final all % |','|---|---|---|---:|---:|---:|---:|']
    for x in summaries:lines.append('| '+' | '.join(str(x[k]) for k in ['seed','method','status','matrix_points',*keys])+' |')
    lines+=['','Final metrics are withheld for incomplete paths. Screening rules may select different numbers and histories; this compares complete policies, not equal protection budgets. Report final old-task wins, initial task acquisition, all-task wins and BWT together. Tasks/evaluation episodes are not training seeds.','', '## Three-seed descriptive means','']
    for method in ['baseline','top3','threshold']:
        rows=[x for x in summaries if x['method']==method]
        if all(x[keys[0]] is not None for x in rows):lines.append(method+': '+str({k:float(np.mean([x[k] for x in rows])) for k in keys}))
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return 0 if all(x['matrix_complete'] and not x['evaluation_mismatches'] and x['status']['status']=='completed' for x in audits) else 1
if __name__=='__main__':raise SystemExit(main())
