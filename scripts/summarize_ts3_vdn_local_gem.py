"""Per-seed stage matrix/ACC/BWT/FWT and method-baseline contrasts; no fake completion."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ts3_vdn_A2_local_gem_20261104'
TASKS='ABCDE'
def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
def csvout(name,rows):
    if rows:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, default=OUT)
    OUT = ROOT / parser.parse_args().result_dir
    plan=json.loads((OUT/'plan.json').read_text());protocol=json.loads((ROOT/plan['items'][0]['protocol']).read_text())
    branches = {(str(item['seed']), item['band']): item['run']
                for item in plan['items'] if item['role'] == 'branch'}
    bands = list(dict.fromkeys(band for _, band in branches))
    groups = ['N', *bands]
    summaries=[];matrix_rows=[];curves=[];diags=[];fig,axes=plt.subplots(
        len(protocol['sources']),len(groups),figsize=(4*len(groups),12),sharex=True,sharey=True,squeeze=False)
    for row,(seed,source) in enumerate(protocol['sources'].items()):
        prefix=ROOT/'runs'/source['prefix'];baseline=ROOT/'runs'/source['baseline']
        aux=ROOT/'runs'/f'ts3_vdn_localgem_baseline_fwt_seed{seed}_20261104_attempt001'
        entries=[('N',baseline,read(aux/'evaluations.jsonl'))]
        for band in bands:
            entries.append((band, ROOT/'runs'/branches[seed,band], []))
        for col,(group,run,auxev) in enumerate(entries):
            ev=read(prefix/'evaluations.jsonl')+read(run/'evaluations.jsonl')+auxev
            matrix=np.full((5,5),np.nan);init={};future={}
            for x in ev:
                if x['label']=='stage_end' and x['focus']:
                    i=x['phase'];j=TASKS.index(x['task']);matrix[i,j]=100*x['win_rate']
                if x['label']=='initial_reference':init[x['task']]=100*x['win_rate']
                if x['label']=='next_unseen':future[x['task']]=100*x['win_rate']
                if x['label'] in ['phase_start','periodic','stage_end'] and x['task']==x['trained_task']:curves.append(dict(seed=int(seed),group=group,t_env=x['t_env'],task=x['task'],win_rate_percent=100*x['win_rate'],label=x['label']))
            for i in range(5):
                seen=matrix[i,:i+1];old=matrix[i,:i]-np.diag(matrix)[:i]
                acc=float(seen.mean()) if np.isfinite(seen).all() else None
                bwt=float(old.mean()) if i>0 and np.isfinite(old).all() else None
                for j in range(i+1):matrix_rows.append(dict(seed=int(seed),group=group,stage=TASKS[i],task=TASKS[j],win_rate_percent=float(matrix[i,j]) if np.isfinite(matrix[i,j]) else None,ACC_percent=acc,BWT_pp=bwt))
            fwt=float(np.mean([future[t]-init[t] for t in TASKS[1:]])) if all(t in init and t in future for t in TASKS[1:]) else None
            state=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else dict(status='queued')
            final=matrix[4];diagonal=np.diag(matrix)
            summaries.append(dict(seed=int(seed),group=group,run=run.name,status=state.get('status'),reference_errors=len(read(run/'gem_errors.jsonl')),t_env=state.get('t_env'),matrix_points=int(np.isfinite(matrix).sum()),ACC_percent=float(final.mean()) if np.isfinite(final).all() else None,old_mean_percent=float(final[:4].mean()) if np.isfinite(final[:4]).all() else None,BWT_pp=float((final[:4]-diagonal[:4]).mean()) if np.isfinite(final[:4]).all() and np.isfinite(diagonal[:4]).all() else None,FWT_pp=fwt,FWT_reference='same-seed untrained reference reconstructed by current initializer; not archived original initialization',FWT_missing=None if fwt is not None else 'missing initial-reference or next-unseen evaluations'))
            for x in read(run/'gem_audit.jsonl'):diags.append(dict(seed=int(seed),group=group,**x))
            ax=axes[row,col];ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis');ax.set_title(f'{seed} / {group}');ax.set_xticks(range(5));ax.set_xticklabels(TASKS);ax.set_yticks(range(5));ax.set_yticklabels(TASKS)
            for i in range(5):
                for j in range(i+1):
                    if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]<50 else 'black')
    fig.tight_layout();fig.savefig(OUT/'retention_matrices.png',dpi=170);plt.close(fig)
    csvout('summary.csv',summaries);csvout('stage_end_matrix.csv',matrix_rows);csvout('current_task_curves.csv',curves)
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2));(OUT/'gem_diagnostics.json').write_text(json.dumps(diags,indent=2))
    contrast=[]
    for seed in protocol['seeds']:
        base=next(x for x in summaries if x['seed']==seed and x['group']=='N')
        for method in [x for x in summaries if x['seed']==seed and x['group']!='N']:
            contrast.append(dict(seed=seed,group=method['group'],**{k:method[k]-base[k] if method[k] is not None and base[k] is not None else None for k in ['ACC_percent','old_mean_percent','BWT_pp','FWT_pp']}))
    csvout('paired_contrasts.csv',contrast)
    auc=[]
    for seed in protocol['seeds']:
        for group in groups:
            for task in TASKS:
                points={x['t_env']:x['win_rate_percent'] for x in curves if x['seed']==seed and x['group']==group and x['task']==task}
                times=sorted(points)
                if len(times)>1:auc.append(dict(seed=seed,group=group,task=task,first_step=times[0],last_step=times[-1],normalized_learning_AUC_percent=float(np.trapz([points[t] for t in times],times)/(times[-1]-times[0]))))
    csvout('learning_auc.csv',auc)
    fig,axes=plt.subplots(3,5,figsize=(20,10),sharey=True)
    for row,seed in enumerate(protocol['seeds']):
        for col,task in enumerate(TASKS):
            ax=axes[row,col]
            for group in groups:
                xs=sorted([x for x in curves if x['seed']==seed and x['task']==task and x['group']==group],key=lambda x:x['t_env'])
                if xs:ax.plot([(x['t_env']-col*2000000)/1e6 for x in xs],[x['win_rate_percent'] for x in xs],label=group)
            ax.set_title(f'{seed} / {task}');ax.set_ylim(0,100)
    axes[0,-1].legend();fig.tight_layout();fig.savefig(OUT/'current_task_learning.png',dpi=170);plt.close(fig)
    lines=['# TS3 VDN A2 local GEM: equal high/middle/low bands','', 'Existing VDN baseline and A prefixes reused. No baseline or A-prefix training added. New methods: local actual RMSprop increment projection, not full-network GEM and not 0.1 soft protection. Each band21units, ranks1-21/22-42/44-64; rank43 excluded. Latest boundary grouping, no neuronal history union.','', '| Seed | Group | State | Matrix/15 | ACC % | Old mean % | BWT pp | FWT pp |','|---|---|---|---:|---:|---:|---:|---:|']
    if protocol.get('gem_score_threshold') is not None:
        lines[:3] = [
            '# TS3 VDN A2 local GEM: overall score strictly above 3', '',
            'Each seed reuses its original VDN baseline, A prefix, and initial FWT reference. '
            'At each boundary the selected A2 slice is replaced by the units with score > 3. '
            'An empty or undefined selection gives ordinary VDN updates for that stage.',
        ]
    for x in summaries:lines.append('| '+' | '.join(str(x[k]) for k in ['seed','group','status','matrix_points','ACC_percent','old_mean_percent','BWT_pp','FWT_pp'])+' |')
    lines+=['','FWT uses an explicit same-seed untrained reference reconstructed by the current initializer, not an archived historical initializer. Baseline next-unseen evaluation uses archived checkpoints; no learning. Missing values are N/A. Auxiliary numerical residuals/errors do not establish victory retention or terminate the queue. Incomplete runs are not completion evidence. Three seeds only; tasks and episodes are not training replicates.','', 'See paired_contrasts.csv, gem_diagnostics.json and original runs. Independently review training terminal status, exit codes and evaluation completeness before claiming completion.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
