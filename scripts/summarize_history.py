"""Summarize actual endpoints and within-seed neuron comparisons; safe for partial runs."""
import csv,itertools,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/vdn_task_history_20261103'
def csvwrite(name,rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    queue=json.loads((OUT/'queue_status.json').read_text());records=[];scores=[];endpoints=[];errors=[]
    for run,node in queue['runs'].items():
        rd=ROOT/'runs'/run
        status=json.loads((rd/'status.json').read_text()) if (rd/'status.json').exists() else {}
        endpoints.append(dict(run=run,seed=node['seed'],path=node['path'],queue_state=node['state'],training_status=status.get('status','missing'),t_env=status.get('t_env',0)))
        for error in rd.glob('detections/*/error.json'):errors.append(dict(run=run,file=str(error),detail=json.loads(error.read_text())))
        for f in rd.glob('detections/*boundary*/result.json'):
            result=json.loads(f.read_text())
            for layer,item in result.items():
                vector=item.get('scores')
                if vector is None:errors.append(dict(run=run,layer=layer,detail='undefined scores'));continue
                record=dict(run=run,seed=node['seed'],path=node['path'],task=node['path'][-1],layer=layer,states=item['states'],count=item['count'],scores=vector,selected=item['sets']['overall']);records.append(record)
                for i,value in enumerate(vector):scores.append(dict(run=run,seed=node['seed'],path=node['path'],task=node['path'][-1],layer=layer,neuron=i,score=value,selected=i in record['selected']))
    comparisons=[]
    for a,b in itertools.combinations(records,2):
        if (a['seed'],a['layer'])!=(b['seed'],b['layer']):continue
        same=a['task']==b['task']
        if not same and not(len(a['path'])==len(b['path'])==1):continue
        x=np.array(a['scores']);y=np.array(b['scores']);sx=set(a['selected']);sy=set(b['selected']);u=sx|sy
        tx=set(np.argsort(-x,kind='stable')[:3]);ty=set(np.argsort(-y,kind='stable')[:3])
        comparisons.append(dict(seed=a['seed'],layer=a['layer'],path_a=a['path'],path_b=b['path'],kind='same_task_history' if same else 'scratch_different_tasks',spearman=float(spearmanr(x,y)[0]) if np.ptp(x)>0 and np.ptp(y)>0 else None,wasserstein=float(wasserstein_distance(x,y)),mean_absolute_indexed_change=float(np.abs(x-y).mean()),selected_a=len(sx),selected_b=len(sy),threshold_jaccard=len(sx&sy)/len(u) if u else None,top3_overlap=len(tx&ty)/3,top3_jaccard=len(tx&ty)/len(tx|ty)))
    csvwrite('endpoints.csv',endpoints);csvwrite('scores.csv',scores);csvwrite('comparisons.csv',comparisons)
    (OUT/'diagnostic_errors.json').write_text(json.dumps(errors,indent=2))
    if records:
        fig,axes=plt.subplots(3,2,figsize=(15,12))
        for row,seed in enumerate([3535,2222,1111]):
            for col,layer in enumerate(['A1','A2']):
                ax=axes[row,col]
                subset=sorted([r for r in records if r['seed']==seed and r['layer']==layer],key=lambda r:(len(r['path']),r['path']))
                if subset:
                    im=ax.imshow([r['scores'] for r in subset],aspect='auto',vmin=0,vmax=max(3,max(max(r['scores']) for r in subset)));ax.set_yticks(range(len(subset)));ax.set_yticklabels([r['path'] for r in subset]);fig.colorbar(im,ax=ax,label='Score')
                ax.set_title(f'Seed {seed} / {layer}');ax.set_xlabel('Neuron index')
        fig.tight_layout();fig.savefig(OUT/'score_heatmaps.png',dpi=170);plt.close(fig)
    completed=sum(n['state']=='completed' for n in queue['runs'].values())
    lines=['# VDN task/history analysis',f'Completed training nodes: {completed}/45. Diagnostic errors/undefined results: {len(errors)}.',f'Available layer score vectors: {len(records)}/90. See endpoints.csv for actual steps; scores.csv and comparisons.csv for neuron data.','', 'S/Z/C: mirror 5v5; three seeds3535/2222/1111; each stage nominal2M at cumulative episode boundaries. Shared prefixes are reused observations, not independent repetitions. No protection/reset; replay, optimizer, targets and exploration retained.','', 'Endpoint score uses last32 current-task training episodes and the unchanged ts4_diagnostics.score. Score>3 and top3 (ties by ascending index) are descriptive. Wasserstein compares unordered score distributions; Spearman, indexed changes and overlaps compare labeled units within the same training seed. Empty threshold unions have undefined Jaccard. No cross-seed neuron index pairing.','', 'Comparisons cover scratch cross-task distributions and all same-final-task histories (scratch, each predecessor, and both predecessor orders). Same-stage histories provide equal-total-budget order contrasts; scratch vs transfer matches current-task exposure but differs in total steps, replay and exploration. On-policy score differences also include policy/data-distribution changes. They do not establish causal importance or protection benefit.','', 'Stage-end win rates are in each run/evaluations.jsonl (128 episodes per seen task); periodic current-task curves use32 episodes/25k steps. Raw models,32-episode score inputs and prefix continuation states stay remote. Queue completion is pending review, not an automatic scientific conclusion.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
