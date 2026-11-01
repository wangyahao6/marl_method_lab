import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/2s3z_type_matched_ablation_20261101'
def main():
 cases=sum([json.loads(p.read_text()) for p in sorted(OUT.glob('seed*.json'))],[]);summary=[]
 for case in cases:
  if case['skipped']:continue
  for recipient in ['stalker','zealot']:
   cs=[x for x in case['conditions'] if x['recipient']==recipient]
   for group in sorted(set(x['group'] for x in cs)):
    selected=[x for x in cs if x['group']==group]
    assert all(x['n_masked']==case['k'] for x in selected if group not in ['unmasked','type_unique','overall_unique'])
    if group in ['type_unique','overall_unique']:assert all(x['n_masked']==len(case['type_unique']) for x in selected)
    record=dict(seed=case['seed'],layer=case['layer'],source=case['source'],recipient=recipient,group=group,k=case['k'],repeats=len(selected),n_masked=selected[0]['n_masked'])
    for metric in ['flip_rate','abs_margin_change','margin_erosion','legal_q_mae']:
     vals=[x['pooled'][metric] for x in selected]
     record[metric]=float(np.mean(vals));record[metric+'_q05']=float(np.quantile(vals,.05));record[metric+'_q95']=float(np.quantile(vals,.95))
    for metric in ['zero_fraction_selection','zero_fraction_evaluation']:record[metric]=selected[0][metric]
    summary.append(record)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 with (OUT/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 names=['type_high','overall_high','random','type_low','overall_low'];colors=['#D55E00','#0072B2','#777777','#009E73','#56B4E9']
 for metric,label in [('flip_rate','Legal greedy action change (%)'),('abs_margin_change','Mean absolute original-action margin change')]:
  fig,axes=plt.subplots(2,2,figsize=(13,8))
  for i,layer in enumerate(['A1','A2']):
   for j,source in enumerate(['stalker','zealot']):
    ax=axes[i,j]
    for gidx,(name,color) in enumerate(zip(names,colors)):
     values=[];lower=[];upper=[]
     for seed in [1111,2222,3535]:
      r=next((r for r in summary if r['seed']==seed and r['layer']==layer and r['source']==source and r['recipient']==source and r['group']==name),None)
      v=r[metric] if r else np.nan;scale=100 if metric=='flip_rate' else 1
      values.append(v*scale);lower.append((v-r[metric+'_q05'])*scale if r else 0);upper.append((r[metric+'_q95']-v)*scale if r else 0)
     # Quantiles are repeat distributions, not training-seed confidence intervals.
     ax.bar(np.arange(3)+(gidx-2)*.15,values,.15,label=name,color=color,yerr=[lower,upper] if name=='random' else None,capsize=2)
    ax.set_xticks(range(3));ax.set_xticklabels(['1111','2222','3535']);ax.set_title(f'{layer}: source = recipient = {source}');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
  handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=9);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/(metric+'.png'),dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(12,7))
 for i,layer in enumerate(['A1','A2']):
  for j,seed in enumerate([1111,2222,3535]):
   matrix=np.full((2,2),np.nan)
   for sidx,source in enumerate(['stalker','zealot']):
    for ridx,recipient in enumerate(['stalker','zealot']):
     rs=[r for r in summary if r['seed']==seed and r['layer']==layer and r['source']==source and r['recipient']==recipient]
     if rs:matrix[sidx,ridx]=100*(next(r for r in rs if r['group']=='type_high')['flip_rate']-next(r for r in rs if r['group']=='overall_high')['flip_rate'])
   ax=axes[i,j];im=ax.imshow(matrix,cmap='RdBu_r',vmin=-15,vmax=15)
   for u in range(2):
    for v in range(2):ax.text(v,u,'N/A' if np.isnan(matrix[u,v]) else f'{matrix[u,v]:+.1f}',ha='center',va='center')
   ax.set_xticks([0,1]);ax.set_xticklabels(['target Stalker','target Zealot']);ax.set_yticks([0,1]);ax.set_yticklabels(['source Stalker','source Zealot']);ax.set_title(f'{layer}, seed {seed}')
 fig.subplots_adjust(left=.12,wspace=.7,hspace=.5,right=.84);cax=fig.add_axes([.9,.15,.018,.7]);fig.colorbar(im,cax=cax,label='Type high minus overall high: action-change percentage points');fig.savefig(OUT/'cross_type_difference.png',dpi=160);plt.close(fig)
 for r in summary:
  if r['source']==r['recipient'] and r['group'] in ['type_low','overall_low']:
   print(r['seed'],r['layer'],r['source'],r['group'],'zero fraction',r['zero_fraction_selection'],r['zero_fraction_evaluation'])
 print('checks passed: all primary nonzero mask sizes matched; unique comparisons matched; random repeats not training seeds')
if __name__=='__main__':main()
