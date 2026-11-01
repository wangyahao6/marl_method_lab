import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/2s3z_type_ablation_winrate_20261101_attempt002'
def main():
 plan=json.loads((OUT/'plan.json').read_text());results={};errors=[]
 for task in plan['tasks']:
  d=OUT/'tasks'/task['task_id']
  if (d/'result.json').exists():results[task['task_id']]=json.loads((d/'result.json').read_text())
  if (d/'error.json').exists():errors.append(task['task_id'])
 baseline={v['task']['seed']:v for v in results.values() if not v['task']['ids']};rows=[]
 for m in plan['mapping']:
  if m['task_id'] not in results:continue
  r=results[m['task_id']];b=baseline.get(m['seed']);rows.append(dict(**m,episodes=r['episodes'],wins=r['wins'],win_rate=r['win_rate'],mean_return=r['mean_return'],baseline_win_rate=None if b is None else b['win_rate'],win_rate_drop=None if b is None else b['win_rate']-r['win_rate']))
 if rows:
  with (OUT/'winrates.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 groups={}
 for r in rows:
  key=(r['seed'],r['layer'],r['source'],r['recipient'],r['group']);groups.setdefault(key,[]).append(r)
 summary=[]
 for key,vs in groups.items():
  values=[r['win_rate'] for r in vs];summary.append(dict(zip(['seed','layer','source','recipient','group'],key),k=vs[0]['k'],n_masked=vs[0]['n_masked'],repeats=len(vs),mean_win_rate=float(np.mean(values)),win_rate_q05=float(np.quantile(values,.05)),win_rate_q95=float(np.quantile(values,.95)),mean_return=float(np.mean([r['mean_return'] for r in vs])),baseline_win_rate=vs[0]['baseline_win_rate']))
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 if summary:
  fig,axes=plt.subplots(2,2,figsize=(13,8));names=['unmasked','type_high','overall_high','random','type_low','overall_low']
  for i,l in enumerate(['A1','A2']):
   for j,s in enumerate(['stalker','zealot']):
    ax=axes[i,j]
    for z,g in enumerate(names):
     y=[]
     for seed in [1111,2222,3535]:
      r=next((x for x in summary if x['seed']==seed and x['layer']==l and x['source']==s and x['recipient']==s and x['group']==g),None);y.append(r['mean_win_rate']*100 if r else np.nan)
     ax.bar(np.arange(3)+(z-2.5)*.13,y,.13,label=g)
    ax.set_xticks(range(3));ax.set_xticklabels(['1111','2222','3535']);ax.set_ylim(0,105);ax.set_title(l+' source = target '+s);ax.set_ylabel('Real rollout win rate (%)')
  h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=6);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'winrates.png',dpi=160);plt.close(fig)
 review=dict(expected_tasks=len(plan['tasks']),completed_tasks=len(results),failed_tasks=len(errors),failed_ids=errors,completed_episodes=sum(r['episodes'] for r in results.values()),parameters_unchanged=all(r['parameters_unchanged'] for r in results.values()),complete=len(results)==len(plan['tasks']) and not errors and all(r['episodes']==32 and r['parameters_unchanged'] for r in results.values()))
 (OUT/'review.json').write_text(json.dumps(review,indent=2))
 (OUT/'REPORT.md').write_text('# Real-rollout matched type ablation\n\n'+json.dumps(review,indent=2)+'\n\nSee winrates.csv, summary.json and winrates.png. Each condition uses four common environment-seed blocks of eight greedy real episodes (32 total). Random mask sets have separate 32-episode evaluations; their distribution is not extra training seeds. Identical interventions are deduplicated and baseline reused per model. Common initialization seeds do not guarantee identical stochastic events after policies diverge. This frozen-model test measures team win-rate effects, not soft-protection training efficacy. Incomplete or failed tasks must remain explicit.\n')
 print(json.dumps(review))
if __name__=='__main__':main()
