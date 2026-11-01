import json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/2s3z_pure_detection_20261101'
SEEDS=[1111,2222,3535]
def main():
 OUT.mkdir(exist_ok=True,parents=True);rows=[];fig,ax=plt.subplots()
 for seed in SEEDS:
  run=ROOT/'runs'/f'qmix_2s3z_detect_seed{seed}_20261101_attempt003'
  files=list((run/'sacred').glob('*/info.json'))
  if files:
   info=json.loads(files[0].read_text());y=info.get('test_battle_won_mean',[]);x=info.get('test_battle_won_mean_T',[])
   if len(x)==len(y):ax.plot(x,y,label=str(seed))
  path=run/'detection/records.jsonl'
  records=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
  f,axes=plt.subplots(3,1,figsize=(10,10),sharex=True)
  for axis,k in zip(axes,['A1','A2','H']):
   groups=list(records[0]['layers'][k]) if records and records[0]['layers'] else []
   for g in groups:
    counts=[len(r['layers'][k][g]['selected']) if r['layers'][k][g]['selected'] is not None else np.nan for r in records]
    axis.plot([r['step'] for r in records],counts,label=g)
    for r,n in zip(records,counts):
     v=r['layers'][k][g];rows.append([seed,r['step'],k,g,n,len(v['overall_missed']) if v['overall_missed'] is not None else '',v['temporal_overlap']['jaccard'] if v['temporal_overlap'] else '',v['half_overlap']['jaccard'] if v['half_overlap'] else ''])
   axis.set_ylabel(k+' selected');axis.legend(fontsize=7,ncol=4)
  axes[-1].set_xlabel('Environment steps');f.tight_layout();f.savefig(OUT/f'detection_seed{seed}.png');plt.close(f)
  if records and records[-1]['layers']:
   f,axes=plt.subplots(3,1,figsize=(13,9))
   for axis,k in zip(axes,['A1','A2','H']):
    groups=records[-1]['layers'][k];names=list(groups);arr=np.array([groups[g]['score'] if groups[g]['score'] is not None else [np.nan]*64 for g in names]);im=axis.imshow(arr,aspect='auto',vmin=0,vmax=max(3,np.nanmax(arr)));axis.set_yticks(range(len(names)));axis.set_yticklabels(names);axis.set_title(k+' final normalized scores');f.colorbar(im,ax=axis)
   f.tight_layout();f.savefig(OUT/f'heatmap_seed{seed}.png');plt.close(f)
 ax.set_xlabel('Environment steps');ax.set_ylabel('Test win rate');ax.legend();fig.tight_layout();fig.savefig(OUT/'training_curves.png');plt.close(fig)
 with (OUT/'detection_summary.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['seed','step','layer','group','selected','overall_missed','temporal_jaccard','half_jaccard']);w.writerows(rows)
 states={}
 for seed in SEEDS:
  run=ROOT/'runs'/f'qmix_2s3z_detect_seed{seed}_20261101_attempt003'
  states[seed]={n:json.loads((run/n).read_text()) for n in ['launcher_status.json','training_complete.json'] if (run/n).exists()}
  d=run/'detection'
  readlines=lambda name: [json.loads(x) for x in (d/name).read_text().splitlines() if x.strip()] if (d/name).exists() else []
  records=readlines('records.jsonl');audits=readlines('audit.jsonl');errors=readlines('errors.jsonl')
  states[seed]['diagnostic_review']=dict(successful_detections=len(records),expected=20,errors=len(errors),audit_differences=sum(not x['training_state_unchanged'] or not x['rng_restored'] for x in audits),raw_windows=len(list(d.glob('*/input.th'))))
  states[seed]['diagnostic_review']['complete']=len(records)==20 and not errors and len(audits)==20 and all(x['training_state_unchanged'] and x['rng_restored'] for x in audits)

 (OUT/'REPORT.md').write_text('# 2s3z QMIX pure detection\n\nCompletion evidence (training steps and process status):\n\n```json\n'+json.dumps(states,indent=2)+'\n```\n\nSee detection_summary.csv and per-seed curves/heatmaps. Raw snapshots and last-32-episode windows remain in runs/*/detection. Empty-set Jaccard is undefined. Per-group scores are separately normalized; missed sets describe threshold dilution, not causal importance. Temporal stability combines model and data-window changes; episode halves test within-window reproducibility. No neuron intervention or reset is enabled. Detection audit/errors must be inspected before drawing conclusions.\n')
if __name__=='__main__':main()
