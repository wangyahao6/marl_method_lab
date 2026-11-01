"""Summarize actual heterogeneous ABA runs; single seed, no across-seed inference."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/hetero_agent_softprotect_3535_20261101';PARENT=ROOT/'runs/hetero_Aprefix_seed3535_20261101_attempt001'
GROUPS=['N','T1','G1','R1','T2','G2','R2','T12','G12','R12']
def records(run):
 p=run/'evaluations.jsonl';return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def area(points,start,end):
 points=sorted({x:y for x,y in points}.items());x=np.array([p[0] for p in points]);y=np.array([p[1] for p in points])
 if len(x)<2 or x[0]>start or x[-1]<end:return None
 xx=np.r_[start,x[(x>start)&(x<end)],end];return float(np.trapz(np.interp(xx,x,y),xx)/(end-start))
def main():
 OUT.mkdir(exist_ok=True,parents=True);prefix=records(PARENT);base=next((r for r in prefix if r['label']=='A1_end'),None);rows=[];reviews=[]
 for task in ['R','M']:
  fig,axes=plt.subplots(1,3,figsize=(17,5))
  for group in GROUPS:
   run=ROOT/'runs'/f'hetero_{task}_{group}_seed3535_20261101_attempt001';rs=records(run);status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
   reviews.append(dict(task=task,group=group,status=status.get('status','not_started'),t_env=status.get('t_env')))
   init=next((r for r in rs if r['label']=='return_A_before_update'),None);bend=next((r for r in rs if r['label']=='B_end'),None);final=next((r for r in rs if r['label']=='A2_end'),None)
   restore=json.loads((run/'branch_restore.json').read_text()) if (run/'branch_restore.json').exists() else {};origin=restore.get('t_env',1000000)
   bpoints=[(r['t_env'],r['win_rate']) for r in rs if r['phase']==1 and r['task']==task]
   old=[(origin,base['win_rate'])] if base else [];old += [(r['t_env'],r['win_rate']) for r in rs if r['phase']==1 and r['task']=='A']
   back=([(init['t_env'],init['win_rate'])] if init else [])+[(r['t_env'],r['win_rate']) for r in rs if r['phase']==2 and r['task']=='A']
   for ax,points in zip(axes,[bpoints,old,back]):
    if points:ax.plot([(x-(init['t_env'] if ax is axes[2] and init else origin))/1e6 for x,y in points],[y for x,y in points],label=group)
   if not (base and init and bend and final):continue
   a2start=init['t_env'];threshold=max(0,base['win_rate']-.05);regular=[r for r in rs if r['phase']==2 and not r['focus'] and r['task']=='A'];recover=None
   for first,second in zip(regular,regular[1:]):
    if first['win_rate']>=threshold and second['win_rate']>=threshold:recover=second['t_env']-a2start;break
   protection=json.loads((run/'protection.json').read_text())
   rows.append(dict(task=task,group=group,seed=3535,A1_win=base['win_rate'],return_A_win=init['win_rate'],forgetting=base['win_rate']-init['win_rate'],B_end_win=bend['win_rate'],A2_end_win=final['win_rate'],B_mean_auc=area(bpoints,origin,bend['t_env']),return_200k_mean_auc=area(back,a2start,a2start+200000),recovery_confirmed_steps=recover,protected_parameters=protection['unique_parameters'],actual_terminal=final['t_env']))
  for ax,title in zip(axes,['B learning','A retention during B','Return A recovery']):ax.set_title(task+' '+title);ax.set_xlabel('Training steps from phase start (million)');ax.set_ylim(-.03,1.03);ax.grid(alpha=.2)
  axes[0].set_ylabel('Real win rate');h,l=axes[0].get_legend_handles_labels()
  if h:fig.legend(h,l,loc='lower center',ncol=10)
  fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(OUT/f'curves_{task}.png',dpi=160);plt.close(fig)
 if rows:
  with (OUT/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 contrasts=[]
 for r in rows:
  if r['group'].startswith('T'):
   for control in ['N','G'+r['group'][1:],'R'+r['group'][1:]]:
    c=next((x for x in rows if x['task']==r['task'] and x['group']==control),None)
    if c:contrasts.append(dict(task=r['task'],method=r['group'],control=control,return_A_difference=r['return_A_win']-c['return_A_win'],return_200k_auc_difference=r['return_200k_mean_auc']-c['return_200k_mean_auc'] if r['return_200k_mean_auc'] is not None and c['return_200k_mean_auc'] is not None else None,B_end_difference=r['B_end_win']-c['B_end_win']))
 (OUT/'contrasts.json').write_text(json.dumps(contrasts,indent=2));(OUT/'review.json').write_text(json.dumps(dict(prefix=base,branches=reviews,complete_comparisons=len(rows),expected=20),indent=2))
 (OUT/'REPORT.md').write_text('# Heterogeneous Agent soft protection\n\nSingle training seed 3535, common A prefix. See summary.csv, contrasts.json, curves_R/M.png and review.json. A1/A2 joint budget exceeds individual-layer budgets; no synergy claim. Full-episode absolute boundaries can overshoot. New SC2 process on B fork, distribution RNG restored; no engine-level bitwise continuation claim. Completion requires every branch training state plus focus evaluations; missing results remain missing.\n\n'+json.dumps(reviews,indent=2))
if __name__=='__main__':main()
