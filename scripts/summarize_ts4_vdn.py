"""Compact export of actual TS4 win rates and neuron observations."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ts4_vdn_agent_4242_20261102'
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
def export(name,data):
    if not data:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in data for k in row)));w.writeheader();w.writerows(data)
def main():
    evaluations=[];counts=[];changes=[];retention=[];report=['# TS4 VDN: actual observations','']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for group in ['N','A1_linear','A2_linear']:
        run=ROOT/'runs'/f'ts4_vdn_{group}_seed4242_20261102_attempt001'
        status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'status':'missing'}
        report.append(f"- {group}: {status.get('status')}; actual steps {status.get('t_env')}")
        ev=rows(run/'evaluations.jsonl')
        evaluations.extend(dict(group=group,**row) for row in ev)
        first={}
        for row in ev:
            if row['label']=='stage_end':first.setdefault(row['task'],row)
        for row in ev:
            ref=first.get(row['task'])
            if ref and row['phase']>ref['phase']:
                retention.append(dict(group=group,**row,reference_phase=ref['phase'],reference_win_rate=ref['win_rate'],retention_change_pp=100*(row['win_rate']-ref['win_rate'])))
        for row in rows(run/'detection_counts.jsonl'):
            for layer,count in row['counts'].items():
                counts.append(dict(group=group,t_env=row['t_env'],phase=row['phase'],label=row['label'],layer=layer,count=count,ids=json.dumps(row['ids'][layer])))
        for item in rows(run/'neuron_changes.jsonl'):
            for layer,value in item.pop('layers').items():
                changes.append(dict(group=group,layer=layer,**item,**{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in value.items()}))
        for ax,task in zip(axes,['stalker','zealot','colossus']):
            values=[e for e in ev if e['task']==task]
            ax.plot([x['t_env'] for x in values],[100*x['win_rate'] for x in values],label=group,marker='.',markersize=2)
            ax.set_title(task);ax.set_xlabel('training environment steps');ax.set_ylim(0,100)
            for boundary in [1000000,1500000,2000000]:ax.axvline(boundary,color='gray',alpha=.25)
    axes[0].set_ylabel('greedy win rate (%)');axes[-1].legend();fig.tight_layout();fig.savefig(OUT/'win_rates.png',dpi=160);plt.close(fig)
    export('retention.csv',retention);export('evaluations.csv',evaluations);export('detection_counts.csv',counts);export('neuron_changes.csv',changes)
    report+=['','Single training seed. Scores proxy state-weighted activity; drift and set turnover do not establish causal importance or harmful interference. Selected/unselected parameter-drift contrasts are descriptive, not matched causal controls.','Raw evidence remains in each run. Verify phase budgets, diagnostic errors and evaluation completeness before claiming completion.']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
if __name__=='__main__':main()
