"""Detection-time neuron identities and small analysis tables; no tensor/RNG work."""
import csv
import json
from pathlib import Path

PATHS={'A':'agent.net[2]','W':'mixer.hyper_w_final[2]','V':'mixer.V[2]'}

def append_csv(path, rows):
    if not rows:return
    exists=path.exists()
    with path.open('a',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        if not exists:writer.writeheader()
        writer.writerows(rows)


def detection_record(result, previous, effective, before_scales, args, output, config, detection_dir):
    context=dict(run_id=output.name,scenario=args.scenario,condition=args.condition,seed=args.seed,
                 protocol_id=args.protocol_id,t_env=result['t_env'],phase=result['phase'],task=result['task'])
    changes={};rows=[]
    for label,layer in result['layers'].items():
        defined=layer['selected'] is not None
        old=set(previous[label]);new=set(effective[label])
        union=old|new
        changes[label]=dict(defined=defined,selected=layer['selected'],previous=sorted(old),
            effective_next=sorted(new),entered=sorted(new-old),retained=sorted(new&old),
            exited=sorted(old-new),jaccard=(len(new&old)/len(union) if union else None) if defined else None)
        width=len(layer['score']) if layer.get('score') is not None else len(before_scales[label])
        for j in range(width):
            change=('undefined_keep' if not defined else 'retained' if j in new&old
                    else 'entered' if j in new-old else 'exited' if j in old-new else 'unselected')
            rows.append(dict(**context,layer=label,layer_path=PATHS[label],neuron_index=j,
                neuron_id=f'{label}:{j}',episodes=len(result['episodes']),valid_states=result['valid_states'],
                activity=layer['activity'][j] if layer.get('activity') is not None else None,
                normalized_score=layer['score'][j] if layer.get('score') is not None else None,
                threshold=config['threshold_beta'],defined=defined,
                selected=(j in new) if defined else None,previous_selected=j in old,
                effective_next_selected=j in new,change=change,
                scale_before=before_scales[label][j],
                scale_at_switch=config['step_scale'] if j in new else before_scales[label][j],
                result_path=str(detection_dir.relative_to(output)/'result.json')))
    return dict(context=context,changes=changes),rows


def write_detection(output, event, rows):
    with (output/'selection_history.jsonl').open('a') as stream:
        stream.write(json.dumps(event,allow_nan=False)+'\n')
    append_csv(output/'neuron_detections.csv',rows)


def export_experiment(root):
    """Can be rerun during training; frequencies stay within each run/layer/unit."""
    root=Path(root);all_rows=[];events=[];frequency={};adjacent=[]
    for run in sorted((root/'runs').glob('*')):
        path=run/'neuron_detections.csv'
        if not path.exists():continue
        with path.open(newline='') as stream:records=list(csv.DictReader(stream))
        all_rows.extend(records)
        for row in records:
            key=(row['run_id'],row['layer'],row['neuron_index'])
            if key not in frequency:
                frequency[key]=dict(run_id=row['run_id'],scenario=row['scenario'],seed=row['seed'],
                    layer=row['layer'],neuron_index=row['neuron_index'],neuron_id=row['neuron_id'],
                    valid_detections=0,selected_count=0,first_selected_step=None,last_selected_step=None)
            f=frequency[key]
            if row['defined']!='True':continue
            f['valid_detections']+=1
            if row['selected']=='True':
                f['selected_count']+=1
                if f['first_selected_step'] is None:f['first_selected_step']=int(row['t_env'])
                f['last_selected_step']=int(row['t_env'])
        for line in (run/'selection_history.jsonl').read_text().splitlines():
            event=json.loads(line);events.append(event)
            for label,c in event['changes'].items():
                adjacent.append(dict(**event['context'],layer=label,defined=c['defined'],
                    selected_count=len(c['selected']) if c['defined'] else None,
                    effective_count=len(c['effective_next']),entered=len(c['entered']),
                    retained=len(c['retained']),exited=len(c['exited']),jaccard=c['jaccard']))
    if not all_rows:return
    out=root/'results/neuron_analysis';out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('detections.csv',all_rows),('changes.csv',adjacent),('frequency.csv',list(frequency.values()))]:
        if name=='frequency.csv':
            for f in rows:f['selection_rate']=f['selected_count']/f['valid_detections'] if f['valid_detections'] else None
        temp=out/(name+'.tmp')
        with temp.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        temp.replace(out/name)
    (out/'selection_history.json').write_text(json.dumps(events,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    export_experiment(parser.parse_args().root)
