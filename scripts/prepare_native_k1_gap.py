"""Plan four native TKPR-K1 branches and audit full, identical FT historical inputs."""
import argparse
import csv
import json
from pathlib import Path
import sys
import subprocess

import numpy as np
import torch as th

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'drivers/continual'),str(ROOT/'scripts')]
from action_set_protocol import validate,validate_fork,matching_fixed_reference_protocol
from common.preference_prefix import reference_input_hash
from drivers.continual.memory.fixed_reference import file_sha256
from learners.ranking_preference import reference_order,full_rank_pairs,_topk_actions,topk_preference
from common.runtime import episode_spec
from components.episode_buffer import ReplayBuffer
from learners.action_set_preference import batch_episodes,decision_masks

BATCH='ts5_native_tkpr_k1_gap_20261126'


def read(path):return json.loads(path.read_text())
def write(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def inventory(old,legal,decisions,gap):
    """Diagnostic reconstruction uses exactly the native helper calls; never used to select training data."""
    q,_,_,_=reference_order(old,legal,.1)
    active=decisions&(legal.sum(-1)>=2)
    _,_,_,pairs=full_rank_pairs(q,legal,active,dict(min_gap=gap))
    effective=(legal.sum(-1)-1).clamp_min(0).clamp_max(1)
    top=_topk_actions(old,legal,effective)
    selected=pairs&(top.unsqueeze(-1)|top.unsqueeze(-2))
    pair_counts=selected.sum((-1,-2))
    loss,count,summary=topk_preference(old,old,legal,decisions,dict(k=1,weight=1.,temperature=1.,min_gap=gap,q_scale_floor=.1))
    if int(summary[2])!=int(pair_counts.sum()) or int(count)!=int((pair_counts>0).sum()):
        raise ValueError('Diagnostic pair counts do not match the unmodified native implementation')
    n=legal.sum(-1)
    gaps=(q.unsqueeze(-1)-q.unsqueeze(-2)).abs()[selected].numpy()
    return dict(decision_states=int(decisions.sum()),native_at_least_two_actions=int(active.sum()),
        states_with_pairs=int((pair_counts>0).sum()),zero_pair_decision_states=int((decisions&(pair_counts==0)).sum()),
        selected_pairs=int(pair_counts.sum()),selected_pairs_n2_n3=int(pair_counts[decisions&((n==2)|(n==3))].sum()),
        decision_states_n2_n3=int((decisions&((n==2)|(n==3))).sum()),
        qgap_mean=float(gaps.mean()) if gaps.size else None,qgap_median=float(np.median(gaps)) if gaps.size else None,
        qgap_std=float(gaps.std()) if gaps.size else None,qgap_min=float(gaps.min()) if gaps.size else None,
        native_loss_at_reference=float(loss)),selected


def prepare():
    out=ROOT/'results'/BATCH;out.mkdir(parents=True,exist_ok=True)
    configdir=ROOT/'configs/continual/action_preference/ts5/native_k1_gap';configdir.mkdir(parents=True,exist_ok=True)
    tasks=[];evaluations=[];audit=[];provenance=[]
    implementation=dict(baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),universal_newlines=True).strip(),
        source_sha256={name:file_sha256(ROOT/name) for name in ('src/learners/ranking_preference.py','src/learners/q_learner.py')})
    (out/'implementation_manifest.json').write_text(json.dumps(implementation,indent=2)+'\n')
    python=str(ROOT/'.venv/bin/python')
    for seed in (2222,3535):
        parent=ROOT/'runs/tc_coverage'/('ts5_tc_coverage_20261123_reference_seed%d'%seed)
        parent_protocol=read(parent/'protocol.json');checkpoint=read(parent/'retention.json')['stages'][0]['checkpoint']
        scheme,groups,preprocess=episode_spec(read(parent/'env_info.json'))
        scheme=ReplayBuffer(scheme,groups,1,2,preprocess=preprocess).scheme
        identity=dict(seed=seed,parent_run_id=parent.name,s_checkpoint=str(parent/checkpoint),
            s_agent_sha256=file_sha256(parent/checkpoint/'agent.th'),s_optimizer_sha256=file_sha256(parent/checkpoint/'opt.th'),
            s_return_state_sha256=file_sha256(parent/'return_state.th'))
        for phase in range(4):
            source=parent/'action_set_memory'/('phase%04d'%phase)/'success.th';pool=th.load(source,map_location='cpu')
            batch,_=batch_episodes(pool['episodes'],scheme,groups,'cpu',include_good=False,include_old_q=True)
            legal,decisions=decision_masks(batch);old=batch['old_q_values'][:,:-1]
            common=dict(identity,phase=phase,task=pool['task'],episodes=len(pool['episodes']),
                stored_decision_rows=sum(ep['length']*old.size(-2) for ep in pool['episodes']),
                source_pool=str(source),pool_sha256=file_sha256(source),input_sha256=reference_input_hash(pool))
            masks={}
            for gap in (0,.5):
                values,mask=inventory(old,legal,decisions,gap)
                audit.append(dict(common,min_gap=gap,**values));masks[gap]=mask
            if (masks[.5]&~masks[0]).any():raise ValueError('Gap0.5 added pairs outside the native gap0 inventory')
            provenance.append(common)
        for gap,tag in ((0,'gap0'),(.5,'gap05')):
            run_id='%s_%s_seed%d'%(BATCH,tag,seed)
            protocol={k:v for k,v in parent_protocol.items() if k not in ('id','method','pair_ablation_experiment','algorithm')}
            protocol.update(id=run_id,fixed_reference_history=True,topk_preference=dict(k=1,weight=1.,temperature=1.,min_gap=gap,q_scale_floor=.1))
            checked=validate(protocol,'vdn_1ff','topk_preference');validate_fork(checked,'shared-branch',str(parent))
            if not matching_fixed_reference_protocol(parent_protocol,checked):raise ValueError('Reference training protocol mismatch')
            path=configdir/('%s_seed%d.json'%(tag,seed));path.write_text(json.dumps(protocol,indent=2)+'\n')
            metadata=dict(experiment_group_id=BATCH,experiment_group_title='TS5 native TKPR-K1 gap-only ablation',
                branch_id=tag,branch_title='Native TKPR-K1 '+tag,attempt=1,seed=seed,min_gap=gap,results=str(out))
            run=ROOT/'runs'/run_id
            task=dict(id=run_id,argv=[python,'scripts/run.py','--suite','vdn_action_set','--algorithm','vdn_1ff',
                '--method','topk_preference','--seed',str(seed),'--protocol',str(path.relative_to(ROOT)),
                '--run-id',run_id,'--preference-role','shared-branch','--parent',str(parent)],
                cwd=str(ROOT),type='training',run_dir=str(run),dependencies=[],queue=BATCH,queue_limit=4,
                resources=dict(gpu=0,gpu_mib=1400,ram_mib=3500),metadata=metadata)
            tasks.append(task)
            evaluation=run_id+'_independent_final';evaluations.append(evaluation)
            tasks.append(dict(id=evaluation,argv=[python,'scripts/evaluate.py','--run',str(run),'--episodes','128'],
                cwd=str(ROOT),type='evaluation',dependencies=[run_id],queue=BATCH,queue_limit=4,
                resources=dict(gpu_mib=0,ram_mib=2500),metadata=metadata))
    tasks.append(dict(id=BATCH+'_report',argv=[python,'scripts/report_native_k1_gap.py',str(out/'plan.json')],
        cwd=str(ROOT),type='summary',dependencies=evaluations,dependency_mode='terminal',queue=BATCH,queue_limit=4,
        resources=dict(gpu_mib=0,ram_mib=3500),metadata=dict(experiment_group_id=BATCH,experiment_group_title='TS5 native TKPR-K1 gap-only ablation')))
    tasks.append(dict(id=BATCH+'_publication',argv=[python,'scripts/report_native_k1_gap.py',str(out/'plan.json'),'--publish'],
        cwd=str(ROOT),type='summary',dependencies=[BATCH+'_report'],queue=BATCH,queue_limit=4,
        resources=dict(gpu_mib=0,ram_mib=3500),metadata=dict(experiment_group_id=BATCH,experiment_group_title='TS5 native TKPR-K1 gap-only ablation')))
    plan=dict(id=BATCH,tasks=tasks,seeds=[2222,3535],method='topk_preference',k=1,new_training_tasks=4,
              reference_runs={str(s):'ts5_tc_coverage_20261123_reference_seed%d'%s for s in (2222,3535)})
    write(out/'historical_pair_inventory.csv',audit)
    (out/'reference_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(out/'plan.json'),training_tasks=4,inventory_rows=len(audit))))


if __name__=='__main__':
    th.set_num_threads(1);prepare()
