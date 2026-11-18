"""Frozen-model parameter, representation and output comparisons on fixed histories."""
import argparse,csv,hashlib,itertools,json,os,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch as th
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from controllers import REGISTRY

from modules.agents.rnn_agent import LEGACY_FC1_KEYS

def write(p,o):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(o,indent=2,allow_nan=False));tmp.replace(p)
def csvwrite(p,rows):
    if not rows:return
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def cka(x,y):
    x=x.astype(np.float64);y=y.astype(np.float64);x=x-x.mean(0);y=y-y.mean(0)
    den=np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
    return float(np.linalg.norm(x.T@y)**2/den) if den>0 else None
def pdistance(a,b):
    x=np.concatenate([v.reshape(-1) for v in a]).astype(np.float64);y=np.concatenate([v.reshape(-1) for v in b]).astype(np.float64)
    nx=np.linalg.norm(x);ny=np.linalg.norm(y)
    return dict(cosine=float(x@y/(nx*ny)) if nx*ny else None,normalized_l2=float(np.linalg.norm(x-y)/np.sqrt((nx*nx+ny*ny)/2)) if nx*nx+ny*ny else None)
@th.no_grad()
def forward(cfg,state,batch):
    args=SimpleNamespace(**dict(cfg,device='cpu',use_cuda=False))
    mac=REGISTRY[args.mac](batch.scheme,batch.groups,args);mac.agent.load_state_dict(state);mac.agent.eval();mac.init_hidden(batch.batch_size)
    captured={'A1':[],'A2':[]}
    handles=[module.register_forward_hook(lambda m,i,o,k=k:captured[k].append(o.detach().clone())) for k,module in [('A1',mac.agent.net[2]),('A2',mac.agent.net[4])]]
    qs=th.stack([mac.forward(batch,t=t) for t in range(batch.max_seq_length-1)],1)
    for h in handles:h.remove()
    lengths=batch['filled'].squeeze(-1).sum(1)-1
    valid=th.arange(batch.max_seq_length-1)[None,:]<lengths[:,None]
    avail=batch['avail_actions'][:,:-1].bool()
    live=valid[:,:,None]&avail[:,:,:,1:].any(-1)
    features={k:th.stack(v,1).reshape(batch.batch_size,batch.max_seq_length-1,args.n_agents,-1)[live].numpy() for k,v in captured.items()}
    return features,qs[live].numpy(),avail[live].numpy(),dict(episodes=batch.batch_size,valid_agent_states=int(live.sum()),valid_transitions=int(valid.sum()),decision_states=int((avail[live].sum(-1)>1).sum()))
def run(out,protocol):
    th.set_num_threads(1);np.random.seed(0);th.manual_seed(0);start=time.time()
    # Independent known invariance check, no environment or training.
    x=np.arange(60,dtype=float).reshape(20,3);y=x[:,::-1]*2+7
    checks=dict(cka_self=cka(x,x),cka_permutation_scale_shift=cka(x,y),identical_parameters=pdistance([x],[x]))
    write(out/'implementation_checks.json',checks)
    if abs(checks['cka_self']-1)>1e-10 or abs(checks['cka_permutation_scale_shift']-1)>1e-10:raise RuntimeError('CKA implementation check failed')
    queue=json.loads((ROOT/protocol['source_queue']).read_text());manifest=[];models={};missing=[]
    for name,node in queue['runs'].items():
        rd=ROOT/'runs'/name;sp=rd/'status.json';status=json.loads(sp.read_text()) if sp.exists() else {}
        if status.get('status')!='completed':missing.append(dict(run=name,path=node['path'],seed=node['seed'],status=status.get('status','not_started')));continue
        files=list((rd/'models').glob('*/agent.th'))
        if len(files)!=1:missing.append(dict(run=name,reason='expected one endpoint agent model'));continue
        f=files[0];state=th.load(f,map_location='cpu');cfg=json.loads((rd/'config.json').read_text())
        key=(node['seed'],node['path']);models[key]=(cfg,state)
        manifest.append(dict(run=name,seed=node['seed'],path=node['path'],model=str(f),sha256=sha(f),actual_cumulative_steps=status['t_env']))
    write(out/'manifest.json',dict(models=manifest,unavailable_at_snapshot=missing,source_queue=str(ROOT/protocol['source_queue']),created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    parameter_rows=[];representation_rows=[];output_rows=[];coverage=[];probe_records=[];errors=[]
    for seed in protocol['seeds']:
        available={p:(cfg,state) for (s,p),(cfg,state) in models.items() if s==seed}
        sets={}
        for target in 'SZC':
            prior=[c for c in 'SZC' if c!=target]
            paths=[target]+[p+target for p in prior]+[''.join(p)+target for p in itertools.permutations(prior)]+prior+[''.join(p) for p in itertools.permutations(prior)]
            sets[target]=[p for p in paths if p in available]
            for p in paths:coverage.append(dict(seed=seed,target=target,path=p,available=p in available,role='target_scratch' if p==target else 'one_prior_then_target' if p.endswith(target) and len(p)==2 else 'two_priors_then_target' if p.endswith(target) else 'prior_only'))
            for a,b in itertools.combinations(sets[target],2):
                sa=available[a][1];sb=available[b][1]
                for group,prefix in [('A1_incoming','net.0.'),('A2_incoming','net.3.'),('GRU','rnn.'),('Q_head','fc2.'),('used_network',None)]:
                    keys=[k for k in sa if k.startswith(prefix)] if prefix else [k for k in sa if k not in LEGACY_FC1_KEYS]
                    parameter_rows.append(dict(seed=seed,target=target,path_a=a,path_b=b,group=group,**pdistance([sa[k].numpy() for k in keys],[sb[k].numpy() for k in keys])))
        for probe in 'SZC':
            runpath=ROOT/'runs'/f'vdnh_{probe}_seed{seed}_20261103_attempt001'
            files=list(runpath.glob('detections/*/screen_input.th'))
            if len(files)!=1:errors.append(dict(seed=seed,probe=probe,error='probe missing/ambiguous'));continue
            pf=files[0];batch=th.load(pf,map_location='cpu')['batch'];batch.to('cpu');cache={}
            for path,(cfg,state) in available.items():
                try:cache[path]=forward(cfg,state,batch)
                except Exception:errors.append(dict(seed=seed,probe=probe,path=path,error=traceback.format_exc()))
            if not cache:continue
            probe_records.append(dict(seed=seed,probe=probe,file=str(pf),sha256=sha(pf),**next(iter(cache.values()))[3]))
            for target,paths in sets.items():
                for a,b in itertools.combinations([p for p in paths if p in cache],2):
                    fa,qa,avail,counts=cache[a];fb,qb,_,_=cache[b]
                    common=dict(seed=seed,target=target,probe=probe,path_a=a,path_b=b,agent_states=counts['valid_agent_states'])
                    for layer in ['A1','A2']:representation_rows.append(dict(**common,layer=layer,linear_cka=cka(fa[layer],fb[layer])))
                    decisions=avail.sum(-1)>1
                    aa=np.argmax(np.where(avail,qa,-np.inf),axis=-1);ab=np.argmax(np.where(avail,qb,-np.inf),axis=-1)
                    ma=(qa*avail).sum(-1)/avail.sum(-1);mb=(qb*avail).sum(-1)/avail.sum(-1)
                    adv=((qa-ma[:,None])-(qb-mb[:,None]))[avail]
                    output_rows.append(dict(**common,decision_states=int(decisions.sum()),greedy_agreement=float((aa[decisions]==ab[decisions]).mean()) if decisions.any() else None,legal_q_rmse=float(np.sqrt(np.mean((qa[avail]-qb[avail])**2))),legal_centered_q_rmse=float(np.sqrt(np.mean(adv**2)))))
            write(out/'status.json',dict(status='running',seed=seed,probe=probe,elapsed_seconds=time.time()-start))
            del cache,batch
    for name,rows in [('parameters',parameter_rows),('representations',representation_rows),('outputs',output_rows),('coverage',coverage)]:csvwrite(out/f'{name}.csv',rows)
    write(out/'probes.json',probe_records);write(out/'errors.json',errors)
    # Equal weight per training seed, no sample-level significance tests.
    aggregates=[]
    for family,rows,metrics,extra in [('parameters',parameter_rows,['cosine','normalized_l2'],['group']),('representations',representation_rows,['linear_cka'],['probe','layer']),('outputs',output_rows,['greedy_agreement','legal_q_rmse','legal_centered_q_rmse'],['probe'])]:
        groups={}
        for row in rows:
            key=tuple(row[k] for k in ['target','path_a','path_b']+extra);groups.setdefault(key,[]).append(row)
        for key,rs in groups.items():
            for metric in metrics:
                vals=[r[metric] for r in rs if r[metric] is not None]
                aggregates.append(dict(family=family,target=key[0],path_a=key[1],path_b=key[2],condition='/'.join(str(k) for k in key[3:]),metric=metric,n_seeds=len(vals),mean=float(np.mean(vals)) if vals else None,std=float(np.std(vals,ddof=1)) if len(vals)>1 else None))
    csvwrite(out/'seed_summary.csv',aggregates)
    from plot_network_similarity import plot_and_report
    plot_and_report(out,manifest,missing,errors,representation_rows,output_rows,aggregates)
    write(out/'status.json',dict(status='completed' if not errors else 'completed_with_errors',models=len(manifest),missing_models=len(missing),parameter_rows=len(parameter_rows),representation_rows=len(representation_rows),output_rows=len(output_rows),elapsed_seconds=time.time()-start))
def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True)
    try:run(out,json.loads(Path(a.protocol).read_text()))
    except BaseException:write(out/'status.json',dict(status='failed',error=traceback.format_exc()));raise
if __name__=='__main__':main()
