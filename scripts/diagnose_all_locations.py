"""Read-only frozen-model diagnostics; no environment, optimizer, or protection."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch as th
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'drivers/continual')]
from controllers.basic_controller import BasicMAC
from modules.mixers.qmix import QMixer
from detection_common import sensitivity, parameter_hash


def scored(a):
    avg = float(np.mean(a))
    s = a/avg if avg > 0 else None
    return dict(activity=a.tolist(), score=None if s is None else s.tolist(),
                selected=None if s is None else np.flatnonzero(s > 3).tolist())


def aggregate(z, omega):
    return scored(np.einsum('n,nj->j', omega, np.abs(z))/len(omega))


def by_agent(z, omega, eligible):
    out=[]
    for i in range(z.shape[1]):
        valid=eligible[:,i]
        out.append(aggregate(z[valid,i],omega[valid]) if valid.any()
                   else dict(activity=None,score=None,selected=None))
    return out


def dilution(pooled, per):
    union=set().union(*(set(p['selected'] or []) for p in per))
    missed=union-set(pooled['selected'] or [])
    return dict(per_agent_counts=[None if p['selected'] is None else len(p['selected']) for p in per],
                union=sorted(union), diluted_ids=sorted(missed), diluted_count=len(missed))


@th.no_grad()
def analyze(path):
    data=th.load(path,map_location='cpu')
    args=SimpleNamespace(**data['args']);args.device='cpu';args.use_cuda=False
    batch=data['batch'];batch.to('cpu')
    mac=BasicMAC(batch.scheme,batch.groups,args);mac.agent.load_state_dict(data['agent'])
    mixer=QMixer(args);mixer.load_state_dict(data['mixer'])
    mac.agent.eval();mixer.eval()
    before=parameter_hash(mac,mixer)
    captures={k:[] for k in ['A1','A2','H']}
    handles=[]
    try:
        for key,module in [('A1',mac.agent.net[2]),('A2',mac.agent.net[4]),('H',mac.agent.rnn)]:
            handles.append(module.register_forward_hook(lambda m,i,o,k=key: captures[k].append(o.detach().clone())))
        mac.init_hidden(batch.batch_size)
        qs=th.stack([mac.forward(batch,t=t) for t in range(batch.max_seq_length)],1)
    finally:
        for handle in handles:handle.remove()
    mask=batch['filled'][:,:-1].float().clone()
    mask[:,1:]*=1-batch['terminated'][:,:-2].float()
    lengths=batch['filled'].squeeze(-1).sum(1)-1
    good=mask.squeeze(-1).bool() & (th.arange(batch.max_seq_length-1)[None,:]<lengths[:,None])
    states=batch['state'][:,:-1][good]
    q=qs[:,:-1][good]
    actions=batch['actions'][:,:-1].squeeze(-1)[good]
    avail=batch['avail_actions'][:,:-1][good]
    c,_=sensitivity(mixer,q,actions,avail,states)
    if c.mean() <= 0:
        return dict(input=str(path),undefined='zero_mean_sensitivity')
    omega=c/c.mean()
    episodes=np.broadcast_to(np.arange(batch.batch_size)[:,None],good.shape)[good.numpy()]
    eligible=(avail.sum(-1)>1).numpy()
    features={k:th.stack([v.reshape(batch.batch_size,mac.n_agents,-1) for v in values],1)[:,:-1][good].numpy().astype(np.float64) for k,values in captures.items()}
    selected_q=q.gather(2,actions.unsqueeze(-1)).squeeze(-1)
    w1=mixer.hyper_w_1(states).abs().reshape(-1,mac.n_agents,mixer.embed_dim)
    b1=mixer.hyper_b_1(states)
    hidden=th.nn.functional.elu(th.bmm(selected_q.unsqueeze(1),w1).squeeze(1)+b1)
    other={'U':mixer.hyper_w_1[:2](states),'W':mixer.hyper_w_final[:3](states),'V':mixer.V[:3](states),'M':hidden}
    result=dict(input=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        seed=args.seed,episodes=batch.batch_size,valid_states=len(c),mean_sensitivity=float(c.mean()),
        n_agents=mac.n_agents,obs_agent_id=args.obs_agent_id,obs_last_action=args.obs_last_action,
        threshold='strict > 3',weight='c/mean(c)',task='stalker',t_env=1000000,
        scope='A1 endpoint, before first soft-protection update; offline detection only',layers={},agent_analysis={})
    for key,z in features.items():
        pooled=aggregate(np.abs(z).mean(1),omega)
        per=by_agent(z,omega,np.ones_like(eligible))
        active=by_agent(z,omega,eligible)
        d=dilution(pooled,per)
        active_mean=np.divide((np.abs(z)*eligible[:,:,None]).sum(1),eligible.sum(1)[:,None],out=np.zeros((len(z),z.shape[-1])),where=eligible.sum(1)[:,None]>0)
        active_good=eligible.any(1)
        active_pooled=aggregate(active_mean[active_good],omega[active_good])
        halves=[]
        for part in [episodes%2==0,episodes%2==1]:
            pp=aggregate(np.abs(z[part]).mean(1),omega[part])
            aa=by_agent(z[part],omega[part],np.ones_like(eligible[part]))
            halves.append(dilution(pp,aa))
        # Episode-wise identity permutations retain within-episode temporal structure
        # and exactly preserve pooled activations. Exploratory identity null only.
        rng=np.random.default_rng(20261101)
        counts=[]
        # Aggregate sufficient statistics per episode first, to avoid copying trajectories.
        ep_num=np.stack([np.einsum('n,nij->ij',omega[episodes==e],np.abs(z[episodes==e])) for e in range(batch.batch_size)])
        for _ in range(200):
            perms=np.stack([rng.permutation(mac.n_agents) for e in range(batch.batch_size)])
            aa=ep_num[np.arange(batch.batch_size)[:,None],perms].sum(0)/len(z)
            counts.append(dilution(pooled,[scored(x) for x in aa])['diluted_count'])
        result['layers'][key]=dict(width=z.shape[-1],**pooled)
        result['agent_analysis'][key]=dict(per_agent=per,**d,eligible_per_agent=active,
            eligible_samples=eligible.sum(0).tolist(),eligible_pooled=active_pooled,
            eligible_dilution=dilution(active_pooled,active),episode_halves=halves,
            identity_permutation=dict(repeats=200,seed=20261101,null_diluted_quantiles=np.quantile(counts,[0,.5,.95,1]).tolist(),
                exploratory_tail_fraction=(1+sum(x>=d['diluted_count'] for x in counts))/201))
    for key,z in other.items():result['layers'][key]=dict(width=z.shape[-1],**aggregate(z.numpy().astype(np.float64),omega))
    # Output coordinates are logged separately: they are not hidden-neuron locations.
    outputs={'Q_agent':q.numpy().astype(np.float64),'hyper_w1_abs':w1.flatten(1).numpy(),
             'hyper_w_final_abs':mixer.hyper_w_final(states).abs().numpy(),
             'hyper_b1':b1.numpy(),'V_output':mixer.V(states).numpy(),
             'Q_total':mixer(selected_q.unsqueeze(1),states.unsqueeze(1)).reshape(-1,1).numpy()}
    result['output_coordinates']={k:aggregate(np.abs(z).mean(1) if z.ndim==3 else z,omega) for k,z in outputs.items()}
    result['parameters_unchanged']=before==parameter_hash(mac,mixer)
    ref=json.loads((path.parent/'result.json').read_text())
    result['existing_WV_match']={k:dict(selected_match=set(result['layers'][k]['selected'])==set(ref['layers'][k]['selected']),max_score_error=float(np.max(np.abs(np.array(result['layers'][k]['score'])-ref['layers'][k]['score'])))) for k in ['W','V']}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',action='append',required=True);p.add_argument('--output',required=True);o=p.parse_args()
    th.set_num_threads(1)
    out=Path(o.output);out.mkdir(parents=True,exist_ok=False)
    results=[]
    for source in o.input:
        r=analyze(Path(source));results.append(r)
        (out/('seed'+str(r['seed'])+'.json')).write_text(json.dumps(r,indent=2,allow_nan=False))
        print(json.dumps(dict(seed=r['seed'],counts={k:len(v['selected']) for k,v in r['layers'].items()},agent={k:{x:v[x] for x in ['per_agent_counts','diluted_count','eligible_dilution','episode_halves','identity_permutation']} for k,v in r['agent_analysis'].items()},parameters_unchanged=r['parameters_unchanged'],existing_WV_match=r['existing_WV_match'])),flush=True)
    with (out/'counts.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['seed','layer','width','pooled_count','agent_counts','union_count','diluted_count'])
        for r in results:
            for k,v in r['layers'].items():
                a=r['agent_analysis'].get(k,{})
                w.writerow([r['seed'],k,v['width'],len(v['selected']),a.get('per_agent_counts',''),len(a['union']) if 'union' in a else '',a.get('diluted_count','')])
    (out/'provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=o.input,training=False,protection=False,environment_interactions=0),indent=2))

if __name__=='__main__':main()
