"""VDN-specific exact action sensitivity and longitudinal fixed-history probes."""
import copy,random
import numpy as np
import torch as th
from controllers import REGISTRY as controllers
from protection import combine,rng_state
from hetero_train import restore_rng

def vdn_sensitivity(q,actions,avail):
    chosen=q.gather(2,actions.unsqueeze(-1))
    allowed=avail.bool().clone();allowed.scatter_(2,actions.unsqueeze(-1),False)
    return ((q-chosen).abs()*allowed).max(-1).values.mean(-1)

@th.no_grad()
def score(learner,args,cpu):
    saved=rng_state()
    try:
        batch=copy.deepcopy(cpu);batch.to(args.device)
        mac=controllers[args.mac](batch.scheme,batch.groups,args)
        mac.agent.load_state_dict(learner.mac.agent.state_dict());mac.agent.to(args.device);mac.agent.eval()
        captured={'A1':[],'A2':[]}
        handles=[module.register_forward_hook(lambda m,i,o,key=key:captured[key].append(o.detach())) for key,module in [('A1',mac.agent.net[2]),('A2',mac.agent.net[4])]]
        mac.init_hidden(batch.batch_size)
        q=th.stack([mac.forward(batch,t=t) for t in range(batch.max_seq_length)],1)[:,:-1]
        for h in handles:h.remove()
        lengths=batch['filled'].squeeze(-1).sum(1)-1
        good=batch['filled'][:,:-1].squeeze(-1).bool() & (th.arange(batch.max_seq_length-1,device=args.device)[None,:]<lengths[:,None])
        q=q[good];actions=batch['actions'][:,:-1].squeeze(-1)[good];avail=batch['avail_actions'][:,:-1][good].bool()
        c=vdn_sensitivity(q,actions,avail)
        cm=float(c.mean());result={}
        for key in captured:
            z=th.stack([v.reshape(batch.batch_size,args.n_agents,-1) for v in captured[key]],1)[:,:-1][good].abs().mean(1)
            activity=(z*(c/cm)[:,None]).mean(0) if cm>0 else None
            scores=activity/activity.mean() if activity is not None and float(activity.mean())>0 else None
            ids=None if scores is None else th.where(scores>3)[0].cpu().tolist()
            result[key]=dict(sets={'overall':ids},count=None if ids is None else len(ids),scores=None if scores is None else scores.cpu().tolist(),activity=None if activity is None else activity.cpu().tolist(),states=int(good.sum()),c_mean=cm)
        return result
    finally:restore_rng(saved)

def vectors(agent):
    d={k:v.detach().cpu().numpy().copy() for k,v in agent.state_dict().items()}
    return {'A1':np.concatenate([d['net.0.weight'],d['net.0.bias'][:,None],d['net.3.weight'].T],1),
            'A2':np.concatenate([d['net.3.weight'],d['net.3.bias'][:,None],d['rnn.weight_ih'].T],1)}

def compare(old,new,old_vectors,new_vectors):
    result={}
    for key in ['A1','A2']:
        before=old[key];after=new[key];a=before['sets']['overall'];b=after['sets']['overall']
        delta=new_vectors[key]-old_vectors[key];absolute=np.sqrt((delta**2).mean(1));den=np.linalg.norm(old_vectors[key],axis=1);relative=np.linalg.norm(delta,axis=1)/np.maximum(den,1e-12)
        item=dict(parameter_rms_change=absolute.tolist(),parameter_relative_l2_change=relative.tolist())
        if a is not None and b is not None:
            x,y=set(a),set(b);u=x|y
            item.update(original_ids=a,current_ids=b,intersection=sorted(x&y),exited=sorted(x-y),entered=sorted(y-x),jaccard=len(x&y)/len(u) if u else None,retained_fraction=len(x&y)/len(x) if x else None,
                        original_selected_current_scores=[after['scores'][j] for j in a])
            for label,ids in [('selected',a),('unselected',[j for j in range(64) if j not in x])]:
                item[label]=dict(count=len(ids),mean_rms_change=float(absolute[ids].mean()) if ids else None,mean_relative_l2_change=float(relative[ids].mean()) if ids else None)
        result[key]=item
    return result
