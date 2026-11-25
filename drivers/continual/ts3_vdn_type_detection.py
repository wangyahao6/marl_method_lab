"""A2 type-conditioned score>3 union under exact VDN team sensitivity."""
import copy
import numpy as np
import torch as th
from controllers import REGISTRY as controllers
from common.checkpoints import cpu_snapshot
from memory import History
from common.runtime import restore_rng, rng_state
from memory.episodes import combine
from ts4_diagnostics import vdn_sensitivity

@th.no_grad()
def select(learner,args,fifo,out):
    saved=rng_state()
    try:
        cpu=combine([x[0] for x in fifo]);types=np.array([x[1] for x in fifo])
        History.save(out/'screen_input.th', dict(batch=cpu,args=vars(args),agent=cpu_snapshot(learner.mac.agent),types=types.tolist()))
        batch=copy.deepcopy(cpu);batch.to(args.device)
        mac=controllers[args.mac](batch.scheme,batch.groups,args);mac.agent.load_state_dict(learner.mac.agent.state_dict());mac.agent.to(args.device);mac.agent.eval()
        captured=[];handle=mac.agent.net[4].register_forward_hook(lambda m,i,o:captured.append(o.detach().clone()))
        try:
            mac.init_hidden(batch.batch_size)
            q=th.stack([mac.forward(batch,t=t) for t in range(batch.max_seq_length)],1)[:,:-1]
        finally:handle.remove()
        lengths=batch['filled'].squeeze(-1).sum(1)-1
        good=batch['filled'][:,:-1].squeeze(-1).bool() & (th.arange(batch.max_seq_length-1,device=args.device)[None,:]<lengths[:,None])
        c=vdn_sensitivity(q[good],batch['actions'][:,:-1].squeeze(-1)[good],batch['avail_actions'][:,:-1][good]);cm=float(c.mean()) if len(c) else 0.
        z=th.stack([v.reshape(batch.batch_size,args.n_agents,-1) for v in captured],1)[:,:-1][good].abs().cpu().numpy().astype(float)
        flat=np.broadcast_to(types[:,None,:],(*good.shape,args.n_agents))[good.cpu().numpy()];weight=c.cpu().numpy()/cm if cm>0 else None
        records={};sets={}
        for name,mask in [('overall',np.ones(z.shape[:2],bool)),('stalker',flat=='stalker'),('zealot',flat=='zealot')]:
            activity=(z*mask[:,:,None]*weight[:,None,None]).sum((0,1))/mask.sum() if weight is not None and mask.sum() else None
            scores=activity/activity.mean() if activity is not None and activity.mean()>0 else None
            ids=np.flatnonzero(scores>3).tolist() if scores is not None else None
            sets[name]=ids;records[name]=dict(activity=None if activity is None else activity.tolist(),scores=None if scores is None else scores.tolist(),selected=ids,agent_state_pairs=int(mask.sum()))
        chosen=sorted(set(sets['stalker'])|set(sets['zealot'])) if all(sets[t] is not None for t in ['stalker','zealot']) else None
        return {'A2':dict(sets=dict(overall=chosen,stalker=sets['stalker'],zealot=sets['zealot'],overall_threshold=sets['overall']),group_statistics=records,states=int(good.sum()),c_mean=cm,selection='type_threshold_union',threshold_beta=3,defined=chosen is not None)}
    finally:restore_rng(saved)
