"""VDN state-weighted activity and overall top-k for one feedforward layer."""
import copy
import torch as th
from controllers import REGISTRY as controllers
from protection import combine,rng_state
from hetero_train import restore_rng
from ts4_diagnostics import vdn_sensitivity

@th.no_grad()
def select(learner,args,fifo,out=None,*,save_input=True):
    saved=rng_state()
    try:
        cpu=combine([x[0] for x in fifo])
        if save_input:
            th.save(dict(batch=cpu,args=vars(args),agent={k:v.detach().cpu().clone() for k,v in learner.mac.agent.state_dict().items()}),out/'screen_input.th')
        batch=copy.deepcopy(cpu);batch.to(args.device)
        mac=controllers[args.mac](batch.scheme,batch.groups,args);mac.agent.load_state_dict(learner.mac.agent.state_dict());mac.agent.to(args.device);mac.agent.eval()
        layer, index = ('A', 2) if args.agent == 'single_ff_rnn' else ('A2', 4)
        captured=[];handle=mac.agent.net[index].register_forward_hook(lambda m,i,o:captured.append(o.detach()))
        mac.init_hidden(batch.batch_size)
        try:q=th.stack([mac.forward(batch,t=t) for t in range(batch.max_seq_length)],1)[:,:-1]
        finally:handle.remove()
        lengths=batch['filled'].squeeze(-1).sum(1)-1
        good=batch['filled'][:,:-1].squeeze(-1).bool() & (th.arange(batch.max_seq_length-1,device=args.device)[None,:]<lengths[:,None])
        c=vdn_sensitivity(q[good],batch['actions'][:,:-1].squeeze(-1)[good],batch['avail_actions'][:,:-1][good]);cm=float(c.mean()) if len(c) else 0.
        z=th.stack([v.reshape(batch.batch_size,args.n_agents,-1) for v in captured],1)[:,:-1][good].abs().mean(1)
        activity=(z*(c/cm)[:,None]).mean(0).cpu().numpy().astype(float) if cm>0 else None
        scores=activity/activity.mean() if activity is not None and activity.mean()>0 else None
        ids=sorted(sorted(range(len(activity)),key=lambda j:(-activity[j],j))[:args.top_k]) if activity is not None else None
        return {layer:dict(sets={'overall':ids},activity=None if activity is None else activity.tolist(),scores=None if scores is None else scores.tolist(),states=int(good.sum()),c_mean=cm,k=args.top_k,tie_break='ascending index',selection='top_k',defined=ids is not None)}
    finally:restore_rng(saved)
