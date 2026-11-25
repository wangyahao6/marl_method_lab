"""VDN activation screening with optional state weighting and overall top-k for one feedforward layer."""
import copy
import torch as th
from controllers import REGISTRY as controllers
from common.checkpoints import cpu_snapshot
from memory import History
from common.runtime import restore_rng, rng_state
from memory.episodes import combine
from ts4_diagnostics import vdn_sensitivity

@th.no_grad()
def select(learner,args,fifo,out=None,*,save_input=True,state_weighted=True):
    saved=rng_state()
    try:
        cpu=combine([x[0] for x in fifo])
        if save_input:
            History.save(out/'screen_input.th', dict(batch=cpu,args=vars(args),agent=cpu_snapshot(learner.mac.agent)))
        batch=copy.deepcopy(cpu);batch.to(args.device)
        mac=controllers[args.mac](batch.scheme,batch.groups,args);mac.agent.load_state_dict(learner.mac.agent.state_dict());mac.agent.to(args.device);mac.agent.eval()
        layer, index = ('A', 2) if args.agent == 'single_ff_rnn' else ('A2', 4)
        captured=[];handle=mac.agent.net[index].register_forward_hook(lambda m,i,o:captured.append(o.detach()))
        try:
            mac.init_hidden(batch.batch_size)
            outputs = [mac.forward(batch, t=t) for t in range(batch.max_seq_length)]
        finally:
            handle.remove()
        lengths=batch['filled'].squeeze(-1).sum(1)-1
        good=batch['filled'][:,:-1].squeeze(-1).bool() & (th.arange(batch.max_seq_length-1,device=args.device)[None,:]<lengths[:,None])
        z=th.stack([v.reshape(batch.batch_size,args.n_agents,-1) for v in captured],1)[:,:-1][good].abs().mean(1)
        cm = None
        if state_weighted:
            q = th.stack(outputs, 1)[:, :-1]
            c = vdn_sensitivity(q[good], batch['actions'][:, :-1].squeeze(-1)[good],
                                batch['avail_actions'][:, :-1][good])
            cm = float(c.mean()) if len(c) else 0.
            mean_activity = (z * (c / cm)[:, None]).mean(0) if cm > 0 else None
        else:
            mean_activity = z.mean(0) if len(z) else None
        activity = None if mean_activity is None else mean_activity.cpu().numpy().astype(float)
        scores=activity/activity.mean() if activity is not None and activity.mean()>0 else None
        top_k = args.top_k
        ids = None
        if activity is not None and top_k is not None:
            ids = sorted(sorted(range(len(activity)), key=lambda j: (-activity[j], j))[:top_k])
        return {layer:dict(sets={'overall':ids},activity=None if activity is None else activity.tolist(),scores=None if scores is None else scores.tolist(),states=int(good.sum()),c_mean=cm,k=top_k,tie_break='ascending index',selection='scores_only' if top_k is None else 'top_k',defined=scores is not None if top_k is None else ids is not None)}
    finally:restore_rng(saved)
