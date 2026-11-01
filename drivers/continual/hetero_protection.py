"""Boolean-union actual RMSprop increments, A1/A2; frozen boundary screening."""
import copy
import numpy as np
import torch as th
from controllers import REGISTRY as controllers
from modules.mixers.qmix import QMixer
from detection_common import sensitivity
from protection import combine,rng_state,digest
import random

@th.no_grad()
def select(learner,args,fifo,out):
    state=rng_state()
    try:
        cpu=combine([x[0] for x in fifo])
        th.save(dict(batch=cpu,args=vars(args),agent={k:v.detach().cpu().clone() for k,v in learner.mac.agent.state_dict().items()},mixer={k:v.detach().cpu().clone() for k,v in learner.mixer.state_dict().items()},types=[x[1] for x in fifo]),out/'screen_input.th')
        batch=copy.deepcopy(cpu);batch.to(args.device)
        mac=controllers[args.mac](batch.scheme,batch.groups,args);mix=QMixer(args)
        mac.agent.load_state_dict(learner.mac.agent.state_dict());mix.load_state_dict(learner.mixer.state_dict());mac.agent.to(args.device);mix.to(args.device)
        captured={'A1':[],'A2':[]};handles=[]
        for key,module in [('A1',mac.agent.net[2]),('A2',mac.agent.net[4])]:handles.append(module.register_forward_hook(lambda m,i,o,key=key:captured[key].append(o.detach().clone())))
        mac.init_hidden(batch.batch_size)
        q=th.stack([mac.forward(batch,t=t) for t in range(batch.max_seq_length)],1)
        for h in handles:h.remove()
        lengths=batch['filled'].squeeze(-1).sum(1)-1
        good=batch['filled'][:,:-1].squeeze(-1).bool() & (th.arange(batch.max_seq_length-1,device=args.device)[None,:]<lengths[:,None])
        c,_=sensitivity(mix,q[:,:-1][good],batch['actions'][:,:-1].squeeze(-1)[good],batch['avail_actions'][:,:-1][good],batch['state'][:,:-1][good])
        types=np.array([x[1] for x in fifo]);flat=np.broadcast_to(types[:,None,:],(*good.shape,args.n_agents))[good.cpu().numpy()]
        result={}
        for idx,key in enumerate(['A1','A2'],1):
            z=th.stack([v.reshape(batch.batch_size,args.n_agents,-1) for v in captured[key]],1)[:,:-1][good].abs().cpu().numpy().astype(float)
            scores={};sets={};undefined=float(c.mean())<=0
            for group,mask in [('overall',np.ones(z.shape[:2],bool)),('stalker',flat=='stalker'),('zealot',flat=='zealot')]:
                act=(z*mask[:,:,None]*(c/c.mean())[:,None,None]).sum((0,1))/mask.sum() if not undefined and mask.sum() else None
                s=act/act.mean() if act is not None and act.mean()>0 else None
                scores[group]=dict(score=None if s is None else s.tolist(),activity=None if act is None else act.tolist(),agent_state_pairs=int(mask.sum()))
                sets[group]=None if s is None else np.flatnonzero(s>3).tolist()
            valid=all(sets[x] is not None for x in ['stalker','zealot','overall'])
            T=sorted(set(sets['stalker'] or [])|set(sets['zealot'] or [])) if valid else []
            k=len(T);G=sorted(sorted(range(64),key=lambda j:(-scores['overall']['score'][j],j))[:k]) if valid else []
            R=sorted(np.random.default_rng(20261101+100*args.seed+idx).choice(64,k,replace=False).tolist())
            result[key]=dict(T=T,G=G,R=R,k=k,sets=sets,scores=scores,defined=valid)
        return result
    finally:
        random.setstate(state[0]);np.random.set_state(state[1]);th.set_rng_state(state[2])
        if state[3]:th.cuda.set_rng_state_all(state[3])

class AgentProtection:
    def __init__(self,learner,selection,group,record):
        self.learner=learner;self.active=False;self.t_env=0;self.count=0;self.record=record;agent=learner.mac.agent
        entries=[('net.0.weight',agent.net[0].weight),('net.0.bias',agent.net[0].bias),('net.3.weight',agent.net[3].weight),('net.3.bias',agent.net[3].bias),('rnn.weight_ih',agent.rnn.weight_ih)]
        masks={name:th.zeros_like(p,dtype=th.bool) for name,p in entries};self.ids={}
        if group!='N':
            kind=group[0];which=group[1:]
            for layer,n in [('A1','1'),('A2','2')]:
                ids=selection[layer][kind] if n in which else [];self.ids[layer]=ids
                if layer=='A1':masks['net.0.weight'][ids]=True;masks['net.0.bias'][ids]=True;masks['net.3.weight'][:,ids]=True
                else:masks['net.3.weight'][ids,:]=True;masks['net.3.bias'][ids]=True;masks['rnn.weight_ih'][:,ids]=True
        self.entries=[(name,p,masks[name]) for name,p in entries if masks[name].any()]
        self.unique_parameters=sum(int(mask.sum()) for _,p,mask in self.entries)
        self.original=learner.optimiser.step;learner.optimiser.step=self.step
    @th.no_grad()
    def step(self,*args,**kwargs):
        if not self.active or not self.entries:return self.original(*args,**kwargs)
        old=[p[mask].clone() for _,p,mask in self.entries];result=self.original(*args,**kwargs)
        audit=[]
        for (name,p,mask),v in zip(self.entries,old):
            raw=p[mask]-v;p[mask]=v+.1*raw
            if self.count%1000==0:audit.append(dict(parameter=name,elements=int(mask.sum()),raw_norm=float(raw.norm()),actual_norm=float((p[mask]-v).norm())))
        if audit:self.record(dict(t_env=self.t_env,scale=.1,parameters=audit))
        self.count+=1;return result
