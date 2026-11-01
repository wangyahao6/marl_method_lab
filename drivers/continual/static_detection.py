"""Pure snapshot detection on complete recent episodes; isolated model copies."""
import copy,json,random,traceback
from collections import deque
from pathlib import Path
import numpy as np
import torch as th
from protection import cpu_episode,combine,digest,rng_state
from detection_common import sensitivity

def score(a):
    s=a/a.mean() if a.mean()>0 else None
    return dict(activity=a.tolist(),score=None if s is None else s.tolist(),selected=None if s is None else np.flatnonzero(s>3).tolist())
def overlap(a,b):
    if a is None or b is None:return None
    a,b=set(a),set(b)
    return dict(intersection=sorted(a&b),union=sorted(a|b),jaccard=len(a&b)/len(a|b) if a|b else None)

class Detector:
    def __init__(self,learner,args):
        self.learner=learner;self.args=args;self.fifo=deque(maxlen=32);self.next=100000;self.previous={};self.last=-1
        self.out=Path(args.local_results_path)/'detection';self.out.mkdir()
    def observe(self,batch,env,step):
        types=[int(env.agents[i].unit_type) for i in range(self.args.n_agents)]
        labels={int(env.stalker_id):'stalker',int(env.zealot_id):'zealot'}
        self.fifo.append((cpu_episode(batch),[labels.get(t,str(t)) for t in types]))
        if step>=self.next or step>=self.args.t_max:
            nominal=self.next
            while self.next<=step:self.next+=100000
            self.detect(step,nominal)
    @th.no_grad()
    def detect(self,step,nominal):
        if step==self.last:return
        self.last=step
        state=rng_state();before=digest([self.learner.mac.agent.state_dict(),self.learner.mixer.state_dict(),self.learner.target_mac.agent.state_dict(),self.learner.target_mixer.state_dict(),self.learner.optimiser.state_dict(),self.learner.mac.hidden_states])
        try:
            self.compute(step,nominal)
        except Exception:
            with (self.out/'errors.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,error=traceback.format_exc()))+'\n')
        finally:
            random.setstate(state[0]);np.random.set_state(state[1]);th.set_rng_state(state[2])
            if state[3]:th.cuda.set_rng_state_all(state[3])
            after=digest([self.learner.mac.agent.state_dict(),self.learner.mixer.state_dict(),self.learner.target_mac.agent.state_dict(),self.learner.target_mixer.state_dict(),self.learner.optimiser.state_dict(),self.learner.mac.hidden_states])
            with (self.out/'audit.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,training_state_unchanged=before==after,rng_restored=digest(state)==digest(rng_state())))+'\n')
    def compute(self,step,nominal):
        batch=combine([x[0] for x in self.fifo]);batch.to(self.args.device)
        mac=copy.deepcopy(self.learner.mac);mix=copy.deepcopy(self.learner.mixer)
        for network in [mac.agent,mix]:
            for module in network.modules():module._forward_hooks.clear()
        mac.agent.eval();mix.eval();capt={k:[] for k in ['A1','A2','H']};handles=[]
        for k,m in [('A1',mac.agent.net[2]),('A2',mac.agent.net[4]),('H',mac.agent.rnn)]:
            handles.append(m.register_forward_hook(lambda m,i,o,k=k:capt[k].append(o.detach().clone())))
        mac.init_hidden(batch.batch_size)
        q=th.stack([mac.forward(batch,t=t) for t in range(batch.max_seq_length)],1)
        for h in handles:h.remove()
        mask=batch['filled'][:,:-1].squeeze(-1).bool()
        lengths=batch['filled'].squeeze(-1).sum(1)-1
        good=mask & (th.arange(batch.max_seq_length-1,device=mask.device)[None,:]<lengths[:,None])
        c,_=sensitivity(mix,q[:,:-1][good],batch['actions'][:,:-1].squeeze(-1)[good],batch['avail_actions'][:,:-1][good],batch['state'][:,:-1][good])
        dest=self.out/str(step);dest.mkdir()
        th.save(dict(batch=combine([x[0] for x in self.fifo]),args=vars(self.args),agent={k:v.cpu() for k,v in mac.agent.state_dict().items()},mixer={k:v.cpu() for k,v in mix.state_dict().items()},types=[x[1] for x in self.fifo]),dest/'input.th')
        ep=np.broadcast_to(np.arange(batch.batch_size)[:,None],good.shape)[good.cpu().numpy()]
        typearr=np.array([x[1] for x in self.fifo])[ep]
        weights=c/c.mean() if c.mean()>0 else None
        result=dict(step=step,nominal=nominal,episodes=len(self.fifo),valid_states=len(c),mean_sensitivity=float(c.mean()),types=[x[1] for x in self.fifo],layers={})
        if weights is not None:
            for k,values in capt.items():
                z=th.stack([v.reshape(batch.batch_size,self.args.n_agents,-1) for v in values],1)[:,:-1][good].abs().cpu().numpy().astype(float)
                groups={'overall':np.ones(z.shape[:2],bool)}
                groups.update({'type_'+t:typearr==t for t in sorted(set(typearr.flatten()))})
                groups.update({'agent_'+str(i):np.broadcast_to(np.arange(self.args.n_agents)[None,:]==i,z.shape[:2]) for i in range(self.args.n_agents)})
                records={}
                for name,g in groups.items():
                    def calc(sel):
                        den=g[sel].sum();return score((z[sel]*g[sel,:,None]*weights[sel,None,None]).sum((0,1))/den) if den else dict(activity=None,score=None,selected=None)
                    record=calc(np.ones(len(z),bool))
                    halves=[calc(ep%2==p)['selected'] for p in [0,1]]
                    record['half_overlap']=overlap(*halves)
                    record['temporal_overlap']=overlap(self.previous.get((k,name)),record['selected'])
                    self.previous[k,name]=record['selected'];records[name]=record
                overall=records['overall']['selected'];extra={}
                for name,record in records.items():
                    record['overall_overlap']=overlap(overall,record['selected'])
                    record['overall_missed']=sorted(set(record['selected'])-set(overall)) if overall is not None and record['selected'] is not None else None
                result['layers'][k]=records
        (dest/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        with (self.out/'records.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
