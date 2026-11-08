"""Task-wise TD-loss GEM, projecting selected A2 parameter slices.

RL/RMSprop adaptation of Lopez-Paz & Ranzato (2017), eq.8. No upstream code
copied. The unselected increment is unconstrained; this is LOCAL, not full GEM.
"""
import copy,itertools,time,traceback
import numpy as np
import torch as th
from protection import combine


def partition(activity, width=64, count=21):
    if activity is None:return None
    a=np.asarray(activity,dtype=float)
    if a.shape!=(width,) or not np.isfinite(a).all():return None
    order=sorted(range(width),key=lambda j:(-a[j],j))
    # 1-based descending ranks: high1:21, middle22:42, low44:64, excluded43.
    return dict(high=sorted(order[:count]),middle=sorted(order[count:2*count]),
                low=sorted(order[-count:]),excluded=sorted(order[2*count:-count]),
                descending_order=order)


def above_threshold(item, threshold):
    scores = item.get('scores')
    if scores is None:
        scores = item.get('group_statistics', {}).get('overall', {}).get('scores')
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (64,) or not np.isfinite(scores).all():
        return None
    return np.flatnonzero(scores > threshold).tolist()


def indices_for(agent, ids):
    result=[];offset=0
    for name,p in agent.named_parameters():
        if not p.requires_grad:continue
        mask=th.zeros_like(p,dtype=th.bool)
        if name in ('net.3.weight','net.3.bias'):mask[ids]=True
        elif name=='rnn.weight_ih':mask[:,ids]=True
        result.append(mask.reshape(-1).nonzero().reshape(-1)+offset);offset+=p.numel()
    return th.cat(result),offset


def project_increment(d0, gradients):
    """Exact small active-set cone projection (<=4 constraints), float64.

    Solve min ||d-d0||^2 / 2, G d <= 0. Homogeneous row/vector normalization
    and active-set enumeration avoid a QP dependency or positive dual margin.
    Zero is always feasible; numerical fallbacks are logged, never a queue gate.
    """
    d0=np.asarray(d0,dtype=np.float64);G=np.asarray(gradients,dtype=np.float64)
    if G.size==0:return d0.copy(),dict(triggered=False,zero_rows=0,fallback=False,residual=0.)
    norms=np.linalg.norm(G,axis=1);valid=np.isfinite(G).all(1)&np.isfinite(norms)&(norms>0)
    A=G[valid]/norms[valid,None];n=float(np.linalg.norm(d0));bad=int((~valid).sum())
    if n==0 or not len(A):return d0.copy(),dict(triggered=False,zero_rows=bad,fallback=False,residual=0.)
    x=d0/n;dots=A@x
    if np.max(dots)<=0:return d0.copy(),dict(triggered=False,zero_rows=bad,fallback=False,residual=float(max(0,np.max(dots))))
    gram=A@A.T;best=np.zeros_like(x);cost=float(x@x);found=False
    for count in range(1,len(A)+1):
        for active in itertools.combinations(range(len(A)),count):
            ix=list(active);lam=np.linalg.lstsq(gram[np.ix_(ix,ix)],dots[ix],rcond=1e-12)[0]
            if np.min(lam)<-1e-9:continue
            candidate=x-A[ix].T@lam
            if np.max(A@candidate)>1e-9:continue
            change=float(np.sum((candidate-x)**2))
            if change<=cost+1e-12:best=candidate;cost=change;found=True
    return best*n,dict(triggered=True,zero_rows=bad,fallback=not found,residual=float(max(0,np.max(A@best))))


def td_loss(mac,target_mac,batch,args):
    """Same masked VDN 1-step semi-gradient TD objective as QLearner.train.

    The detached Double-Q target is fixed within this loss evaluation. Recurrent
    histories start at episode0; targets are never differentiated through.
    """
    rewards=batch['reward'][:,:-1];actions=batch['actions'][:,:-1]
    terminated=batch['terminated'][:,:-1].float();mask=batch['filled'][:,:-1].float().clone()
    mask[:,1:]*=1-terminated[:,:-1];avail=batch['avail_actions']
    mac.init_hidden(batch.batch_size)
    q=th.stack([mac.forward(batch,t=t) for t in range(batch.max_seq_length)],1)
    chosen=th.gather(q[:,:-1],3,actions).squeeze(3).sum(2,keepdim=True)
    with th.no_grad():
        target_mac.init_hidden(batch.batch_size)
        tq=th.stack([target_mac.forward(batch,t=t) for t in range(batch.max_seq_length)],1)[:,1:]
        tq[avail[:,1:]==0]=-9999999
        if args.double_q:
            live=q.detach().clone();live[avail==0]=-9999999
            aq=live[:,1:].max(3,keepdim=True)[1]
            nxt=th.gather(tq,3,aq).squeeze(3)
        else:nxt=tq.max(3)[0]
        targets=rewards+args.gamma*(1-terminated)*nxt.sum(2,keepdim=True)
    loss=(((chosen-targets)*mask)**2).sum()/mask.sum()
    return loss


class A2LocalGEM:
    def __init__(self,learner,band,record,error_record,*,score_threshold=None):
        self.score_threshold = score_threshold
        self.learner=learner;self.band=band;self.record=record;self.error_record=error_record
        self.reference=copy.deepcopy(learner.mac);self.target=copy.deepcopy(learner.target_mac)
        self.reference.agent.eval();self.target.agent.eval()
        self.params=[p for p in learner.mac.agent.parameters() if p.requires_grad]
        self.refparams=[p for p in self.reference.agent.parameters() if p.requires_grad]
        self.memories={};self.device_memories={};self.ids=[]
        self.indices,self.dim=indices_for(learner.mac.agent,[])
        self.t_env=0;self.phase=0;self.reference_rows=[];self.count=0;self.stats={}
        self.original_train=learner.train;self.original_step=learner.optimiser.step
        learner.train=self.train;learner.optimiser.step=self.step

    def update(self,task,selection,cpu_batch,t_env):
        self.flush();self.memories[task]=copy.deepcopy(cpu_batch)
        device=copy.deepcopy(cpu_batch);device.to(self.learner.args.device);self.device_memories[task]=device
        item = selection.get('A2', {})
        if self.score_threshold is None:
            activity = item.get('activity')
            if activity is None:
                activity = item.get('group_statistics', {}).get('overall', {}).get('activity')
            groups = partition(activity)
            defined = groups is not None
            if defined:
                self.ids = groups[self.band]
                self.indices, self.dim = indices_for(self.learner.mac.agent, self.ids)
            screening = dict(groups=groups, scheduling='latest boundary, no history union; undefined retains previous ids')
        else:
            selected = above_threshold(item, self.score_threshold)
            defined = selected is not None
            self.ids = selected if defined else []
            self.indices, self.dim = indices_for(self.learner.mac.agent, self.ids)
            screening = dict(score_threshold=self.score_threshold,
                             scheduling='latest boundary, no history union; undefined disables projection')
        return dict(source_task=task,t_env=t_env,band=self.band,defined=defined,
                    ids=self.ids,unique_parameters=int(self.indices.numel()),
                    memory_tasks=list(self.memories),memory_episodes={k:int(v.batch_size) for k,v in self.memories.items()},
                    **screening)

    def train(self,batch,t_env,episode_num,**kwargs):
        self.t_env=t_env;start=time.monotonic();self.reference_rows=[]
        if self.indices.numel():
            self.reference.agent.load_state_dict(self.learner.mac.agent.state_dict())
            self.target.agent.load_state_dict(self.learner.target_mac.agent.state_dict())
            for task,memory in self.device_memories.items():
                try:
                    loss=td_loss(self.reference,self.target,memory,self.learner.args)
                    grads=th.autograd.grad(loss,self.refparams,allow_unused=True)
                    full=th.cat([(th.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(self.refparams,grads)])
                    local=full[self.indices].detach().double().cpu().numpy()
                    self.reference_rows.append((task,local,float(loss.detach())))
                    del loss,grads,full
                except Exception:
                    self.error_record(dict(t_env=t_env,task=task,error=traceback.format_exc(),action='reference skipped this update; base learner continues'))
                finally:self.reference.hidden_states=None;self.target.hidden_states=None
        self.reference_seconds=time.monotonic()-start
        result=self.original_train(batch,t_env,episode_num,**kwargs)
        self.count+=1
        if self.count%1000==0:self.flush()
        return result

    @th.no_grad()
    def step(self,*args,**kwargs):
        before=th.cat([p.reshape(-1) for p in self.params])[self.indices].clone()
        result=self.original_step(*args,**kwargs)
        if not self.indices.numel():return result
        after=th.cat([p.reshape(-1) for p in self.params])[self.indices]
        d0=(after-before).double().cpu().numpy()
        rows=np.array([x[1] for x in self.reference_rows])
        fixed,info=project_increment(d0,rows)
        values=before+th.as_tensor(fixed,device=before.device,dtype=before.dtype)
        offset=local_offset=0
        for p in self.params:
            chosen=self.indices[(self.indices>=offset)&(self.indices<offset+p.numel())]-offset
            length=int(chosen.numel())
            if length:p.view(-1)[chosen]=values[local_offset:local_offset+length]
            local_offset+=length;offset+=p.numel()
        actual=(th.cat([p.reshape(-1) for p in self.params])[self.indices]-before).double().cpu().numpy()
        s=self.stats
        s['updates']=s.get('updates',0)+1;s['projections']=s.get('projections',0)+int(info['triggered'])
        s['fallbacks']=s.get('fallbacks',0)+int(info['fallback']);s['zero_or_invalid_reference_rows']=s.get('zero_or_invalid_reference_rows',0)+info['zero_rows']
        for key,value in [('raw_increment_norm',np.linalg.norm(d0)),('actual_increment_norm',np.linalg.norm(actual)),('correction_norm',np.linalg.norm(actual-d0)),('reference_seconds',self.reference_seconds)]:s[key+'_sum']=s.get(key+'_sum',0.)+float(value)
        for task,g,loss in self.reference_rows:
            k=s.setdefault('tasks',{}).setdefault(task,dict(updates=0,conflicts=0,td_loss_sum=0.,max_actual_local_dot=0.,max_normalized_actual_local_dot=0.,cosine_sum=0.))
            norm=float(np.linalg.norm(g));dot=float(g@d0);den=norm*float(np.linalg.norm(d0))
            k['updates']+=1;k['conflicts']+=int(dot>0);k['td_loss_sum']+=loss
            k['cosine_sum']+=dot/den if den>0 else 0.
            k['max_actual_local_dot']=max(k['max_actual_local_dot'],float(g@actual))
            denominator=norm*float(np.linalg.norm(actual))
            if denominator>0:k['max_normalized_actual_local_dot']=max(k['max_normalized_actual_local_dot'],float(g@actual)/denominator)
        return result

    def flush(self):
        if self.stats:self.record(dict(t_env=self.t_env,phase=self.phase,band=self.band,ids=self.ids,parameters=int(self.indices.numel()),**self.stats));self.stats={}

    def save(self,path):
        self.flush();th.save(dict(band=self.band,ids=self.ids,memories=self.memories,count=self.count,kind='local RMSprop increment projection; not complete trainer resume'),path)
