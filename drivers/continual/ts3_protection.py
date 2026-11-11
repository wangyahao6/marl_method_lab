"""Single Agent layer: latest-selection ramp versus permanent historical union."""
import torch as th
from release_schedule import ReleaseSchedule

class TS3Protection:
    def __init__(self, learner, layer, mode, record, *, protect_output=True):
        self.learner=learner; self.layer=layer; self.mode=mode; self.record=record
        self.schedule=ReleaseSchedule({layer:learner.args.rnn_hidden_dim}, .1, 100000) if layer else None
        self.protect_output = protect_output
        self.t_env=0; self.count=0
        self.original=learner.optimiser.step
        if layer: learner.optimiser.step=self.step

    def update(self, selection, t_env):
        if not self.layer: return {'mode':'baseline','t_env':t_env}
        item=selection[self.layer]; chosen=item.get('sets',{}).get('overall')
        if chosen is not None:
            if self.mode in ('union', 'chi_union'): chosen=sorted(set(chosen)|self.schedule.protected[self.layer])
            self.schedule.replace({self.layer:chosen},t_env)
        return dict(self.schedule.state(t_env),mode=self.mode,layer=self.layer,
                    detection_defined=item.get('defined', chosen is not None) if self.mode in ('chi', 'chi_union') else chosen is not None,
                    latest_selected=item.get('sets',{}).get('overall'))

    def scales(self, t_env):
        agent=self.learner.mac.agent; result={}
        if not self.layer:return result
        v=th.tensor(self.schedule.values(t_env)[self.layer],device=agent.net[0].weight.device)
        if self.layer=='A':
            result={'net.0.weight':v[:,None].expand_as(agent.net[0].weight),
                    'net.0.bias':v,
                    'rnn.weight_ih':v[None,:].expand_as(agent.rnn.weight_ih)}
        elif self.layer=='A1':
            result={'net.0.weight':v[:,None].expand_as(agent.net[0].weight),
                    'net.0.bias':v,'net.3.weight':v[None,:].expand_as(agent.net[3].weight)}
        else:
            result={'net.3.weight':v[:,None].expand_as(agent.net[3].weight),
                    'net.3.bias':v,'rnn.weight_ih':v[None,:].expand_as(agent.rnn.weight_ih)}
        if not self.protect_output:
            del result['net.3.weight' if self.layer == 'A1' else 'rnn.weight_ih']
        return result

    @th.no_grad()
    def step(self,*args,**kwargs):
        scales=self.scales(self.t_env); params=dict(self.learner.mac.agent.named_parameters())
        entries=[(name,params[name],v,v<1) for name,v in scales.items() if (v<1).any()]
        old=[p[mask].clone() for _,p,v,mask in entries]
        result=self.original(*args,**kwargs); audit=[]
        for (name,p,v,mask),before in zip(entries,old):
            delta=p[mask]-before; p[mask]=before+v[mask]*delta
            if self.count%1000==0:
                audit.append(dict(parameter=name,elements=int(mask.sum()),scale_min=float(v[mask].min()),scale_max=float(v[mask].max()),raw_norm=float(delta.norm()),actual_norm=float((p[mask]-before).norm())))
        if audit:self.record(dict(t_env=self.t_env,layer=self.layer,mode=self.mode,parameters=audit,schedule=self.schedule.state(self.t_env)))
        self.count+=1
        return result
