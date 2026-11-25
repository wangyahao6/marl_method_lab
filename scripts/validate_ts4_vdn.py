"""Offline VDN mapping and reset structure check; no training rollout."""
import sys,copy,json,logging
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'drivers/continual'))
import ts4_train as t
from ts4_diagnostics import score,vdn_sensitivity
from types import SimpleNamespace
import torch as th
from protection import digest,rng_state
th.set_num_threads(1)
out=ROOT/'results/ts4_vdn_agent_4242_20261102';out.mkdir(parents=True,exist_ok=True)
data=th.load(ROOT/'runs/ts3_Aprefix_seed3535_20261102_attempt001/detections/A_1000013/screen_input.th',map_location='cpu')
args=SimpleNamespace(**data['args']);args.mixer='vdn';args.name='vdn';args.use_cuda=False;args.device='cpu';batch=data['batch']
mac=t.base.controllers[args.mac](batch.scheme,batch.groups,args);mac.agent.load_state_dict(data['agent'])
learner=t.base.learners[args.learner](mac,batch.scheme,t.base.Logger(logging.getLogger('offline')),args)
learner.remove_activation_hooks()
before=digest([mac.agent.state_dict(),rng_state()]);result=score(learner,args,batch)
unchanged=before==digest([mac.agent.state_dict(),rng_state()])
q=th.randn(17,5,11);avail=th.rand(17,5,11)>.4;actions=th.randint(0,11,(17,5));avail.scatter_(2,actions[:,:,None],True)
c=vdn_sensitivity(q,actions,avail);chosen=q.gather(2,actions[:,:,None]).squeeze(-1);brute=[]
for i in range(5):
    diffs=[]
    for a in range(11):
        alt=chosen.clone();alt[:,i]=q[:,i,a]
        diffs.append((alt.sum(-1)-chosen.sum(-1)).abs()*avail[:,i,a]*(actions[:,i]!=a))
    brute.append(th.stack(diffs).max(0).values)
error=float((c-th.stack(brute,1).mean(1)).abs().max());checks=[]
for layer in ['A1','A2']:
    clone=copy.deepcopy(learner);prot=t.TS3Protection(clone,layer,'linear',lambda x:None)
    prot.update({layer:dict(sets={'overall':[0,7]})},0)
    for step in [0,50000,100000]:
        if step==0:prot.update({layer:dict(sets={'overall':[]})},0)
        prot.t_env=step;scales=prot.scales(step);old={k:v.clone() for k,v in clone.mac.agent.named_parameters()};reference=copy.deepcopy(clone.mac.agent)
        opt=th.optim.RMSprop(reference.parameters(),lr=args.lr,alpha=args.optim_alpha,eps=args.optim_eps);opt.load_state_dict(copy.deepcopy(clone.optimiser.state_dict()))
        for p in clone.mac.agent.parameters():p.grad=th.ones_like(p)
        for p in reference.parameters():p.grad=th.ones_like(p)
        opt.step();clone.optimiser.step();params=dict(clone.mac.agent.named_parameters());refs=dict(reference.named_parameters())
        worst=max(float((params[k]-(v+(refs[k]-v)*scales.get(k,1))).abs().max()) for k,v in old.items())
        checks.append(dict(layer=layer,t_env=step,max_update_error=worst))
cfg=t.h.config(4242,out);cfg['env_args']['capability_config']['team_gen']['weights']=[1.,0.,0.]
env=t.TS4Protoss(**cfg['env_args']);audits=[];env.audit=audits.append
try:
    for unit in ['stalker','zealot','colossus']:env.set_task(unit);env.reset()
finally:env.close()
shape_match=len({(a['obs_shape'],a['state_shape']) for a in audits})==1
# Allied SMAC custom IDs and enemy Blizzard IDs are different by design.
pure_each_side=all(len(set(a['ally_types']))==len(set(a['enemy_types']))==1 for a in audits)
report=dict(score_model_rng_unchanged=unchanged,vdn_sensitivity_enumeration_error=error,offline_update_checks=checks,score_counts={k:v['count'] for k,v in result.items()},environment_reset_only=audits,shape_match=shape_match,pure_each_side=pure_each_side,training_validation_performed=False,initial_check_correction='Allied custom type IDs differ from enemy Blizzard IDs; compare purity separately per side.')
t.write(out/'implementation_validation.json',report)
print(json.dumps(report))
assert unchanged and error<1e-5 and all(x['max_update_error']<1e-6 for x in checks) and shape_match and pure_each_side
