"""Necessary offline actual-step/schedule checks and exact-task shape inspection; no training."""
import copy,json,os,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];os.environ['MARL_PROTOCOL']=str(ROOT/'configs/continual/neuron_protection/softprotect/ts3_agent_3535.json');sys.path.insert(0,str(ROOT/'drivers/continual'))
import ts3_train as t
from ts3_protection import TS3Protection
from protection import digest
import torch as th
OUT=ROOT/'results/ts3_agent_3535_20261102'
def selected(ids):return {k:dict(sets={'overall':ids}) for k in ['A1','A2']}
def main():
    th.set_num_threads(1);package=th.load(ROOT/'runs/qmix_2s3z_detect_seed1111_20261101_attempt003/detection/2000016/input.th',map_location='cpu');batch=package['batch'];a=SimpleNamespace(**package['args']);a.device='cpu';a.use_cuda=False
    mac=t.base.controllers[a.mac](batch.scheme,batch.groups,a);results=[]
    for layer in ['A1','A2']:
        for mode in ['linear','union']:
            agent=copy.deepcopy(mac.agent);reference=copy.deepcopy(agent);op=th.optim.RMSprop(agent.parameters(),lr=.0005,alpha=.99,eps=1e-5);refop=th.optim.RMSprop(reference.parameters(),lr=.0005,alpha=.99,eps=1e-5)
            learner=SimpleNamespace(mac=SimpleNamespace(agent=agent),optimiser=op);p=TS3Protection(learner,layer,mode,lambda x:None)
            p.update(selected([0,2]),1000000);p.update(selected([2,3]),2000000);p.t_env=2050000
            values=p.schedule.values(p.t_env)[layer]
            assert abs(values[0]-(.55 if mode=='linear' else .1))<1e-12
            assert values[2]==.1 and values[3]==.1 and values[1]==1
            masks=p.scales(p.t_env);expected_count=3*(agent.net[0].in_features+1+64) if layer=='A1' else 3*257
            assert sum(int((x<1).sum()) for x in masks.values())==expected_count
            for x,y in zip(agent.parameters(),reference.parameters()):
                if x.requires_grad:x.grad=th.full_like(x,.02);y.grad=th.full_like(y,.02)
            before={k:x.detach().clone() for k,x in agent.named_parameters()};op.step();refop.step()
            for name,x in agent.named_parameters():
                y=dict(reference.named_parameters())[name];scale=masks.get(name,th.ones_like(x));expected=before[name]+scale*(y.detach()-before[name])
                assert th.allclose(x,expected,rtol=0,atol=1e-7)
            assert digest(op.state_dict())==digest(refop.state_dict())
            p.update(selected([]),2100000);p.update(selected([]),2150000)
            assert p.schedule.values(2150000)[layer][2]==(.55 if mode=='linear' else .1)
            p.update(selected([0]),2150000);assert p.schedule.values(2150000)[layer][0]==.1
            p.update({k:dict(sets={'overall':None}) for k in ['A1','A2']},2150000);assert p.schedule.values(2150000)[layer][0]==.1
            results.append(dict(layer=layer,mode=mode,actual_RMSprop_scaling=True,optimizer_moments_unchanged=True,repeat_and_reentry_and_empty_and_undefined=True,unique_scaled_parameters=expected_count))
    cfg=t.h.config(3535,OUT);env=t.h.ExactProtoss(**cfg['env_args']);structures=[]
    try:
        for task in 'ABCDE':
            env.set_task(task);env.reset();raw=env.wrapped.env
            structures.append(dict(task=task,requested=t.P['tasks'][task],ally=[int(u.unit_type) for u in raw.agents.values()],enemy=[int(u.unit_type) for u in raw.enemies.values()],obs=env.get_obs_size(),state=env.get_state_size(),actions=env.get_total_actions(),reward_denominator=float(raw.max_reward)))
    finally:env.close()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'implementation_validation.json').write_text(json.dumps(dict(offline=results,structures=structures,training_steps=0),indent=2));print(json.dumps(dict(offline_checks=results,structures=structures,training_steps=0)))
if __name__=='__main__':main()
