"""Offline correctness checks on an existing real detection batch; no SC2/training."""
import copy, importlib.util, json, logging, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch as th
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'drivers/continual'))
os.environ['MARL_PROTOCOL']=str(ROOT/'configs/continual/neuron_protection/softprotect/location_AWV_switch_1111.json')
import train as t
import protection as p
from modules.mixers.qmix import QMixer

def main():
    th.set_num_threads(1)
    fixture=Path('/home/wyh/qmix_mixer_knife_fig6_latest_ramp/runs/qmix_task_switch_softprotect_seed1111_attempt001/detections/3000000/input.th')
    package=th.load(fixture,map_location='cpu')
    args=SimpleNamespace(**package['args']);args.device='cpu';args.use_cuda=False
    batch=package['batch'];batch.to('cpu')
    mac=t.controllers[args.mac](batch.scheme,batch.groups,args);mac.agent.load_state_dict(package['agent'])
    mixer=QMixer(args);mixer.load_state_dict(package['mixer'])
    config=t.PROTOCOL['method']
    before=p.digest([mac.agent.state_dict(),mixer.state_dict(),p.rng_state()])
    hidden=mac.hidden_states
    all_result=p.screen(mac,mixer,batch,config)
    after=p.digest([mac.agent.state_dict(),mixer.state_dict(),p.rng_state()])
    assert before==after and mac.hidden_states is hidden
    # Recompute shared-agent activity explicitly over state-agent pairs.
    captured=[]
    h=mac.agent.net[2].register_forward_hook(lambda m,i,o:captured.append(o.detach()))
    try:
        with th.no_grad():
            mac.init_hidden(batch.batch_size)
            qs=th.stack([mac.forward(batch,t=k) for k in range(batch.max_seq_length)],1)
    finally:
        h.remove();mac.hidden_states=hidden
    mask=batch['filled'][:,:-1].float().clone()
    mask[:,1:]*=1-batch['terminated'][:,:-2].float()
    lengths=batch['filled'].squeeze(-1).sum(1)-1
    good=mask.squeeze(-1).bool() & (th.arange(batch.max_seq_length-1)[None,:]<lengths[:,None])
    states=batch['state'][:,:-1][good]
    c,_=p.sensitivity(mixer,qs[:,:-1][good],batch['actions'][:,:-1].squeeze(-1)[good],batch['avail_actions'][:,:-1][good],states)
    z=th.stack([x.reshape(batch.batch_size,mac.n_agents,-1) for x in captured],1)[:,:-1][good].abs().numpy()
    expected=((c/c.mean())[:,None,None]*z.astype(np.float64)).sum(axis=(0,1))/(len(c)*mac.n_agents)
    assert np.allclose(all_result['layers']['A']['activity'],expected,rtol=2e-7,atol=1e-9)
    assert set(all_result['layers']['A']['selected'])==set(np.flatnonzero(expected/expected.mean()>3))
    # W/V computations must preserve the previous implementation exactly.
    spec=importlib.util.spec_from_file_location('old_protection',Path('/home/wyh/qmix_mixer_knife_fig6_latest_ramp/scripts/protection.py'))
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old_result=old.screen(mac,mixer,batch,dict(config,layers=['W','V']))
    assert all(all_result['layers'][l]==old_result['layers'][l] for l in ['W','V'])
    tests=[]
    for location,layers in [('A',['A']),('W',['W']),('V',['V']),('AWV',['A','W','V'])]:
        cfg=dict(config,layers=layers)
        result=p.screen(mac,mixer,batch,cfg)
        assert set(result['layers'])==set(layers)
        assert all(result['layers'][l]==all_result['layers'][l] for l in layers)
        agent=copy.deepcopy(mac.agent);mix=copy.deepcopy(mixer)
        original=SimpleNamespace(mac=SimpleNamespace(agent=agent),mixer=mix)
        original.optimiser=th.optim.RMSprop(list(agent.parameters())+list(mix.parameters()),lr=.0005,alpha=.99,eps=1e-5)
        protected=copy.deepcopy(original)
        protector=p.SoftProtection(protected,cfg,ROOT,lambda *a:None,lambda *a:None)
        protector.latest={l:[0,2] for l in layers};protector.enter('zealot',1,500000)
        checks=[]
        for step,factor in [(500000,.1),(550000,.55),(600000,1.)]:
            if step==550000:
                protector.latest={l:[] for l in layers};protector.enter('stalker',2,500000)
            protector.t_env=step
            op=list(original.mac.agent.parameters())+list(original.mixer.parameters())
            pp=list(protected.mac.agent.parameters())+list(protected.mixer.parameters())
            # Re-align parameter values to compare one optimizer increment at a time;
            # optimizer states remain their own, and receive identical gradients.
            with th.no_grad():
                for a,b in zip(op,pp):b.copy_(a)
            previous=[x.detach().clone() for x in op]
            for x in op+pp:
                if x.requires_grad:x.grad=th.full_like(x,.02)
            original.optimiser.step();protected.optimiser.step()
            maps={id(param):axis for _,_,param,axis in protector.params}
            max_error=0.
            for a,b,start in zip(op,pp,previous):
                expected_value=a.detach().clone()
                if id(b) in maps:
                    axis=maps[id(b)];mask=th.zeros_like(b,dtype=th.bool)
                    if axis==0:mask[[0,2]]=True
                    else:mask[:,[0,2]]=True
                    expected_value[mask]=start[mask]+factor*(a.detach()[mask]-start[mask])
                max_error=max(max_error,float((b.detach()-expected_value).abs().max()))
            assert max_error<2e-7
            assert p.digest(original.optimiser.state_dict())==p.digest(protected.optimiser.state_dict())
            checks.append(dict(t_env=step,expected_scale=factor,max_parameter_error=max_error,optimizer_state_equal=True))
        tests.append(dict(location=location,screen_layers=list(result['layers']),selected={l:result['layers'][l]['selected'] for l in layers},increment_checks=checks))
    report=dict(passed=True,fixture=str(fixture),episodes=batch.batch_size,valid_states=len(c),agent_activity='weighted mean over valid state-agent pairs',WV_identical_to_previous=True,screen_preserves_parameters_rng_hidden=True,tests=tests)
    dest=ROOT/'results/location_ablation_seed1111';dest.mkdir(parents=True,exist_ok=True)
    (dest/'implementation_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
