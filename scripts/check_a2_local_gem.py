"""Direct offline verification on existing episodes; zero SC2 interaction/training steps."""
import copy,json,os,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
os.environ['MARL_PROTOCOL']=str(ROOT/'configs/continual/gradient_constraint/gem/ts3_vdn_A2_local_gem_20261104.json')
sys.path.insert(0,str(ROOT/'drivers/continual'))
import ts3_train as t
from a2_local_gem import partition,indices_for,project_increment,td_loss,A2LocalGEM
from protection import rng_state,digest
import numpy as np
import torch as th
from scipy.optimize import minimize

def main():
    th.set_num_threads(1)
    source=next((ROOT/'runs/ts3_vdn_top3_Aprefix_seed1234_20261102_attempt001/detections').glob('A_*/screen_input.th'))
    payload=th.load(source,map_location='cpu');batch=payload['batch'];args=SimpleNamespace(**payload['args']);args.device='cpu';args.use_cuda=False
    class Log:
        def __init__(self):self.items={}
        def log_stat(self,k,v,*a,**kw):self.items[k]=float(v)
    def learner():
        th.manual_seed(321);log=Log();mac=t.base.controllers[args.mac](batch.scheme,batch.groups,args)
        result=t.base.learners[args.learner](mac,batch.scheme,log,args)
        for network in [mac.agent,result.target_mac.agent,result.mixer,result.target_mixer]:
            for module in network.modules():module._forward_hooks.clear()
        result.mac.agent.load_state_dict(payload['agent']);result.target_mac.agent.load_state_dict(payload['agent'])
        return result,log
    checks={};rng=np.random.RandomState(492)
    errors=[];residuals=[]
    for trial in range(12):
        G=rng.normal(size=(4,9));d=rng.normal(size=9)
        actual,info=project_increment(d,G)
        expected=minimize(lambda x:.5*np.sum((x-d)**2),np.zeros(9),jac=lambda x:x-d,constraints=[dict(type='ineq',fun=lambda x:-G@x,jac=lambda x:-G)],method='SLSQP',options=dict(ftol=1e-11,maxiter=200))
        assert expected.success
        errors.append(float(np.linalg.norm(actual-expected.x)));residuals.append(float(max(0,np.max(G@actual))))
    assert max(errors)<1e-6 and max(residuals)<1e-7
    checks['qp_vs_independent_slsqp']=dict(cases=12,max_vector_error=max(errors),max_constraint_residual=max(residuals))
    x,info=project_increment([1.,2.],[[1.,0.],[-1.,0.],[0.,0.],[1.,0.]])
    assert np.allclose(x,[0,2],atol=1e-8)
    checks['qp_dependent_opposite_zero_rows']=True
    groups=partition(list(range(64)))
    assert all(len(groups[x])==21 for x in ['high','middle','low'])
    assert not(set(groups['high'])&set(groups['middle']) or set(groups['high'])&set(groups['low']) or set(groups['middle'])&set(groups['low']))
    assert len(groups['excluded'])==1
    tie=partition([0.]*64);assert tie['high']==list(range(21)) and tie['excluded']==[42]
    checks['equal_groups_and_ties']=groups
    n,log=learner();exact=float(td_loss(n.mac,n.target_mac,batch,args).detach());n.mac.hidden_states=None;n.target_mac.hidden_states=None
    before={k:p.detach().clone() for k,p in n.mac.agent.named_parameters()}
    n.train(batch,2000100,0)
    assert abs(log.items['loss']-exact)<1e-5
    checks['td_loss_matches_unmodified_learner']=dict(reference=exact,unmodified=log.items['loss'])
    records=[];errors=[]
    for band in ['high','middle','low']:
        l,_=learner();saved=digest(rng_state());events=[]
        g=A2LocalGEM(l,band,events.append,errors.append)
        selection=dict(A2=dict(activity=list(range(64))))
        update=g.update('A',selection,batch,2000058)
        assert digest(rng_state())==saved
        assert len(g.ids)==21 and len(g.indices)==5397
        idx=g.indices.cpu().numpy();others=np.ones(g.dim,bool);others[idx]=False
        l.train(batch,2000100,0);g.flush()
        full=th.cat([p.detach().reshape(-1) for p in g.params]).numpy()
        ordinary=th.cat([p.detach().reshape(-1) for p in n.mac.agent.parameters() if p.requires_grad]).numpy()
        assert np.array_equal(full[others],ordinary[others])
        assert len(events)==1 and events[0]['updates']==1
        assert events[0]['tasks']['A']['max_actual_local_dot']<1e-5
        # Optimizer's accumulation is unchanged by actual-increment projection.
        a=l.optimiser.state_dict()['state'];b=n.optimiser.state_dict()['state']
        for key in a:
            for name in a[key]:
                if th.is_tensor(a[key][name]):assert th.equal(a[key][name],b[key][name])
                else:assert a[key][name]==b[key][name]
        records.append(dict(band=band,unique_parameters=len(idx),outside_selection_exactly_matches_baseline=True,optimizer_state_exactly_matches_baseline=True,audit=events[0]))
    assert not errors
    checks['real_rmsprop_local_slice_updates']=records
    checks.update(environment_training_steps=0,validation='passed',source=str(source))
    (ROOT/'results/ts3_vdn_A2_local_gem_20261104/implementation_validation.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(dict(validation='passed',environment_training_steps=0,real_bands=3,qp_cases=12,max_qp_error=checks['qp_vs_independent_slsqp']['max_vector_error'],loss=exact)))
if __name__=='__main__':main()
