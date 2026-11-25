"""Offline slices/RMSprop/release and restorable training-state checks."""
import copy,json,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
os.environ['MARL_PROTOCOL']=str(ROOT/'configs/continual/neuron_protection/softprotect/hetero_agent_softprotect_ABA.json');sys.path.insert(0,str(ROOT/'drivers/continual'))
import hetero_train as h
from hetero_protection import AgentProtection
from protection import digest,rng_state
import torch as th
th.set_num_threads(1)
def main():
 package=th.load(ROOT/'runs/qmix_2s3z_detect_seed1111_20261101_attempt003/detection/2000016/input.th',map_location='cpu');batch=package['batch'];a=SimpleNamespace(**package['args']);a.device='cpu';a.use_cuda=False
 mac=h.base.controllers[a.mac](batch.scheme,batch.groups,a)
 selection={'A1':{'T':[0,2],'G':[1,3],'R':[4,5]},'A2':{'T':[1,3,4],'G':[0,2,5],'R':[6,7,8]}}
 results=[]
 for group in ['T1','G1','R1','T2','G2','R2','T12','G12','R12','N']:
  agent=copy.deepcopy(mac.agent);reference=copy.deepcopy(agent)
  op=th.optim.RMSprop(agent.parameters(),lr=.0005,alpha=.99,eps=1e-5);refop=th.optim.RMSprop(reference.parameters(),lr=.0005,alpha=.99,eps=1e-5)
  l=SimpleNamespace(mac=SimpleNamespace(agent=agent),optimiser=op);protector=AgentProtection(l,selection,group,lambda x:None);protector.active=True
  k1=2 if '1' in group else 0;k2=3 if '2' in group else 0;D=agent.net[0].in_features
  expected=k1*(D+1+64)+257*k2-k1*k2;assert protector.unique_parameters==expected
  for p,q in zip(agent.parameters(),reference.parameters()):
   if p.requires_grad:p.grad=th.full_like(p,.02);q.grad=th.full_like(q,.02)
  old={n:p.detach().clone() for n,p in agent.named_parameters()}
  op.step();refop.step()
  masks={name:mask for name,p,mask in protector.entries}
  for name,p in agent.named_parameters():
   raw=dict(reference.named_parameters())[name];expected_value=raw.detach().clone()
   if name in masks:expected_value[masks[name]]=old[name][masks[name]]+.1*(raw.detach()[masks[name]]-old[name][masks[name]])
   assert th.allclose(p,expected_value,rtol=0,atol=1e-7)
  assert digest(op.state_dict())==digest(refop.state_dict())
  # Align values, preserve optimizer moments; immediate all-one A2 release.
  with th.no_grad():
   for p,q in zip(agent.parameters(),reference.parameters()):p.copy_(q)
  protector.active=False;op.step();refop.step()
  assert all(th.equal(p,q) for p,q in zip(agent.parameters(),reference.parameters()))
  results.append(dict(group=group,mask_elements=expected,actual_increment_scaling=True,immediate_release=True))
 # Serialize real module/target/optimizer/replay/RNG fields, recover and compare.
 clean={k:v for k,v in batch.scheme.items() if k not in ['filled','actions_onehot']}
 buffer=h.base.ReplayBuffer(clean,batch.groups,64,batch.max_seq_length,preprocess={'actions':('actions_onehot',[h.base.OneHot(a.n_actions)])});buffer.insert_episode_batch(batch)
 learner=h.base.learners[a.learner](mac,buffer.scheme,h.base.Logger(__import__('logging').getLogger('offline')),a)
 learner.remove_activation_hooks()
 learner.train(batch,100,0)
 state=dict(agent=mac.agent.state_dict(),target=learner.target_mac.agent.state_dict(),mixer=learner.mixer.state_dict(),target_mixer=learner.target_mixer.state_dict(),optimizer=learner.optimiser.state_dict(),buffer=buffer,rng=rng_state(),target_counter=learner.last_target_update_episode,episode=32,updates=1,t_env=100)
 with tempfile.TemporaryDirectory() as d:
  f=Path(d)/'state.th';th.save(state,f);loaded=th.load(f,map_location='cpu')
  assert digest(state)==digest(loaded)
  h.restore_rng(state['rng']);sample1=buffer.sample(8);h.restore_rng(loaded['rng']);sample2=loaded['buffer'].sample(8)
  assert digest(sample1.data.transition_data)==digest(sample2.data.transition_data)
 out=ROOT/'results/hetero_agent_softprotect_3535_20261101';out.mkdir(exist_ok=True,parents=True);(out/'offline_validation.json').write_text(json.dumps(dict(protection=results,snapshot_roundtrip=True,replay_sampling_restored=True,training_environment_steps=0),indent=2));print('all offline slice/update/release/state checks passed')
if __name__=='__main__':main()
