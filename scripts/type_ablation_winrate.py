"""Real SMAC rollouts for frozen matched-count ablations. No optimizer/training."""
import argparse,hashlib,json,os,random,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch as th
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
os.environ['SC2PATH']=str(ROOT/'runtime/StarCraftII')
from controllers.basic_controller import BasicMAC
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from smac.env import StarCraft2Env
SOURCE=ROOT/'results/2s3z_type_matched_ablation_20261101'
OUT=ROOT/'results/2s3z_type_ablation_winrate_20261101_attempt002'
ENV_SEEDS=[81001,81002,81003,81004]

def save(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)

def plan():
 OUT.mkdir(parents=True,exist_ok=False)
 tasks={};mapping=[];models={}
 for seed in [1111,2222,3535]:
  cases=json.loads((SOURCE/f'seed{seed}.json').read_text())
  run=ROOT/'runs'/f'qmix_2s3z_detect_seed{seed}_20261101_attempt003'
  step=json.loads((run/'training_complete.json').read_text())['t_env'];path=run/'detection'/str(step)/'input.th'
  models[str(seed)]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  for case in cases:
   for condition in case['conditions']:
    ids=condition['ids'];layer=case['layer'] if ids else None;recipient=condition['recipient'] if ids else None
    key=json.dumps([seed,layer,recipient,sorted(ids)])
    task_id=hashlib.sha256(key.encode()).hexdigest()[:16]
    tasks[task_id]=dict(task_id=task_id,seed=seed,layer=layer,recipient=recipient,ids=sorted(ids))
    mapping.append(dict(task_id=task_id,seed=seed,layer=case['layer'],source=case['source'],recipient=condition['recipient'],group=condition['group'],repeat=condition['repeat'],k=case['k'],n_masked=len(ids)))
 save(OUT/'plan.json',dict(models=models,tasks=list(tasks.values()),mapping=mapping,environment_seeds=ENV_SEEDS,episodes_per_seed=8,selection_source=str(SOURCE),mask_list_sha256={str(seed):hashlib.sha256((SOURCE/f'seed{seed}.json').read_bytes()).hexdigest() for seed in [1111,2222,3535]}))
 print('unique tasks',len(tasks),'episodes',len(tasks)*32,'mapped comparisons',len(mapping))

@th.no_grad()
def evaluate(task,package,output):
 a=SimpleNamespace(**package['args']);a.device='cpu';a.use_cuda=False
 batch=package['batch'];batch.to('cpu')
 mac=BasicMAC(batch.scheme,batch.groups,a);mac.agent.load_state_dict(package['agent']);mac.agent.eval()
 before=hashlib.sha256(b''.join(v.numpy().tobytes() for v in mac.agent.state_dict().values())).hexdigest()
 args=dict(a.env_args);args['map_name']='2s3z'
 schema={k:v for k,v in batch.scheme.items() if k not in ['filled','actions_onehot']};groups=batch.groups;records=[];handles=[];current_types=[]
 try:
  if task['ids']:
   module=mac.agent.net[2 if task['layer']=='A1' else 4]
   def mask(m,i,o):
    z=o.reshape(1,a.n_agents,-1).clone();affected=th.tensor([x==task['recipient'] for x in current_types])
    selected=th.zeros_like(z,dtype=th.bool);selected[:,:,task['ids']]=affected[None,:,None]
    return z.masked_fill(selected,0).reshape_as(o)
   handles.append(module.register_forward_hook(mask))
  for block,env_seed in enumerate(ENV_SEEDS):
   random.seed(env_seed);np.random.seed(env_seed);th.manual_seed(env_seed)
   args['seed']=env_seed;env=None
   try:
    env=StarCraft2Env(**args)
    for episode in range(8):
     env.reset();current_types[:]=['stalker' if env.agents[i].unit_type==env.stalker_id else 'zealot' if env.agents[i].unit_type==env.zealot_id else str(env.agents[i].unit_type) for i in range(a.n_agents)]
     # Actual environment schema, original observation/previous-action/ID inputs.
     trajectory=EpisodeBatch(schema,groups,1,env.episode_limit+1,preprocess={'actions':('actions_onehot',[OneHot(out_dim=a.n_actions)])},device='cpu')
     mac.init_hidden(1);terminated=False;t=0;total=0.;info={}
     while not terminated:
      trajectory.update(dict(state=[env.get_state()],obs=[env.get_obs()],avail_actions=[env.get_avail_actions()]),ts=t)
      q=mac.forward(trajectory,t)
      avail=trajectory['avail_actions'][:,t].bool();actions=q.masked_fill(~avail,-float('inf')).argmax(-1)
      reward,terminated,info=env.step(actions[0].numpy())
      trajectory.update(dict(actions=actions,reward=[(reward,)],terminated=[(terminated and not info.get('episode_limit',False),)]),ts=t)
      total+=float(reward);t+=1
     record=dict(task_id=task['task_id'],training_seed=task['seed'],environment_seed=env_seed,block=block,episode=episode,won=bool(info.get('battle_won',False)),return_value=total,length=t,types=current_types.copy(),info={k:v.item() if isinstance(v,np.generic) else v for k,v in info.items()})
     records.append(record)
     with (output/'episodes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   finally:
    if env is not None:env.close()
 finally:
  for h in handles:h.remove()
 after=hashlib.sha256(b''.join(v.numpy().tobytes() for v in mac.agent.state_dict().values())).hexdigest()
 save(output/'result.json',dict(task=task,episodes=len(records),wins=sum(x['won'] for x in records),win_rate=float(np.mean([x['won'] for x in records])),mean_return=float(np.mean([x['return_value'] for x in records])),parameters_unchanged=before==after,blocks=[dict(environment_seed=s,wins=sum(x['won'] for x in records if x['environment_seed']==s),episodes=8) for s in ENV_SEEDS]))

def worker(seed):
 th.set_num_threads(1);plan=json.loads((OUT/'plan.json').read_text());package=th.load(plan['models'][str(seed)]['path'],map_location='cpu')
 tasks=[x for x in plan['tasks'] if x['seed']==seed];status=dict(seed=seed,total=len(tasks),completed=0,failed=0,state='running')
 for task in tasks:
  output=OUT/'tasks'/task['task_id'];output.mkdir(parents=True,exist_ok=False)
  status['current']=task;save(OUT/f'status_seed{seed}.json',status)
  try:evaluate(task,package,output);status['completed']+=1
  except Exception:
   save(output/'error.json',dict(task=task,traceback=traceback.format_exc()));status['failed']+=1
  save(OUT/f'status_seed{seed}.json',status)
 status['state']='finished';save(OUT/f'status_seed{seed}.json',status)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--seed',type=int);a=p.parse_args()
 if a.plan:plan()
 else:worker(a.seed)
