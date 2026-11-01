"""Frozen final 2s3z models, disjoint episode screening/evaluation, matched masks."""
import csv,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch as th
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'drivers/continual')]
from controllers.basic_controller import BasicMAC
from modules.mixers.qmix import QMixer
from detection_common import sensitivity,parameter_hash
OUT=ROOT/'results/2s3z_type_matched_ablation_20261101'
TYPES=['stalker','zealot'];LAYERS=['A1','A2'];METRICS=['flip_rate','abs_margin_change','margin_erosion','legal_q_mae']

def valid(batch):
 lengths=batch['filled'].squeeze(-1).sum(1)-1
 return batch['filled'][:,:-1].squeeze(-1).bool() & (th.arange(batch.max_seq_length-1)[None,:]<lengths[:,None])

@th.no_grad()
def replay(mac,batch,layer=None,ids=(),recipient=None,types=None,capture=False):
 values={k:[] for k in LAYERS};handles=[]
 for k,module in [('A1',mac.agent.net[2]),('A2',mac.agent.net[4])]:
  if k==layer and len(ids):
   affected=th.tensor(types==recipient)
   def mask(m,i,o):
    z=o.reshape(batch.batch_size,mac.n_agents,-1).clone()
    selector=th.zeros_like(z,dtype=th.bool);selector[:,:,list(ids)]=affected[:,:,None]
    return z.masked_fill(selector,0).reshape_as(o)
   handles.append(module.register_forward_hook(mask))
  if capture:handles.append(module.register_forward_hook(lambda m,i,o,k=k:values[k].append(o.detach().clone())))
 try:
  mac.init_hidden(batch.batch_size)
  qs=th.stack([mac.forward(batch,t=t) for t in range(batch.max_seq_length)],1)
 finally:
  for h in handles:h.remove()
 if capture:values={k:th.stack([v.reshape(batch.batch_size,mac.n_agents,-1) for v in vs],1) for k,vs in values.items()}
 return qs,values

def activity(z,w,group):
 return (z.abs().numpy()*w[:,None,None]*group[:,:,None]).sum((0,1))/group.sum()

def scored(a):return a/a.mean() if a.mean()>0 else None

def measure(q0,q,batch,recipient,types):
 good=valid(batch);avail=batch['avail_actions'][:,:-1].bool()
 eligible=good[:,:,None] & th.tensor(types==recipient)[:,None,:] & (avail.sum(-1)>1)
 original=q0[:,:-1].masked_fill(~avail,-float('inf'));changed=q[:,:-1].masked_fill(~avail,-float('inf'))
 best=original.argmax(-1);newbest=changed.argmax(-1)
 other=avail.clone();other.scatter_(-1,best.unsqueeze(-1),False)
 gap0=q0[:,:-1].gather(-1,best.unsqueeze(-1)).squeeze(-1)-original.masked_fill(~other,-float('inf')).max(-1).values
 gap1=q[:,:-1].gather(-1,best.unsqueeze(-1)).squeeze(-1)-changed.masked_fill(~other,-float('inf')).max(-1).values
 mask=eligible.numpy();delta=(gap1-gap0).numpy()
 flip=(best!=newbest).numpy();qdiff=(q[:,:-1]-q0[:,:-1]).abs().numpy();legal=avail.numpy()
 def calc(ep):
  m=mask[ep];den=int(m.sum())
  if not den:return dict(samples=0)
  lm=m[:,:,:,None]&legal[ep]
  return dict(samples=den,flip_rate=float(flip[ep][m].mean()),abs_margin_change=float(np.abs(delta[ep][m]).mean()),margin_erosion=float(-delta[ep][m].mean()),legal_q_mae=float(qdiff[ep][lm].mean()),baseline_tie_fraction=float((np.abs(gap0.numpy()[ep][m])<=1e-8).mean()))
 return dict(pooled=calc(slice(None)),episodes=[calc(slice(i,i+1)) for i in range(batch.batch_size)])

@th.no_grad()
def main():
 th.set_num_threads(1);OUT.mkdir(exist_ok=True);rows=[];details=[];provenance=[]
 for seed in [1111,2222,3535]:
  run=ROOT/'runs'/f'qmix_2s3z_detect_seed{seed}_20261101_attempt003';step=json.loads((run/'training_complete.json').read_text())['t_env'];path=run/'detection'/str(step)/'input.th'
  p=th.load(path,map_location='cpu');a=SimpleNamespace(**p['args']);a.device='cpu';a.use_cuda=False
  batch=p['batch'];batch.to('cpu');select_ids=list(range(0,32,2));eval_ids=list(range(1,32,2));sb=batch[select_ids];eb=batch[eval_ids]
  st=np.array(p['types'])[select_ids];et=np.array(p['types'])[eval_ids]
  mac=BasicMAC(batch.scheme,batch.groups,a);mac.agent.load_state_dict(p['agent']);mac.agent.eval()
  mix=QMixer(a);mix.load_state_dict(p['mixer']);mix.eval();before=parameter_hash(mac,mix)
  sq,sz=replay(mac,sb,capture=True);eq,ez=replay(mac,eb,capture=True)
  good=valid(sb);c,_=sensitivity(mix,sq[:,:-1][good],sb['actions'][:,:-1].squeeze(-1)[good],sb['avail_actions'][:,:-1][good],sb['state'][:,:-1][good]);w=c/c.mean() if c.mean()>0 else None
  if w is None:raise ValueError('Undefined selection sensitivity; no forced selection')
  stflat=np.broadcast_to(st[:,None,:],(*good.shape,a.n_agents))[good.numpy()]
  egood=valid(eb);etflat=np.broadcast_to(et[:,None,:],(*egood.shape,a.n_agents))[egood.numpy()]
  for layer in LAYERS:
   z=sz[layer][:,:-1][good];ze=ez[layer][:,:-1][egood]
   overall=activity(z,w,np.ones(z.shape[:2],bool));overall_score=scored(overall)
   for source in TYPES:
    group=stflat==source;act=activity(z,w,group);s=scored(act);high=[] if s is None else np.flatnonzero(s>3).tolist();k=len(high)
    case=dict(seed=seed,step=step,layer=layer,source=source,k=k,selection_activity=act.tolist(),selection_score=None if s is None else s.tolist(),overall_score=overall_score.tolist(),conditions=[],skipped=k==0)
    if not k:details.append(case);continue
    rank=lambda v,n,low=False:sorted(range(len(v)),key=lambda j:((v[j] if low else -v[j]),j))[:n]
    global_high=rank(overall,k);local_low=rank(act,k,True);global_low=rank(overall,k,True)
    overlap=set(high)&set(global_high);local_unique=sorted(set(high)-set(global_high));global_unique=sorted(set(global_high)-set(high))
    case.update(type_high=high,global_high=global_high,overlap=sorted(overlap),jaccard=len(overlap)/len(set(high)|set(global_high)),type_unique=local_unique,global_unique=global_unique)
    selections=[('unmasked',[],0),('type_high',high,0),('overall_high',global_high,0),('type_low',local_low,0),('overall_low',global_low,0)]
    rng=np.random.default_rng(20261101+seed*100+LAYERS.index(layer)*10+TYPES.index(source))
    selections += [('random',sorted(rng.choice(64,k,replace=False).tolist()),rep) for rep in range(20)]
    if local_unique:selections += [('type_unique',local_unique,0),('overall_unique',global_unique,0)]
    # The same sampled set is used for both recipients. Unique groups are a
    # separately matched comparison, not compared as k-sized primary groups.
    for name,ids,rep in selections:
     for recipient in TYPES:
      changed,_=replay(mac,eb,layer,ids,recipient,et)
      metrics=measure(eq,changed,eb,recipient,et)
      eligible_z=z.abs().numpy()[stflat==recipient];eval_z=ze.abs().numpy()[etflat==recipient]
      zero_select=float((eligible_z[:,ids]==0).all(0).mean()) if ids else None
      zero_eval=float((eval_z[:,ids]==0).all(0).mean()) if ids else None
      item=dict(group=name,recipient=recipient,repeat=rep,ids=ids,n_masked=len(ids),zero_fraction_selection=zero_select,zero_fraction_evaluation=zero_eval,**metrics)
      case['conditions'].append(item)
      rows.append(dict(seed=seed,layer=layer,source=source,recipient=recipient,k=k,group=name,repeat=rep,n_masked=len(ids),zero_fraction_selection=zero_select,zero_fraction_evaluation=zero_eval,**metrics['pooled']))
      if name=='unmasked':assert all(metrics['pooled'][key]==0 for key in METRICS)
    details.append(case)
  unchanged=before==parameter_hash(mac,mix);assert unchanged
  provenance.append(dict(seed=seed,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),step=step,selection_episode_indices=select_ids,evaluation_episode_indices=eval_ids,parameters_unchanged=unchanged))
  (OUT/f'seed{seed}.json').write_text(json.dumps([x for x in details if x['seed']==seed],indent=2,allow_nan=False))
  print('finished seed',seed,flush=True)
 with (OUT/'metrics.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (OUT/'provenance.json').write_text(json.dumps(dict(inputs=provenance,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),random_repeats=20,environment_interactions=0,optimizer_updates=0),indent=2))
 print('completed',len(rows),'rows',flush=True)
if __name__=='__main__':main()
