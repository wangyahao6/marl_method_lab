# -*- coding: utf-8 -*-
"""Offline TS5 analysis. No environment, optimizer, rollout or input writes.
Run: PYTHONPATH=src:. .venv/bin/python docs/results/comparisons/ts5-tkpr-refresh/analysis/source_drift/analyze_source_drift.py --repo .
"""
import argparse,csv,json,hashlib,itertools,math,sys,subprocess
from pathlib import Path
from collections import Counter,defaultdict
from types import SimpleNamespace
import numpy as np
import torch as th

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=Path('.'));a=ap.parse_args();root=a.repo.resolve();sys.path[:0]=[str(root/'src'),str(root)]
 from controllers import REGISTRY
 from components.episode_buffer import ReplayBuffer
 from drivers.continual.common.runtime import episode_spec
 from learners.action_set_preference import batch_episodes,decision_masks,unroll
 from learners.ranking_preference import topk_preference,topk_statistics,reference_scale,reference_order,full_rank_pairs,_topk_actions
 out=root/'docs/results/comparisons/ts5-tkpr-refresh/analysis/source_drift';out.mkdir(parents=True,exist_ok=True)
 th.set_num_threads(2);th.manual_seed(68121);rng=np.random.default_rng(68121)
 def js(p):return json.loads(p.read_text())
 def dump(n,x):(out/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 def csvsave(n,rows,fields=None):
  keys=fields or list(dict.fromkeys(k for r in rows for k in r))
  with (out/n).open('w') as f:
   w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
 def sha(p):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  return h.hexdigest()
 runs={};manifest=[];stats=[];audit=[];qchecks=[];distributions={};source_lines={}
 for method in ('tkpr_k1','tkpr_refresh'):
  p=root/f'runs/ts5_3535_{method}_20261122_attempt001';cfg=js(p/'config.json');cfg.update(device='cpu',use_cuda=False)
  info=js(p/'env_info.json');scheme,groups,pre=episode_spec(info);scheme=ReplayBuffer(scheme,groups,1,2,preprocess=pre).scheme
  args=SimpleNamespace(**cfg);mac=REGISTRY[args.mac](scheme,groups,args);mac.agent.eval();mac.init_hidden(1)
  settings=cfg['topk_preference' if method=='tkpr_k1' else 'tkpr_refresh']
  models={int(f.parent.name[5:9]):f for f in p.glob('models/*/agent.th')};pools={}
  for f in sorted(p.glob('action_set_memory/*/success.th')):
   pool=th.load(f,map_location='cpu');task=pool['task'];pools[task]=pool
   grouped=defaultdict(list)
   for ep in pool['episodes']:
    if method=='tkpr_refresh':assert 'reference_phase' in ep and 'reference_checkpoint' in ep;src=ep['reference_phase']
    else:src=pool['phase'] # static K1 collection source established by code, not Refresh fallback
    d=ep['data'];length=ep['length']+1
    for k in ('obs','actions_onehot','avail_actions','filled','terminated','old_q_values'):assert len(d[k])==length
    assert d['old_q_values'].shape==d['avail_actions'].shape and th.isfinite(d['old_q_values']).all()
    grouped[src].append(ep)
   for src,eps in sorted(grouped.items()):
    b,_=batch_episodes(eps,scheme,groups,'cpu',include_good=False,include_old_q=True);legal,dec=decision_masks(b);dec=dec.squeeze(-1) if dec.shape[-1]==1 else dec
    dec=dec & (legal.sum(-1)>=2)
    obs=b['obs'][:,:-1][dec].numpy();acts=b['actions_onehot'][:,:-1].argmax(-1)[dec].numpy();lengths=[e['length'] for e in eps]
    counts=np.bincount(acts,minlength=11);cov=len({hashlib.sha256(np.round(v,3).tobytes()).digest() for v in obs})
    stats.append(dict(method=method,task=task,pool_creation_phase=pool['phase'],reference_phase=src,trajectories=len(eps),length_mean=np.mean(lengths),length_std=np.std(lengths,ddof=1) if len(eps)>1 else 0,length_min=min(lengths),length_max=max(lengths),decision_states=int(dec.sum()),unique_rounded_obs=cov,coverage_fraction=cov/max(len(obs),1),obs_rms=float(np.sqrt(np.mean(obs**2))),action_counts=json.dumps(counts.tolist()),legal_action_mean=float(legal.sum(-1)[dec].float().mean()),bank='final_E_postrefresh' if method=='tkpr_refresh' else 'static_original'))
    distributions[method,task,src]=counts/counts.sum()
    # One deterministic complete episode per source: verify reference Q from explicit source checkpoint.
    ep=eps[0];mac.agent.load_state_dict(th.load(models[src],map_location='cpu'))
    b,_=batch_episodes([ep],scheme,groups,'cpu',include_good=False,include_old_q=True)
    with th.no_grad():q=unroll(mac,b,b.max_seq_length)
    err=float((q-b['old_q_values']).abs().max());qchecks.append(dict(method=method,task=task,reference_phase=src,episodes_checked=1,max_abs_error=err,pass_tolerance_2e_5=err<=2e-5))
    print('Q_CHECK',qchecks[-1],flush=True)
    # Reference-Q numeric error is reported, never used to alter stored targets.
   manifest.append(dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=sha(f)))
  for f in list(models.values())+[p/'config.json',p/'protocol.json',p/'retention.json']:
   manifest.append(dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=sha(f)))
  runs[method]=dict(p=p,cfg=cfg,mac=mac,models=models,pools=pools,scheme=scheme,groups=groups,settings=settings)
  audit.append(dict(method=method,checkpoints=sorted(models),pools={k:len(v['episodes']) for k,v in pools.items()},intermediate_checkpoints=[],td_replay_snapshots=[]))
 csvsave('trajectory_source_stats.csv',stats);csvsave('stored_q_checks.csv',qchecks)
 dst=[]
 for (method,task,src),prob in distributions.items():
  for (m,t,s),other in distributions.items():
   if method==m and task==t and src<s:
    mid=(prob+other)/2;mask=prob>0;mask2=other>0
    val=.5*np.sum(prob[mask]*np.log2(prob[mask]/mid[mask]))+.5*np.sum(other[mask2]*np.log2(other[mask2]/mid[mask2]))
    dst.append(dict(method=method,task=task,source_a=src,source_b=s,action_js_divergence_bits=val))
 csvsave('source_distribution_distances.csv',dst)
 dump('input_manifest.json',dict(analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),seed=68121,repeats=10,files=manifest,audit=audit))
 (out/'data_audit.md').write_text('''# 数据审计（先于数值实验）

审计对象：TS5 seed3535 TKPR-K1 / TKPR-Refresh，训练版本 9db0ed1951ded62c00eb1ff27a80c71ae5465dc5。

- `raw_logs/` 在当前公开树不存在；原始 updates/experiment_metrics JSONL 在运行目录可读，压缩归档路径见上级 input_manifest.json，不能把公开路径缺失解释为日志整体缺失。
- 两方法均有 A–E 五个阶段末 agent checkpoint；未发现 D/E 25%、50%、75% 模型。
- K1 原池每阶段独立保存，源码确认引用当阶段模型。Refresh 每任务仅一个原地覆盖的 success.th，现存是 E 末刷新后的池；未发现刷新前后版本快照。池创建 phase 与 reference_phase 不同。
- Refresh 每条轨迹明确携带 reference_phase/reference_checkpoint/old_q_values；完整 obs、state、actions、actions_onehot、avail_actions、reward、filled、terminated 与 bootstrap 时间轴可读。字段和时间维度检查已执行。
- 来源只读取 Refresh 逐轨迹标记；K1 由静态采集代码及池级 checkpoint 确认，未按索引推断。每来源抽查一个完整序列，重新展开其来源 checkpoint，Q 误差见 stored_q_checks.csv。
- 刷新丢弃轨迹不可恢复，episode_ids 不具备跨刷新稳定身份。**D/E 实际训练池及自然来源分布无法重建。**日志只提供采样索引，无法恢复被覆盖的轨迹/Q。
- 未发现当前任务 TD ReplayBuffer、同次更新 minibatch 或含参数位置的训练快照。成功轨迹不能替代当前 TD 数据。实验3跳过，source_td_gradient.csv 明确记录不可用原因。

可执行范围：固定留存轨迹上的不同 checkpoint 排序；C/D/E 阶段末模型位置的留存旧来源梯度；原始日志 TD Loss。梯度样本仅允许 reference_phase < 当前阶段，且排除当前及未来任务；来自最终池的留存子集有幸存偏差，**不是历史真实池**。natural_survivors 按该子集实际轨迹比例每任务采8条（K1跨任务总8条）；balanced_decisions 每来源采8条完整序列，再随机屏蔽多余有效决策以等量，所有 RNN 输入序列完整。后者是条件化对照，不是原训练采样。每设置10次。

源码依据：memory/__init__.py 的 sample/sample_per_task/refresh；action_set_train.py 的 reference_record/collect_memory 与阶段末刷新；ranking_preference.py 的 topk_preference/_full_rank_loss；action_set_preference.py 的 decision_masks/unroll/reduce_task_losses；q_learner.py 的 agent_gradient_metrics。当前 TKPR 函数与训练版本相同（其他方法新增不影响其路径）。不启动环境，不调用 optimizer.step，不修改原始产物。
''')
 print('audit complete',flush=True)
 # Fixed common sequence panel. Every policy sees exactly the same episodes, own hidden reconstruction.
 ranks=[];panel=[]
 for task in 'ABC':
  eps=[]
  for method,r in runs.items():
   grouped=defaultdict(list)
   for i,e in enumerate(r['pools'][task]['episodes']):grouped[e['reference_phase'] if method=='tkpr_refresh' else r['pools'][task]['phase']].append((i,e))
   for src,items in sorted(grouped.items()):
    ids=rng.choice(len(items),min(4,len(items)),replace=False)
    for j in ids:
     idx,e=items[j];eps.append(e);panel.append(dict(task=task,bank_method=method,reference_phase=src,episode_index_within_final_file=idx,sequence_sha256=hashlib.sha256(e['data']['obs'].numpy().tobytes()+e['data']['actions_onehot'].numpy().tobytes()).hexdigest()))
  for method,r in runs.items():
   mac=r['mac'];b,_=batch_episodes(eps,r['scheme'],r['groups'],'cpu',include_good=False,include_old_q=True);legal,dec=decision_masks(b);dec=dec & (legal.sum(-1)>=2)
   qs={}
   for phase in range(4):
    mac.agent.load_state_dict(th.load(r['models'][phase],map_location='cpu'))
    with th.no_grad():qs[phase]=unroll(mac,b,b.max_seq_length-1)[dec].double()
   lm=legal[dec];upper=th.ones(11,11,dtype=th.bool).triu(1);pairs=lm[:,:,None]&lm[:,None,:]&upper
   for x,y in itertools.combinations(range(4),2):
    q1,q2=qs[x],qs[y];s1=(q1[:,:,None]-q1[:,None,:]).sign();s2=(q2[:,:,None]-q2[:,None,:]).sign();conc=((s1*s2)>0)&pairs;disc=((s1*s2)<0)&pairs
    # tau-b per decision; exact Q ties retained, undefined all-tie rows excluded.
    c=conc.sum((-1,-2));d=disc.sum((-1,-2));nt1=((s1!=0)&pairs).sum((-1,-2));nt2=((s2!=0)&pairs).sum((-1,-2));den=(nt1*nt2).double().sqrt();valid=den>0
    top1=q1.masked_fill(~lm,-th.inf).argmax(-1);top2=q2.masked_fill(~lm,-th.inf).argmax(-1)
    def gaps(q):
     _,_,sc=reference_scale(q,lm,.1);vals=q.masked_fill(~lm,-th.inf).topk(2,dim=-1).values;return (vals[:,0]-vals[:,1])/sc.squeeze(-1)
    g1,g2=gaps(q1),gaps(q2);_,_,scale=reference_scale(q1,lm,.1)
    regret=(q1.gather(-1,top1[:,None])-q1.gather(-1,top2[:,None]))/scale
    ranks.append(dict(model_method=method,task=task,checkpoint_a=x,checkpoint_b=y,sequences=len(eps),decisions=int(dec.sum()),top1_agreement=float((top1==top2).double().mean()),kendall_tau_b=float(((c-d)[valid]/den[valid]).mean()),tau_defined=int(valid.sum()),pairwise_ranking_agreement=float(((s1==s2)&pairs).sum()/pairs.sum()),normalized_top1_gap_a=float(g1.mean()),normalized_top1_gap_b=float(g2.mean()),normalized_gap_abs_change=float((g1-g2).abs().mean()),old_model_regret_of_new_top1=float(regret.mean())))
 csvsave('reference_rank_drift.csv',ranks);csvsave('fixed_panel.csv',panel)
 print('rank complete',flush=True)
 # Gradients computed from one common full-batch graph at one parameter position.
 metrics=[];cosines=[];checks=[]
 def vector(loss,params):
  grad=th.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
  return th.cat([(g if g is not None else th.zeros_like(p)).detach().reshape(-1).double() for p,g in zip(params,grad)])
 def cos(x,y):
  den=float(x.norm()*y.norm());return float(x.dot(y)/den) if den else None
 for method,r in runs.items():
  for phase in (2,3,4):
   mac=r['mac'];mac.agent.load_state_dict(th.load(r['models'][phase],map_location='cpu'));params=tuple(p for p in mac.agent.parameters() if p.requires_grad)
   for mode in ('natural_survivors','balanced_decisions'):
    for repeat in range(10):
     order=rng.permutation(phase).tolist()
     slots=Counter(order[i%phase] for i in range(8)) if method=='tkpr_k1' else {i:8 for i in range(phase)}
     for ti,task in enumerate('ABCDE'[:phase]):
      pool=r['pools'][task];allowed=[e for e in pool['episodes'] if method=='tkpr_k1' or e['reference_phase']<phase]
      def src(e):return e['reference_phase'] if method=='tkpr_refresh' else pool['phase']
      groups=defaultdict(list)
      for e in allowed:groups[src(e)].append(e)
      if mode=='natural_survivors':
       ids=rng.choice(len(allowed),slots[ti],replace=len(allowed)<slots[ti]);eps=[allowed[i] for i in ids]
      else:
       eps=[seqs[i] for source,seqs in sorted(groups.items()) for i in rng.choice(len(seqs),8,replace=len(seqs)<8)]
      b,_=batch_episodes(eps,r['scheme'],r['groups'],'cpu',include_good=False,include_old_q=True);legal,dec=decision_masks(b);dec=dec & (legal.sum(-1)>=2);old=b['old_q_values'][:,:-1]
      with th.no_grad():
       norm,_,_,_=reference_order(old,legal,r['settings']['q_scale_floor']);_,_,_,pairs=full_rank_pairs(norm,legal,dec,r['settings']);top=_topk_actions(old,legal,(legal.sum(-1)-1).clamp_min(0).clamp_max(1));pairs=pairs&(top.unsqueeze(-1)|top.unsqueeze(-2));valid=pairs.any(-1).any(-1)
      labels=th.tensor([src(e) for e in eps]);masks={s:(labels==s)[:,None,None].expand_as(dec) for s in groups}
      if mode=='balanced_decisions':
       target=min(int((valid&m).sum()) for m in masks.values());selected=th.zeros_like(dec)
       for s,m in masks.items():
        inds=(valid&m).reshape(-1).nonzero().flatten();chosen=inds[th.from_numpy(rng.choice(len(inds),target,replace=False))];selected.reshape(-1)[chosen]=True
       dec=dec&selected
      current=unroll(mac,b,b.max_seq_length-1)
      full,count,_=topk_preference(current,old,legal,dec,r['settings']);gfull=vector(full,params);gs={};combined=th.zeros_like(gfull);N=int(count)
      temp=[]
      for s,m in masks.items():
       loss,c,summary=topk_preference(current,old,legal,dec&m,r['settings']);n=int(c);g=vector(loss,params);weight=n/max(N,1);contrib=g*weight;combined+=contrib;gs[s]=g
       info=topk_statistics(summary.detach().tolist())
       temp.append(dict(method=method,checkpoint_phase=phase,task=task,mode=mode,repeat=repeat,reference_phase=s,source_available_trajectories=len(groups[s]),available_total=len(allowed),sampled_episodes=int((labels==s).sum()),loss=float(loss.detach()),decision_states=info['decision_states'],valid_states=n,valid_pairs=info['valid_pairs'],grad_norm=float(g.norm()),within_task_weight=weight,between_task_weight=1/phase,aux_weight=r['settings']['weight'],weighted_grad_norm=float(contrib.norm())/phase*r['settings']['weight'],weighted_projection_on_task_gradient=float(contrib.dot(gfull)/gfull.square().sum()) if gfull.norm()>0 else None))
      abs_err=float((combined-gfull).abs().max());rel=float((combined-gfull).norm()/gfull.norm().clamp_min(1e-12));passed=abs_err<=2e-6 or rel<=2e-5
      checks.append(dict(method=method,checkpoint_phase=phase,task=task,mode=mode,repeat=repeat,full_loss=float(full.detach()),full_valid_states=N,full_grad_norm=float(gfull.norm()),max_abs_error=abs_err,relative_l2_error=rel,passed=passed));assert passed,'gradient reconstruction failed'
      for row in temp:row['reconstruction_passed']=passed
      metrics.extend(temp)
      for s,t in itertools.combinations(sorted(gs),2):
       value=cos(gs[s],gs[t]);cosines.append(dict(method=method,checkpoint_phase=phase,task=task,mode=mode,repeat=repeat,source_a=s,source_b=t,cosine=value,negative=int(value<0) if value is not None else None))
      del current,full,gs
    print('gradient',method,phase,mode,flush=True)
 csvsave('source_gradient_metrics.csv',metrics);csvsave('source_gradient_cosine.csv',cosines);csvsave('gradient_reconstruction.csv',checks)
 def aggregate(rows,keys,values,name):
  grouped=defaultdict(list)
  for row in rows:grouped[tuple(row[k] for k in keys)].append(row)
  result=[]
  for key,items in grouped.items():
   row=dict(zip(keys,key));row['repeats']=len(items)
   for val in values:
    v=[r[val] for r in items if r[val] is not None];row[val+'_n']=len(v);row[val+'_mean']=float(np.mean(v)) if v else None;row[val+'_std']=float(np.std(v,ddof=1)) if len(v)>1 else None
   result.append(row)
  csvsave(name,result);return result
 aggregate(metrics,['method','checkpoint_phase','task','mode','reference_phase'],['loss','valid_states','valid_pairs','grad_norm','within_task_weight','weighted_grad_norm','weighted_projection_on_task_gradient'],'source_gradient_summary.csv')
 aggregate(cosines,['method','checkpoint_phase','task','mode','source_a','source_b'],['cosine','negative'],'source_cosine_summary.csv')
 csvsave('source_td_gradient.csv',[dict(checkpoint_phase=p,status='unavailable',reason='No current-task TD ReplayBuffer/minibatch snapshots; cannot recover same-update TD gradient') for p in (2,3,4)])
 # TD loss is exact all-update logged masked TD MSE, not gradient event subsample.
 td=[]
 for method,r in runs.items():
  per=defaultdict(list);ret=js(r['p']/'retention.json')['stages'];filename='topk_preference_updates.jsonl' if method=='tkpr_k1' else 'tkpr_refresh_updates.jsonl';log=r['p']/filename
  manifest.append(dict(path=str(log.relative_to(root)),sha256=sha(log),bytes=log.stat().st_size))
  for line in log.open():
   row=json.loads(line);phase=row['phase']
   if phase not in (2,3,4):continue
   value=row['td_loss'];assert math.isfinite(value)
   start=ret[phase]['start'];end=ret[phase]['t_env'];q=min(3,max(0,int(4*(row['t_env']-start)/(end-start))))
   per[phase,'all'].append(value);per[phase,f'q{q+1}'].append(value)
  for (phase,part),vals in sorted(per.items()):td.append(dict(method=method,phase=phase,segment=part,n=len(vals),mean=float(np.mean(vals)),median=float(np.median(vals)),std=float(np.std(vals,ddof=1))))
 csvsave('td_loss_by_phase.csv',td);dump('input_manifest.json',dict(analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),python=sys.version,torch=th.__version__,numpy=np.__version__,device='cpu',threads=2,seed=68121,repeats=10,files=manifest,audit=audit))
 print('ALL COMPLETE',flush=True)

if __name__=='__main__':main()
