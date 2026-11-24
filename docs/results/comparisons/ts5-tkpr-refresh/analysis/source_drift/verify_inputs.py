"""Verify original artifacts and unchanged loss functions; never writes inputs."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('.'));a=p.parse_args();root=a.repo.resolve();out=Path(__file__).resolve().parent
manifest=json.loads((out/'input_manifest.json').read_text())
for row in manifest['files']:
 h=hashlib.sha256()
 with (root/row['path']).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==row['sha256'],row['path']
count=0
run=root/'runs/ts5_3535_tkpr_refresh_20261122_attempt001'
for path in run.glob('action_set_memory/*/success.th'):
 pool=torch.load(path,map_location='cpu')
 for ep in pool['episodes']:
  checkpoint=run/ep['reference_checkpoint']
  assert checkpoint.exists() and checkpoint.parent.name.startswith('phase{:04d}_'.format(ep['reference_phase']))
  count+=1
funcs={'src/learners/ranking_preference.py':['topk_preference','_full_rank_loss','reference_scale','reference_order','full_rank_pairs','pair_state_loss','summarize','_topk_actions'],'src/learners/action_set_preference.py':['unroll','decision_masks','batch_episodes'],'src/learners/q_learner.py':['_agent_gradient_vector','agent_gradient_metrics']}
for path,names in funcs.items():
 old=ast.parse(subprocess.check_output(['git','show','9db0ed1:'+path],cwd=root,text=True));new=ast.parse((root/path).read_text())
 for name in names:
  x=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name);y=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name)
  assert ast.dump(x)==ast.dump(y),(path,name)
print('Verified',len(manifest['files']),'artifact SHA-256 values,',count,'trajectory reference paths, 13 unchanged training-loss/mask/gradient functions')
