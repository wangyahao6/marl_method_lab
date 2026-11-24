"""Check budget counts, nesting, summary means, unchanged inputs and loss implementation."""
import argparse,ast,csv,hashlib,json,math,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('.'));a=p.parse_args();root=a.repo.resolve();out=Path(__file__).resolve().parent
rows=lambda n:list(csv.DictReader((out/n).open()))
raw=rows('sampling_budget.csv');summary=rows('sampling_budget_summary.csv');checks=rows('verification_checks.csv');manifest=json.loads((out/'input_manifest.json').read_text())
assert len(raw)==1440 and len(checks)==360
for row in checks:
 assert row['parameters_unchanged']=='True' and float(row['aggregation_max_abs_error'])<=2e-6
for ph in (2,3,4):
 for repeat in range(30):
  group=[r for r in raw if int(r['checkpoint_phase'])==ph and int(r['repeat'])==repeat]
  allrows={r['budget']:r for r in group if r['scope']=='ALL'}
  assert int(allrows['K1']['sampled_episodes'])==8
  for n in (2,4,8):assert int(allrows[f'Refresh-{n}']['sampled_episodes'])==ph*n
  for task in 'ABCD'[:ph]:
   ids={r['budget']:json.loads(r['episode_ids']) for r in group if r['scope']==task}
   assert set(ids['Refresh-2'])<=set(ids['Refresh-4'])<=set(ids['Refresh-8'])
   for x in ids.values():assert len(x)==len(set(x))
for row in summary:
 values=[float(r['cosine_full']) for r in raw if all(r[k]==row[k] for k in ('checkpoint_phase','budget','scope'))]
 assert len(values)==30 and math.isclose(sum(values)/30,float(row['cosine_full_mean']),abs_tol=1e-12)
for row in manifest['files']:
 h=hashlib.sha256()
 with (root/row['path']).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==row['sha256'],row['path']
functions={'src/learners/ranking_preference.py':['topk_preference','_full_rank_loss','full_rank_pairs','reference_scale','reference_order','pair_state_loss','summarize'], 'src/learners/action_set_preference.py':['unroll','decision_masks','batch_episodes']}
for path,names in functions.items():
 old=ast.parse(subprocess.check_output(['git','show','9db0ed1:'+path],cwd=root,text=True));new=ast.parse((root/path).read_text())
 for name in names:
  x=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name);y=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name)
  assert ast.dump(x)==ast.dump(y)
path='src/learners/action_set_preference.py';old=ast.parse(subprocess.check_output(['git','show','9db0ed1:'+path],cwd=root,text=True));new=ast.parse((root/path).read_text())
get=lambda tree:next(x for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='ActionSetPreference' for x in c.body if isinstance(x,ast.FunctionDef) and x.name=='reduce_task_losses')
assert ast.dump(get(old))==ast.dump(get(new))
print('Verified 1440 raw rows; 360 unchanged-parameter/aggregation checks; 270 nested task draws; summary means;',len(manifest['files']),'input hashes; training loss and aggregation AST')
