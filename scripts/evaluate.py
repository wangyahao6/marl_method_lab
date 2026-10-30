"""Independent evaluation of a finished run; does not train."""
import argparse, json, os, subprocess, sys, datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',required=True);p.add_argument('--episodes',required=True,type=int)
p.add_argument('--step',type=int,default=0);p.add_argument('--dry-run',action='store_true')
a=p.parse_args();run=Path(a.run).resolve()
meta=json.loads((run/'launch.json').read_text());suite=meta['request']['suite']
stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
out=run/'evaluation'/stamp
env=os.environ.copy();env['MARL_RUN_DIR']=str(out);env.setdefault('SC2PATH',str(ROOT/'runtime/StarCraftII'))
if a.episodes<=0:p.error('positive --episodes required')
if suite=='smac':
    cfg=json.loads((run/'config.json').read_text())
    models=run/'models';choices=[x for x in models.iterdir() if x.is_dir()] if models.exists() else []
    if len(choices)!=1:p.error('expected a single checkpoint family; inspect runs/<id>/models')
    cfg.update(evaluate=True,save_model=False,checkpoint_path=str(choices[0]),load_step=a.step,test_nepisode=a.episodes,local_results_path=str(out))
    cmd=[sys.executable,str(ROOT/'scripts/smac_entry.py'),str(out/'config.json')]
else:
    if a.step:p.error('continual evaluator currently evaluates its final checkpoint only')
    env['MARL_PROTOCOL']=str(run/'protocol.json')
    cmd=[sys.executable,str(ROOT/'drivers/continual/evaluate.py'),'--run',str(run),'--episodes',str(a.episodes),'--output',str(out)]
if a.dry_run:print(json.dumps({'command':cmd,'output':str(out)},indent=2));sys.exit()
out.mkdir(parents=True,exist_ok=False)
if suite=='smac':(out/'config.json').write_text(json.dumps(cfg,indent=2))
(out/'request.json').write_text(json.dumps({'command':cmd,'episodes':a.episodes,'source_run':str(run)},indent=2))
with (out/'console.log').open('w') as log:code=subprocess.call(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
(out/'launcher_status.json').write_text(json.dumps({'exit_code':code,'state':'exited' if code==0 else 'failed'}))
sys.exit(code)
