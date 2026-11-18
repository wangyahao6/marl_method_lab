"""Authorized shared-prefix + 20 branches, max four branches concurrently."""
import argparse,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/hetero_agent_softprotect_3535_20261101';PARENT='hetero_Aprefix_seed3535_20261101_attempt001'
def cmd(run,role,task='R',group='N'):
 result=[str(ROOT/'.venv/bin/python'),'scripts/run.py','--suite','hetero','--algorithm','qmix','--method','baseline' if group=='N' else 'softprotect','--seed','3535','--run-id',run,'--protocol','configs/continual/neuron_protection/softprotect/hetero_agent_softprotect_ABA.json','--hetero-role',role,'--branch-task',task,'--group',group]
 if role=='branch':result+=['--parent',str(ROOT/'runs'/PARENT)]
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--adopt-prefix',action='store_true');options=parser.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);status={'max_concurrent_branches':4,'prefix':dict(run_id=PARENT,state='running',adopted=options.adopt_prefix),'branches':{}}
 def save():
  p=OUT/'queue_status.tmp';p.write_text(json.dumps(status,indent=2));p.replace(OUT/'queue_status.json')
 save()
 if options.adopt_prefix:
  while True:
   entry=json.loads((ROOT/'runs'/PARENT/'launcher_status.json').read_text())
   if entry['state']!='running':
    code=entry.get('exit_code',1);break
   time.sleep(5)
 else:code=subprocess.call(cmd(PARENT,'prefix'),cwd=ROOT)
 status['prefix'].update(state='exited' if code==0 else 'failed',exit_code=code);save()
 if code!=0:return
 todo=[(task,g) for task in ['R','M'] for g in ['N','T1','G1','R1','T2','G2','R2','T12','G12','R12']];active={}
 while todo or active:
  while todo and len(active)<4:
   task,g=todo.pop(0);run=f'hetero_{task}_{g}_seed3535_20261101_attempt001';p=subprocess.Popen(cmd(run,'branch',task,g),cwd=ROOT);active[run]=p;status['branches'][run]=dict(state='running',pid=p.pid);save()
  for run,p in list(active.items()):
   code=p.poll()
   if code is not None:status['branches'][run].update(state='exited' if code==0 else 'failed',exit_code=code);del active[run];save()
  if active:time.sleep(5)
 subprocess.call([str(ROOT/'.venv/bin/python'),'scripts/summarize_hetero_agent.py'],cwd=ROOT)
if __name__=='__main__':main()
