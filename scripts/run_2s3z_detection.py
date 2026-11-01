"""Three concurrent seeds; optionally adopt an already-running seed1111."""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/2s3z_pure_detection_20261101'

def main():
 p=argparse.ArgumentParser();p.add_argument('--adopt-1111',action='store_true');a=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 processes={};status={}
 for seed in [1111,2222,3535]:
  run_id=f'qmix_2s3z_detect_seed{seed}_20261101_attempt002'
  if seed==1111 and a.adopt_1111:
   status[str(seed)]={'state':'running','adopted':True};continue
  cmd=[str(ROOT/'.venv/bin/python'),'scripts/run.py','--suite','smac','--algorithm','qmix','--method','baseline','--map','2s3z','--steps','2000000','--seed',str(seed),'--run-id',run_id,'--set','pure_detection=true','--set','test_interval=10000','--set','test_nepisode=32','--set','save_model_interval=100000']
  processes[seed]=subprocess.Popen(cmd,cwd=ROOT)
  status[str(seed)]={'state':'running','launcher_pid':processes[seed].pid}
 while True:
  for seed in [1111,2222,3535]:
   if status[str(seed)]['state']!='running':continue
   if seed in processes:
    code=processes[seed].poll()
    if code is not None:status[str(seed)]={'state':'exited' if code==0 else 'failed','exit_code':code}
   else:
    path=ROOT/'runs'/f'qmix_2s3z_detect_seed{seed}_20261101_attempt002/launcher_status.json'
    try:
     value=json.loads(path.read_text())
     if value['state']!='running':status[str(seed)]=value
    except (FileNotFoundError,json.JSONDecodeError):pass
  temp=OUT/'queue_status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(OUT/'queue_status.json')
  if all(v['state']!='running' for v in status.values()):break
  time.sleep(5)
 subprocess.call([str(ROOT/'.venv/bin/python'),'scripts/summarize_2s3z_detection.py'],cwd=ROOT)
if __name__=='__main__':main()
