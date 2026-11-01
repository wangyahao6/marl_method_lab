import json,os,subprocess,time,tarfile,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/2s3z_type_ablation_winrate_20261101_attempt002'
def main():
 files=['scripts/type_ablation_winrate.py','scripts/summarize_type_winrate.py','scripts/run_type_winrate.py','docs/2S3Z_TYPE_WINRATE.md']
 identity=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
 (OUT/'source_identity.json').write_text(json.dumps(identity,indent=2))
 with tarfile.open(OUT/'source.tar.gz','w:gz') as tar:
  for folder in ['src','scripts','drivers','configs','docs']:
   for f in (ROOT/folder).rglob('*'):
    if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc':tar.add(f,arcname=str(f.relative_to(ROOT)))
 processes={};logs=[]
 for seed in [1111,2222,3535]:
  log=(OUT/f'worker_seed{seed}.log').open('w');logs.append(log)
  env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
  processes[seed]=subprocess.Popen([str(ROOT/'.venv/bin/python'),'scripts/type_ablation_winrate.py','--seed',str(seed)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 while True:
  status={str(seed):dict(pid=p.pid,state='running' if p.poll() is None else 'exited' if p.returncode==0 else 'failed',exit_code=p.returncode) for seed,p in processes.items()}
  temp=OUT/'queue_status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(OUT/'queue_status.json')
  if all(p.poll() is not None for p in processes.values()):break
  time.sleep(5)
 for log in logs:log.close()
 code=subprocess.call([str(ROOT/'.venv/bin/python'),'scripts/summarize_type_winrate.py'],cwd=ROOT)
 (OUT/'summary_status.json').write_text(json.dumps(dict(exit_code=code)))
if __name__=='__main__':main()
