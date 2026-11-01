import subprocess,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/2s3z_pure_detection_20261101';OUT.mkdir(parents=True,exist_ok=True)
status={}
for seed in [1111,2222,3535]:
 status[str(seed)]={'state':'running'};(OUT/'queue_status.json').write_text(json.dumps(status,indent=2))
 cmd=[str(ROOT/'.venv/bin/python'),'scripts/run.py','--suite','smac','--algorithm','qmix','--method','baseline','--map','2s3z','--steps','2000000','--seed',str(seed),'--run-id',f'qmix_2s3z_detect_seed{seed}_20261101','--set','pure_detection=true','--set','test_interval=10000','--set','test_nepisode=32','--set','save_model_interval=100000']
 code=subprocess.call(cmd,cwd=ROOT)
 status[str(seed)]={'state':'exited' if code==0 else 'failed','exit_code':code};(OUT/'queue_status.json').write_text(json.dumps(status,indent=2))
subprocess.call([str(ROOT/'.venv/bin/python'),'scripts/summarize_2s3z_detection.py'],cwd=ROOT)
