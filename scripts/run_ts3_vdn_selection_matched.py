"""Authorized three missing TS3 VDN screening branches, serial; reuse existing prefixes."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ts3_vdn_selection_matched_20261104'
def command(item):
    return [str(ROOT/'.venv/bin/python'),'scripts/run.py','--suite','ts3','--algorithm','vdn','--method','softprotect','--seed',str(item['seed']),'--run-id',item['run'],'--protocol',item['protocol'],'--ts3-role','branch','--ts3-group','A2_union','--parent',str(ROOT/'runs'/item['parent'])]
def main():
    if (ROOT/'runtime/scheduler/enabled').exists() and '--dry-run' not in sys.argv:
        sys.path.insert(0,str(ROOT/'scheduler'))
        from legacy import submit_queue
        return submit_queue('selection_matched')
    plan=json.loads((OUT/'plan.json').read_text())
    if '--dry-run' in sys.argv:
        for item in plan['items']:
            code=subprocess.call(command(item)+['--dry-run'],cwd=ROOT)
            if code:return code
        return 0
    status=dict(state='running',pid=os.getpid(),started=time.time(),max_parallel=1,runs={})
    def save():
        tmp=OUT/'queue_status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(OUT/'queue_status.json')
    save()
    for item in plan['items']:
        try:
            process=subprocess.Popen(command(item),cwd=ROOT)
            status['runs'][item['run']]=dict(state='running',pid=process.pid,seed=item['seed'],mode=item['mode']);save()
            code=process.wait()
            status['runs'][item['run']].update(state='exited' if code==0 else 'failed',exit_code=code)
        except Exception:
            status['runs'][item['run']]=dict(state='launch_failed',error=traceback.format_exc())
        save()
    code=subprocess.call([str(ROOT/'.venv/bin/python'),'scripts/summarize_ts3_vdn_selection_matched.py'],cwd=ROOT)
    status.update(state='finished_pending_review' if code==0 and all(x.get('exit_code')==0 for x in status['runs'].values()) else 'finished_with_failures',summary_exit_code=code,ended=time.time());save()
    return code
if __name__=='__main__':sys.exit(main())
