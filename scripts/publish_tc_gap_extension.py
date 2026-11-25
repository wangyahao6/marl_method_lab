"""Package and commit the terminal two-seed extension, preserving the original dataset."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'drivers/continual')]
from offline_tc_gap_extension import verify
from summarize_tc_gap_extension import run as summarize

DATASET='ts5-vdn64-tc-gap-extension'


def read(path):return json.loads(path.read_text())
def rows(path):return list(csv.DictReader(path.open()))
def write(path,data):
    if not data:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in data for k in r)),lineterminator='\n');w.writeheader();w.writerows(data)
def relative(value):
    if isinstance(value,str):return value.replace(str(ROOT)+'/', '')
    if isinstance(value,list):return [relative(v) for v in value]
    if isinstance(value,dict):return {k:relative(v) for k,v in value.items()}
    return value


def publish(plan_path,commit=True):
    plan=read(plan_path);out=plan_path.parent;target=ROOT/'docs/results/parameter-ablations'/DATASET
    if target.exists():raise ValueError('Preserve existing published dataset: '+str(target))
    verify(out)
    summarize(plan_path)
    if read(out/'summary.json')['new_training_completed']!=10:
        raise ValueError('Not all ten new branches completed; partial report retained without publication')
    registry=sqlite3.connect(str(ROOT/'runtime/scheduler/state.sqlite3'));registry.row_factory=sqlite3.Row
    execution=[dict(r) for r in registry.execute('select id,state,reason,updated from tasks where id like ?', (plan['id']+'%',))]
    attempts=[dict(r) for r in registry.execute('select task,created,ended,exit_code from attempts where task like ?', (plan['id']+'%',))];registry.close()
    gate=plan['id']+'_offline_verify'
    gate_end=max(r['ended'] for r in attempts if r['task']==gate and r['exit_code']==0)
    training=[t for t in plan['tasks'] if t['type']=='training'];manifest={}
    for task in training:
        run=Path(task['run_dir'])
        status=read(run/'status.json');launch=read(run/'launch.json');config=read(run/'config.json')
        assert status['status']=='completed' and read(run/'launcher_status.json')['exit_code']==0
        assert config[config['method']]['min_gap']==task['metadata']['min_gap']
        assert len(read(run/'retention.json')['stages'])==5
        starts=[r['created'] for r in attempts if r['task']==task['id']]
        assert starts and min(starts)>=gate_end
        final=[]
        for path in (run/'evaluation').glob('*/results.json'):
            if read(path.parent/'launcher_status.json')['exit_code']==0:final+=read(path)['tasks']
        assert len(final)==5 and all(r['episodes']==128 for r in final),'missing/duplicate independent final evaluation'
        restore=read(run/'branch_restore.json');assert restore['model_hash']==restore['parent_model_hash']
        manifest[run.name]=dict(commit=launch['git_commit'],source_sha256=launch['source_sha256'],restore=restore)
    # Shared indices may be used by other authorized tasks: never include their uncommitted edits.
    common=['docs/results/catalog.json','docs/results/README.md']
    original=ROOT/'docs/results/parameter-ablations/ts5-vdn64-tc-coverage/README.md'
    for file in [*common,str(original.relative_to(ROOT))]:
        if subprocess.call(['git','diff','--quiet','HEAD','--',file],cwd=str(ROOT)):
            raise ValueError('Preserve pending index/report edits before publishing: '+file)
    target.mkdir()
    for path in [out/'REPORT.md',out/'OFFLINE_REPORT.md',out/'feasibility.json',out/'source_provenance.json',out/'summary.json',*out.glob('*.csv')]:
        shutil.copy2(str(path),str(target/path.name))
    for tag in ('gap0','gap05'):
        directory=target/tag;directory.mkdir()
        for path in (out/tag).glob('*.csv'):shutil.copy2(str(path),str(directory/path.name))
        shutil.copytree(str(out/tag/'configs'),str(directory/'configs'))
        for path in (directory/'configs').glob('*.json'):
            data=read(path);data['environment']=read(ROOT/'runs/tc_coverage'/path.stem/'env_info.json')
            path.write_text(json.dumps(relative(data),ensure_ascii=False,indent=2)+'\n')
    for row in (run_rows:=rows(target/'comparison_runs.csv')):
        row['run_dir']=relative(row['run_dir']);row['config']=('gap0' if float(row['min_gap'])==0 else 'gap05')+'/configs/'+row['run_id']+'.json'
    write(target/'comparison_runs.csv',run_rows)
    gap_hist=[]
    old=ROOT/'docs/results/parameter-ablations/ts5-vdn64-tc-coverage'
    for row in rows(old/'qgap_histogram.csv'):
        if int(row['seed']) in (2222,3535):gap_hist.append(dict(min_gap=0,**row))
    for seed in (2222,3535):
        for gap,tag in ((0,'gap0'),(.5,'gap05')):
            for phase in range(4):
                levels=('k1',) if gap==0 else ('k1','low','medium','high')
                data={level:np.load(out/'masks'/('seed%d'%seed)/tag/level/('phase%04d_distributions.npz'%phase)) for level in levels}
                for field in ('qgap','base_qgap','extra_qgap'):
                    defined=[d[field] for d in data.values() if d[field].size]
                    if not defined:continue
                    lo=min(float(d.min()) for d in defined);hi=max(float(d.max()) for d in defined)
                    edges=np.linspace(lo,hi if hi>lo else lo+1e-7,51)
                    for level,values in data.items():
                        values=values[field]
                        if not values.size:continue
                        counts,_=np.histogram(values,edges)
                        gap_hist += [dict(min_gap=gap,seed=seed,phase=phase,level=level,scope=field,lower=edges[i],upper=edges[i+1],upper_inclusive=i==49,pairs=int(n),total_pairs=len(values)) for i,n in enumerate(counts)]
                for d in data.values():d.close()
    write(target/'qgap_histogram.csv',gap_hist)
    histogram=rows(target/'tc_state_histogram.csv')
    histogram += [dict(min_gap=0,**r) for r in rows(old/'tc_state_histogram.csv') if int(r['seed']) in (2222,3535)]
    write(target/'tc_state_histogram.csv',histogram)
    (target/'source_manifest.json').write_text(json.dumps(relative(dict(new_training=manifest,gate_end_epoch=gate_end,execution=execution,attempts=attempts)),indent=2)+'\n')
    (target/'source_provenance.json').write_text(json.dumps(relative(read(target/'source_provenance.json')),indent=2)+'\n')
    report=(out/'REPORT.md').read_text().replace('# TS5-1M TC: K1-only and min_gap=0.5 extension','# TS5-1M · K1-only 与 min_gap=0.5 两种子扩展')
    report += '\n## Publication evidence\n\nAll ten new training states, configurations, independent final evaluations (5 tasks × 128 episodes per model), and 40 frozen history masks were verified before packaging. Training commits and successful gate timing are in [source_manifest.json](source_manifest.json). The original three-seed dataset, including seed4646, is preserved [here](../ts5-vdn64-tc-coverage/README.md). Only seeds2222/3535 are in the present main comparisons.\n\nOffline Q-gap means/medians/std are in offline_statistics.csv and per-cohort offline_statistics.csv; histograms are in qgap_histogram.csv. TC state histograms combine the new controls and reused old TC masks. Raw masks, full history pools, checkpoints and training logs remain remote.\n'
    # Link artifact paths for the actual published cohort layout.
    (target/'README.md').write_text(report)
    for path in target.rglob('*.csv'):path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    index_path=ROOT/'docs/results/README.md';text=index_path.read_text();anchor='| [TS5-1M · 传递闭包覆盖率]';position=text.index(anchor)
    text=text[:position]+'| [TS5-1M · K1 与 gap0/0.5 扩展](parameter-ablations/'+DATASET+'/README.md) | Seed2222/3535；10 新分支、6 旧 TC 分支复用 | 两个同种子四方法对照组、筛后共同状态、固定配额和 Q-gap。 |\n'+text[position:];index_path.write_text(text)
    catalog_path=ROOT/'docs/results/catalog.json';catalog=read(catalog_path);assert not any(e['id']==DATASET for e in catalog['experiments'])
    prefix='parameter-ablations/'+DATASET+'/'
    catalog['experiments'].append(dict(id=DATASET,category='parameter-ablations',report=prefix+'README.md',
       files={p.stem:prefix+p.name for p in target.glob('*.csv')},task_order=['A','B','C','D','E'],training_seeds=[2222,3535],final_phase=4,
       comparison=dict(kind='shared_reference_two_gap_four_arm_extension',match_on=['seed','min_gap','parent_run_id'],arms=['K1-only','Low-TC','Medium-TC','High-TC'],min_gap=[0,.5],reused_gap0_TC_branches=6,new_training_branches=10),
       interpretation_limits=['Seed4646仅保留在原始实验，不进入主对比。','gap0.5四组共用配额可行状态，保留比例约15%–17%；跨gap不控制状态集合。','间隔筛选同样作用于K1基础对，TC额外N−2预算与距离配额不降低。','只有两个训练种子；图结构、TC及残余Q-gap差异不构成标量TC因果证明。']))
    catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    with original.open('a') as f:f.write('\n## 两种子 K1 / min_gap 扩展\n\n原始三种子结果保持不变。新增 seed2222/3535 的 K1-only gap0 与四方法 gap0.5、以及复用旧 TC gap0 的两种子主对比，见 [扩展报告](../'+DATASET+'/README.md)。\n')
    paths=[str(target.relative_to(ROOT)),*common,str(original.relative_to(ROOT))]
    if commit:
        subprocess.check_call(['git','add','--',*paths],cwd=str(ROOT))
        subprocess.check_call(['git','diff','--cached','--check'],cwd=str(ROOT))
        subprocess.check_call(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','--only','-m','Publish two-seed TS5 K1 and min-gap TC comparisons','--',*paths],cwd=str(ROOT))
    print(json.dumps(dict(published=str(target),new_training_completed=10)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('plan',type=Path);parser.add_argument('--no-commit',action='store_true')
    args=parser.parse_args();publish(args.plan.resolve(),commit=not args.no_commit)
