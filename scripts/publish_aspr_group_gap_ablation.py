"""Package terminal ASPR 3G gap-ablation reports; preserve old datasets and original runs."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

from summarize_aspr_group_gap_ablation import summarize

ROOT=Path(__file__).resolve().parents[1]
DATASET="szc-vdn64-aspr-3g-gap-ablation"


def publish(plan_file):
    plan=json.loads(plan_file.read_text());out=plan_file.parent
    conn=sqlite3.connect(str(ROOT/"runtime/scheduler/state.sqlite3"))
    states=[dict(zip(("id","state","reason"),row)) for row in conn.execute(
        "select id,state,reason from tasks where id like ?",(plan["id"]+"%",))];conn.close()
    if any(r["state"] in ("running","pending","starting") for r in states):
        raise RuntimeError("The registered experiment queue is not terminal yet")
    summarize(plan_file)
    (out/"execution_summary.json").write_text(json.dumps(states,indent=2)+"\n")
    with (out/"README.md").open("a") as stream:stream.write("\n调度任务终态见 [执行记录](execution_summary.json)。核验缺项如实保留，不自动增加种子或重跑训练。\n")
    target=ROOT/"docs/results/comparisons"/DATASET
    if target.exists():raise RuntimeError("Dataset already exists; preserve it and review any update explicitly")
    target.mkdir()
    for path in [out/"README.md",out/"source_manifest.json",out/"execution_summary.json",*out.glob("*.csv")]:shutil.copy2(str(path),str(target/path.name))
    shutil.copytree(str(out/"configs"),str(target/"configs"))
    catalog_path=ROOT/"docs/results/catalog.json";catalog=json.loads(catalog_path.read_text())
    catalog["experiments"].append(dict(id=DATASET,category="comparisons",report=f"comparisons/{DATASET}/README.md",
        files={p.stem:f"comparisons/{DATASET}/{p.name}" for p in target.glob("*.csv")},
        task_order=["stalker","zealot","colossus"],training_seeds=[2222,3535],final_phase=2,
        comparison=dict(kind="shared_prefix_factorial_2x2",match_on=["seed","parent_run_id"],factors=dict(
            method=["ASPR-3G","ASPR-3G-Adjacent"],min_gap=[0,0.5]),
            reused_cells=[dict(method="ASPR-3G",min_gap=.5),dict(method="ASPR-3G-Adjacent",min_gap=0)],
            contrasts=["gap0-minus-gap05","adjacent-minus-3g","gap-by-method interaction"]),
        interpretation_limits=["两个训练种子；只新增四条 Z/C 分支，S 与 TKPR-K1 不重跑。",
            "同种子 S 模型与历史池相同，Z 池由各分支独立采集。",
            "三类跨组违反率统一使用阶段末离线口径；旧 3G 日志不含训练时逐类违反数。",
            "实际有效对数和 Q-gap 分布由训练采样编号重建，包含重复抽取。",
            "gap_kept 是间隔筛选后的诊断总体，loss 进一步应用方法拓扑；Adjacent G1-G3 不进入损失。"] ))
    catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+"\n")
    index=ROOT/"docs/results/README.md";text=index.read_text();anchor="| [SZC · ASPR-3G / ASPR-4G]";position=text.index(anchor)
    text=text[:position]+(f"| [SZC · ASPR-3G min_gap 2×2](comparisons/{DATASET}/README.md) | 3G/Adjacent × gap0/0.5，seed 2222、3535，共同 S | 配对 ACC/BWT/FWT/AUC、三类对违反率、实际约束量与 Q-gap 分布。 |\n")+text[position:];index.write_text(text)
    paths=[str(target.relative_to(ROOT)),"docs/results/catalog.json","docs/results/README.md"]
    subprocess.check_call(["git","diff","--check"],cwd=str(ROOT))
    subprocess.check_call(["git","add","--",*paths],cwd=str(ROOT))
    subprocess.check_call(["git","-c","user.name=Codex","-c","user.email=codex@openai.com","commit","--only",
        "-m","Publish shared-prefix ASPR three-group min-gap 2x2 ablation","--",*paths],cwd=str(ROOT))
    print(subprocess.check_output(["git","rev-parse","HEAD"],cwd=str(ROOT),universal_newlines=True).strip())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("plan",type=Path);publish(parser.parse_args().plan.resolve())
