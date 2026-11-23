"""Package terminal SZC adjacent results and commit the isolated public dataset (no training)."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

from summarize_aspr_3g_adjacent import export

ROOT = Path(__file__).resolve().parents[1]
DATASET = "szc-vdn64-aspr-3g-adjacent"


def publish(plan_file):
    out = plan_file.parent
    plan = json.loads(plan_file.read_text())
    conn = sqlite3.connect(str(ROOT / "runtime/scheduler/state.sqlite3"))
    states = [dict(zip(("id", "state", "reason"), row)) for row in conn.execute(
        "select id,state,reason from tasks where id like ?", (plan["id"] + "%",))]
    conn.close()
    terminal = {"completed", "failed", "blocked", "cancelled", "unknown"}
    if any(row["state"] not in terminal for row in states):
        raise RuntimeError("Publication must wait for all batch tasks to reach terminal state")
    export(plan_file)
    (out / "execution_summary.json").write_text(json.dumps(states, indent=2) + "\n")
    with (out / "README.md").open("a") as stream:
        stream.write("\n调度执行状态见 [执行汇总](execution_summary.json)。失败与缺项按原样保留，不自动重跑。\n")
    target = ROOT / "docs/results/comparisons" / DATASET
    if target.exists():
        raise RuntimeError("Public dataset already exists; preserve it and review any follow-up explicitly")
    target.mkdir()
    files = [out / "README.md", out / "source_manifest.json", out / "execution_summary.json", *out.glob("*.csv")]
    for path in files:
        shutil.copy2(str(path), str(target / path.name))
    shutil.copytree(str(out / "configs"), str(target / "configs"))
    catalog_path = ROOT / "docs/results/catalog.json"
    catalog = json.loads(catalog_path.read_text())
    entry = dict(id=DATASET, category="comparisons", report=f"comparisons/{DATASET}/README.md",
        files={path.stem: f"comparisons/{DATASET}/{path.name}" for path in target.glob("*.csv")},
        task_order=["stalker", "zealot", "colossus"], training_seeds=[2222, 3535], final_phase=2,
        comparison=dict(kind="shared_prefix", match_on=["seed", "parent_run_id"],
                        treatments=["ASPR-3G-Adjacent"], primary_control="TKPR-K1"),
        interpretation_limits=["两个训练种子；两方法均 min_gap=0。",
            "同种子 S 检查点和胜局池相同；Z/C 独立训练与采集。",
            "G1-G3 仅诊断，不进入相邻组损失；实际采样包含重复轨迹。",
            "全量历史池离线统计与实际训练采样统计不可混合；组大小分布行重复携带动作对总数。"])
    catalog["experiments"].append(entry)
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    index = ROOT / "docs/results/README.md"
    text = index.read_text()
    anchor = "| [SZC · ASPR-3G / ASPR-4G]"
    position = text.index(anchor)
    text = text[:position] + (f"| [SZC · ASPR-3G-Adjacent vs TKPR-K1](comparisons/{DATASET}/README.md) | "
        "双层 VDN64，seed 2222、3535，共享 2M S，RMSprop，min_gap=0 | "
        "逐阶段胜率、ACC/BWT/FWT/AUC、配对差、采样动作对与 G1-G3 离线违反率。 |\n") + text[position:]
    index.write_text(text)
    # Validate references and joins in the newly created, completed package only.
    for path in entry["files"].values():
        if not (ROOT / "docs/results" / path).exists():
            raise RuntimeError(f"Missing exported dataset: {path}")
    subprocess.check_call(["git", "diff", "--check"], cwd=str(ROOT))
    paths = [str(target.relative_to(ROOT)), "docs/results/catalog.json", "docs/results/README.md"]
    subprocess.check_call(["git", "add", "--", *paths], cwd=str(ROOT))
    subprocess.check_call(["git", "-c", "user.name=Codex", "-c", "user.email=codex@openai.com",
        "commit", "--only", "-m", "Publish SZC adjacent-group versus matched gap-zero K1 results", "--", *paths], cwd=str(ROOT))
    print(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    publish(parser.parse_args().plan.resolve())
