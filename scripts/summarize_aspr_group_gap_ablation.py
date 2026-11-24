"""Summarize the shared-prefix ASPR-3G x min_gap 2x2 ablation without changing historical artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from summarize_szc_tkpr_k1_shared import read,records,stage_rows,preference_rows,END_METRICS,TASKS


def write(out,name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/(name+".csv")).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def show(x):
    return "N/A" if x is None else f"{x:.2f}"


def summarize(plan_file):
    plan=read(plan_file);out=plan_file.parent;(out/"configs").mkdir(exist_ok=True)
    tables={key:[] for key in ("runs","stage_metrics","evaluations","learning_curves","preference_diagnostics",
        "cross_group_offline","sampled_pairs","q_gap_summary","q_gap_histogram","sampling_audit","independent_final",
        "run_audit","stage_steps","paired_metrics","interaction_metrics","config_audit")}
    manifest=[];parents=set();protocols={}

    def capture(path):
        if path.exists():manifest.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def collect(run,identity,parent=None):
        own=list(records(run/"evaluations.jsonl"));all_eval=list(records(parent/"evaluations.jsonl"))+own if parent else own
        retention=read(run/"retention.json");stages,curves=stage_rows(identity,retention,all_eval,own)
        tables["stage_metrics"].extend(stages);tables["learning_curves"].extend(curves)
        tables["evaluations"].extend(dict(identity,**{k:r[k] for k in ("phase","task","label","t_env","wins","episodes")}) for r in own)
        status,launcher,launch,restore,protocol=[read(run/name) for name in ("status.json","launcher_status.json","launch.json","branch_restore.json","protocol.json")]
        snapshot=dict(identity,launch_git_commit=launch.get("git_commit"),protocol=protocol,
            config=read(run/"config.json"),branch_restore=restore,env_info=read(run/"env_info.json"))
        config_path=f"configs/{run.name}.json";(out/config_path).write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+"\n")
        tables["runs"].append(dict(identity,commit=launch.get("git_commit"),status=status.get("status"),exit_code=launcher.get("exit_code"),
            optimizer=protocol.get("optimizer","rmsprop"),replay_on_switch=protocol.get("replay_on_switch"),
            epsilon_on_switch=protocol.get("epsilon_on_switch"),completed_global_steps=status.get("t_env"),
            trained_steps=status.get("t_env",0)-restore.get("t_env",0) if status else None,
            restore_model_hash=restore.get("model_hash"),prefix_model_hash=restore.get("parent_model_hash"),config=config_path))
        for name in ("status.json","launcher_status.json","launch.json","protocol.json","branch_restore.json","retention.json","evaluations.jsonl"):capture(run/name)
        if not parent:return
        method=protocol.get("method");protocols[identity["method"],identity["min_gap"],identity["seed"]]=protocol
        if method:tables["preference_diagnostics"].extend(preference_rows(identity,run,method))
        for path in sorted((run/"evaluation").glob("*/results.json")):
            data=read(path);execution=read(path.parent/"launcher_status.json")
            for row in data.get("tasks",[]):tables["independent_final"].append(dict(identity,task=row["task"],wins=row["wins"],
                episodes=row["episodes"],win_pct=100*row["wins"]/row["episodes"],exit_code=execution.get("exit_code"),t_env=data["t_env"]))
            capture(path);capture(path.parent/"launcher_status.json")
        diag=out/"diagnostics"/run.name
        for name in ("cross_group_offline","sampled_pairs","q_gap_summary","q_gap_histogram","sampling_audit"):
            path=diag/(name+".csv")
            if path.exists():
                with path.open() as stream:tables[name].extend(dict(identity,**row) for row in csv.DictReader(stream))
                capture(path)
        capture(diag/"source_manifest.json")
        own_stages=[r for r in retention.get("stages",[]) if r["phase"] in (1,2)]
        for row in own_stages:
            phase=row["phase"];prior=next((r for r in retention["stages"] if r["phase"]==phase-1),{})
            nominal=sum(s["steps"] for s in protocol["stages"][:phase+1]);end=row["t_env"]
            tables["stage_steps"].append(dict(identity,phase=phase,task=row["task"],start=row["start"],end=end,
                actual_steps=end-row["start"],nominal_cumulative_end=nominal,budget_reached=end>=nominal,
                start_matches_previous_end=row["start"]==prior.get("t_env")))
        final=next((r for r in stages if r["phase"]==2),None)
        independent=[r for r in tables["independent_final"] if r["run_id"]==run.name]
        independent_match=bool(final and len(independent)==3 and all(r["exit_code"]==0 and abs(r["win_pct"]-final[f'{r["task"]}_win_pct'])<1e-8 for r in independent)) if independent else None
        sampled=[r for r in tables["sampling_audit"] if r["run_id"]==run.name]
        step_rows=[r for r in tables["stage_steps"] if r["run_id"]==run.name]
        tables["run_audit"].append(dict(identity,completed=status.get("status")=="completed",launcher_success=launcher.get("exit_code")==0,
            restored_model_matches_prefix=bool(restore.get("model_hash") and restore["model_hash"]==restore.get("parent_model_hash")),
            two_branch_stages_present=len(step_rows)==2,stage_budgets_reached=bool(step_rows) and all(r["budget_reached"] and r["start_matches_previous_end"] for r in step_rows),
            final_matrix_present=bool(final),independent_final_matches=independent_match,
            sampled_pair_counts_match=bool(sampled) and all(r["counts_match"]=="True" for r in sampled)))

    for spec in plan["runs"]:
        parent=Path(spec["parent"]);run=Path(spec["run_dir"])
        if parent.name not in parents:
            parents.add(parent.name);collect(parent,dict(run_id=parent.name,method="Shared S",min_gap=None,seed=spec["seed"],
                role="shared-prefix",parent_run_id=None,newly_trained=False))
        identity={k:spec[k] for k in ("run_id","method","min_gap","seed","role","parent_run_id","newly_trained")}
        collect(run,identity,parent)
    lookup={(r["method"],r["min_gap"],r["seed"],r["phase"]):r for r in tables["stage_metrics"] if r["parent_run_id"]}
    for seed in (2222,3535):
        for method in ("ASPR-3G","ASPR-3G-Adjacent"):
            a,b=protocols.get((method,0.,seed)),protocols.get((method,.5,seed))
            if a and b:
                x,y=json.loads(json.dumps(a)),json.loads(json.dumps(b));x[x["method"]]["min_gap"]=y[y["method"]]["min_gap"]=None
                tables["config_audit"].append(dict(method=method,seed=seed,only_min_gap_differs=x==y))
        for phase in (1,2):
            comparisons=[("gap0-minus-gap05",(m,0.),(m,.5)) for m in ("ASPR-3G","ASPR-3G-Adjacent")]
            comparisons += [("adjacent-minus-3g",("ASPR-3G-Adjacent",g),("ASPR-3G",g)) for g in (0.,.5)]
            for contrast,(m,g),(cm,cg) in comparisons:
                a,b=lookup.get((m,g,seed,phase)),lookup.get((cm,cg,seed,phase))
                if a and b:tables["paired_metrics"].append(dict(contrast=contrast,method=m,min_gap=g,control_method=cm,control_min_gap=cg,
                    seed=seed,phase=phase,training_task=TASKS[phase],run_id=a["run_id"],control_run_id=b["run_id"],
                    **{k+"_delta":a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in END_METRICS}))
            four=[lookup.get((m,g,seed,phase)) for m,g in (("ASPR-3G-Adjacent",.5),("ASPR-3G-Adjacent",0.),("ASPR-3G",.5),("ASPR-3G",0.))]
            if all(four):tables["interaction_metrics"].append(dict(seed=seed,phase=phase,formula="(Adjacent_gap05-Adjacent_gap0)-(3G_gap05-3G_gap0)",
                **{k+"_interaction":four[0][k]-four[1][k]-four[2][k]+four[3][k] if all(r[k] is not None for r in four) else None for k in END_METRICS}))
    for name,rows in tables.items():write(out,name,rows)
    (out/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    lines=["# SZC · ASPR-3G / Adjacent 的 min_gap 2×2 消融", "",
        "ASPR-3G 与 ASPR-3G-Adjacent 分别比较 min_gap=0 与 0.5。只新增 3G gap0、Adjacent gap0.5 各两个种子分支，"
        "复用已有 3G gap0.5 与 Adjacent gap0。未新增 TKPR-K1 或 S 训练。",
        "seed 2222、3535；每个种子共用原 TKPR-K1 的 2M S 检查点及同一历史 S 胜局池；Z/C 各按原累计协议推进到4M/6M，"
        "结束完整回合。双层 VDN/GRU64、RMSprop、FT、逐任务 ε、温度/权重/标准化/采样/评价均沿用对应方法原协议。"
        "方法实现未改变；两份新增配置相对各自原配置只改 min_gap。", "",
        "## 运行核验", "", "| 方法 | gap | seed | 新增 | 完成/退出0 | 恢复哈希 | 阶段预算 | 终局独立评价 | 采样对数重建 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in tables["run_audit"]:lines.append(f'| {r["method"]} | {r["min_gap"]} | {r["seed"]} | {r["newly_trained"]} | '
        f'{r["completed"] and r["launcher_success"]} | {r["restored_model_matches_prefix"]} | {r["stage_budgets_reached"]} | '
        f'{r["independent_final_matches"]} | {r["sampled_pair_counts_match"]} |')
    lines += ["", "历史 3G 若无额外终局重评，独立评价复核记 N/A；原阶段末128局独立贪心评价仍用于性能指标。"
        "新增四条分支均安排终局每任务128局独立重评。[逐运行核验](run_audit.csv)、[实际阶段步数](stage_steps.csv)、"
        "[仅 min_gap 差异核对](config_audit.csv)保留细节。", "",
        "## 阶段指标", "", "| 方法 | gap | seed | 阶段 | S % | Z % | C % | ACC % | BWT pp | FWT pp | 当前任务 AUC % |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in tables["stage_metrics"]:
        lines.append(f'| {r["method"]} | {r["min_gap"]} | {r["seed"]} | {r["training_task"]} | '+" | ".join(
            show(r.get(k)) for k in (*(f"{t}_win_pct" for t in TASKS),*END_METRICS))+" |")
    lines += ["", "## 两种子最终均值", "", "| 方法 | gap | ACC % | BWT pp | FWT pp | C AUC % |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for m in ("ASPR-3G","ASPR-3G-Adjacent"):
        for g in (0.,.5):
            rows=[r for r in tables["stage_metrics"] if r["method"]==m and r["min_gap"]==g and r["phase"]==2]
            if len(rows)==2:lines.append(f'| {m} | {g} | '+" | ".join(show(mean(r[k] for r in rows)) for k in END_METRICS)+" |")
    final_contrasts=[r for r in tables["paired_metrics"] if r["phase"]==2]
    if final_contrasts:
        lines += ["", "| 对照方向 | 方法 | gap | seed | ΔACC pp | ΔBWT pp | ΔFWT pp | ΔC AUC pp |",
                  "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
        for r in final_contrasts:
            lines.append(f'| {r["contrast"]} | {r["method"]} | {r["min_gap"]} | {r["seed"]} | '+
                " | ".join(show(r[k+"_delta"]) for k in END_METRICS)+" |")
        for method in ("ASPR-3G","ASPR-3G-Adjacent"):
            rows=[r for r in final_contrasts if r["contrast"]=="gap0-minus-gap05" and r["method"]==method]
            if len(rows)==2:
                lines.append(method+"：gap0 相对 gap0.5 的两种子最终均值差为 "+
                    "、".join(k+" "+show(mean(r[k+"_delta"] for r in rows))+" pp" for k in END_METRICS)+"。")
    lines += ["", "[逐种子配对差](paired_metrics.csv)同时报告同方法 gap0−gap0.5 与同 gap 的 Adjacent−3G；"
        "[交互差值](interaction_metrics.csv)定义为 (Adjacent0.5−Adjacent0)−(3G0.5−3G0)。"
        "ACC 与任务端点衡量结果，Z/C AUC 衡量当前任务学习过程，BWT 结合各旧任务端点判断保持。只有两个种子，不作显著性或普遍优劣断言。", "",
        "## 动作对与 Q-gap", "",
        "[三类跨组对的阶段末离线违反率](cross_group_offline.csv)重放各自完整历史胜局池，固定旧 Q/分组，使用阶段末当前 Q；"
        "当前平局计为违反。non_tie_violation_rate 对所有非平局跨组对统计，便于跨 gap 直接对照；"
        "gap_kept_violation_rate 仅统计 |标准化旧 Q 差|≥min_gap 的对；loss_violation_rate 仅用于实际进入该方法损失的对。"
        "Adjacent 的 G1-G3 只做诊断，loss_pairs=0。S 池相同；Z 池由各分支独立采集，不视为共同样本因果对照。",
        "[实际采样动作对数](sampled_pairs.csv)从每次更新的 source_phase/episode_ids 重建，重复轨迹重复计数，"
        "按 G1-G2/G2-G3/G1-G3 分别保存候选、非平局、间隔通过与实际入损失对数；与日志 valid_pairs 总计核对。"
        "旧 3G 日志只记录总违反率，未记录三类对逐更新的违反数，不能恢复训练中逐类违反率；三类比较统一使用阶段末离线口径。",
        "[Q-gap 分布摘要](q_gap_summary.csv)与[直方图](q_gap_histogram.csv)使用实际采样次数加权，分别记录原始旧 Q 差和标准化旧 Q 差，"
        "并区分 non_tie、gap_kept、loss 三个总体。分位数使用加权经验分布逆 CDF；标准化复用原 reference_order()/full_rank_pairs()，"
        "保留原实现精确平局排除和 ≥min_gap 边界。记录分布不改变训练筛选。", "",
        "## 来源与复用", "", "[运行清单](runs.csv)、[配置快照](configs/)、[源文件校验](source_manifest.json)、"
        "[逐次评价](evaluations.csv)、[学习曲线](learning_curves.csv)、[终局重评](independent_final.csv)、"
        "[训练日志汇总](preference_diagnostics.csv)、[采样重建核验](sampling_audit.csv)。",
        "ACC/BWT/FWT 按仓库标准口径，AUC 是各阶段当前任务起点/定期/阶段末贪心胜率对实际步数的梯形积分均值。"
        "已有前缀、方法结果和原始日志只读引用；本批次诊断另存 results/本批次/diagnostics/<run-id>/，不覆盖旧诊断。",
        "[已有 ASPR-3G gap0.5](../szc-vdn64-aspr-groups/README.md)；[已有 Adjacent gap0](../szc-vdn64-aspr-3g-adjacent/README.md)。"
        "完整原始产物留在 rl-navigation-ubuntu:/home/wyh/marl_method_lab/runs/。", ""]
    if tables["cross_group_offline"]:
        lines += ["", "## C 阶段末三类跨组违反率", "",
            "以下均为全部非平局对的离线违反率，避免 min_gap 改变分母掩盖排序变化。", "",
            "| 方法 | gap | seed | 历史任务 | G1-G2 % | G2-G3 % | G1-G3 % |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |"]
        grouped={}
        for r in tables["cross_group_offline"]:
            if int(r["phase"])==2:grouped.setdefault((r["method"],r["min_gap"],r["seed"],r["historical_task"]),{})[r["pair"]]=r
        for key,values in sorted(grouped.items()):
            m,g,seed,historical=key
            lines.append(f'| {m} | {g} | {seed} | {historical} | '+" | ".join(
                show(100*float(values[p]["non_tie_violation_rate"])) if p in values and values[p]["non_tie_violation_rate"] else "N/A"
                for p in ("G1-G2","G2-G3","G1-G3"))+" |")
    if tables["sampled_pairs"]:
        lines += ["", "## 实际训练采样约束量", "",
            "G1-G3 对 Adjacent 的入损失数量始终为 0，其候选与间隔通过计数仍保留在 CSV 供诊断。", "",
            "| 方法 | gap | seed | 训练阶段 | 历史任务 | G1-G2 入损失 | G2-G3 入损失 | G1-G3 入损失 |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: |"]
        grouped={}
        for r in tables["sampled_pairs"]:grouped.setdefault((r["method"],r["min_gap"],r["seed"],r["phase"],r["historical_task"]),{})[r["pair"]]=r
        for key,values in sorted(grouped.items()):
            m,g,seed,phase,historical=key
            lines.append(f'| {m} | {g} | {seed} | {phase} | {historical} | '+
                " | ".join(values[p]["loss_pairs"] if p in values else "N/A" for p in ("G1-G2","G2-G3","G1-G3"))+" |")
    complete=[r for r in tables["run_audit"] if r["newly_trained"] and r["completed"] and r["launcher_success"]]
    if len(complete)<4:lines.insert(2,"**新增实验尚未全部完成；当前已有单元不构成完整 2×2 结果。**\n")
    (out/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(new_completed=len(complete),runs=len(tables["run_audit"]),config_checks=tables["config_audit"]),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("plan",type=Path)
    summarize(parser.parse_args().plan.resolve())
