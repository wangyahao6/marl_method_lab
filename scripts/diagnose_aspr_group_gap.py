"""Read-only 3G gap diagnostics: stage-end PVR and exact actual-sampling Q-gap distributions."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "drivers/continual")]
import numpy as np
import torch as th
from controllers import REGISTRY
from components.episode_buffer import ReplayBuffer
from common.runtime import episode_spec
from learners.action_set_preference import batch_episodes, decision_masks, unroll
from learners.ranking_preference import group_old_q, reference_order, full_rank_pairs

PAIRS = (("G1-G2", 0, 1), ("G2-G3", 1, 2), ("G1-G3", 0, 2))
BINS = {"normalized": [0,.1,.25,.5,.75,1,1.5,2,3,5,10,float("inf")],
        "raw": [0,.01,.05,.1,.25,.5,1,2,5,10,20,100,float("inf")]}


def read(path):
    return json.loads(path.read_text())


def write(out,name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/(name+".csv")).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def ratio(a,b):
    return a/b if b else None


def distribution(parts,usage,mask_name,scale):
    arrays, weights = [], []
    for part,weight in zip(parts,usage):
        if not weight:
            continue
        values=part[scale]
        if mask_name != "non_tie":values=values[part["kept"]] if part["participates"] or mask_name=="gap_kept" else values[:0]
        if values.size:
            arrays.append(values);weights.append(np.full(values.size,weight,dtype=np.int64))
    if not arrays:
        return dict(count=0,mean=None,std=None,min=None,q10=None,q25=None,q50=None,q75=None,q90=None,q95=None,max=None),[0]*(len(BINS[scale])-1)
    values=np.concatenate(arrays).astype(np.float64);weights=np.concatenate(weights)
    order=np.argsort(values,kind="stable");values,weights=values[order],weights[order]
    cumulative=np.cumsum(weights);count=int(cumulative[-1]);avg=float(np.dot(values,weights)/count)
    variance=max(0.,float(np.dot((values-avg)**2,weights)/count))
    quantiles={f"q{int(q*100)}":float(values[min(np.searchsorted(cumulative,q*count,side="left"),len(values)-1)]) for q in (.1,.25,.5,.75,.9,.95)}
    summary=dict(count=count,mean=avg,std=variance**.5,min=float(values[0]),max=float(values[-1]),**quantiles)
    hist=np.histogram(values,bins=np.array(BINS[scale]),weights=weights)[0]
    return summary,[int(x) for x in hist]


def diagnose(run,out):
    th.set_num_threads(1);out.mkdir(parents=True,exist_ok=False)
    config=read(run/"config.json");args=SimpleNamespace(**config);args.device,args.use_cuda="cpu",False
    protocol=read(run/"protocol.json");method=protocol["method"];settings=protocol[method]
    adjacent=method=="aspr_3g_adjacent_preference";gap=settings["min_gap"]
    scheme,groups,preprocess=episode_spec(read(run/"env_info.json"))
    augmented=ReplayBuffer(scheme,groups,1,2,preprocess=preprocess).scheme
    mac=REGISTRY[args.mac](augmented,groups,args)
    restore=read(run/"branch_restore.json");paths={}
    for root in (Path(restore["parent"]),run):
        for path in sorted((root/"action_set_memory").glob("phase*/success.th")):paths[path.parent.name]=path
    pools={};manifest=[]
    for path in paths.values():
        pool=th.load(path,map_location="cpu");episodes=[]
        manifest.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for ep in pool["episodes"]:
            batch,_=batch_episodes([ep],augmented,groups,"cpu",include_good=False,include_old_q=True)
            legal,decisions=decision_masks(batch);old=batch["old_q_values"][:,:-1]
            ids=ep["group_ids"][:-1].unsqueeze(0) if "group_ids" in ep else group_old_q(old,legal,decisions,3,settings["q_scale_floor"])
            normalized,_,_,_=reference_order(old,legal,settings["q_scale_floor"])
            old_gap,candidates,eligible,kept=full_rank_pairs(normalized,legal,decisions,settings,group_ids=ids)
            raw_gap=(old.unsqueeze(-1)-old.unsqueeze(-2)).abs();a,b=ids.unsqueeze(-1),ids.unsqueeze(-2)
            parts={}
            for name,high,low in PAIRS:
                select=((a==high)&(b==low))|((a==low)&(b==high))
                mask=select&eligible
                parts[name]=dict(mask=mask,sign=old_gap[mask].sign(),
                    raw=raw_gap[mask].numpy(),normalized=old_gap[mask].abs().numpy(),
                    kept=kept[mask].numpy(),candidate_pairs=int((select&candidates).sum()),
                    participates=not adjacent or name!="G1-G3")
            episodes.append(dict(episode=ep,parts=parts))
        pools[int(pool["phase"])]=dict(task=pool["task"],episodes=episodes)
    offline=[]
    with th.no_grad():
        for stage in read(run/"retention.json")["stages"]:
            phase=stage["phase"]
            if phase==0:continue
            checkpoint=run/stage["checkpoint"]/"agent.th"
            mac.agent.load_state_dict(th.load(checkpoint,map_location="cpu"))
            manifest.append(dict(path=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
            for source,pool in sorted(pools.items()):
                if source>=phase:continue
                totals={name:Counter() for name,_,_ in PAIRS}
                for ep in pool["episodes"]:
                    batch,_=batch_episodes([ep["episode"]],augmented,groups,"cpu",include_good=False,include_old_q=True)
                    current=unroll(mac,batch,batch.max_seq_length-1)
                    current_gap=current.unsqueeze(-1)-current.unsqueeze(-2)
                    for name,part in ep["parts"].items():
                        violation=(-part["sign"]*current_gap[part["mask"]]>=0).numpy();kept=part["kept"]
                        totals[name].update(candidate_pairs=part["candidate_pairs"],non_tie_pairs=len(violation),
                            gap_kept_pairs=int(kept.sum()),all_violations=int(violation.sum()),
                            kept_violations=int((violation&kept).sum()))
                for name,values in totals.items():
                    participates=not adjacent or name!="G1-G3"
                    offline.append(dict(phase=phase,training_task=stage["task"],t_env=stage["t_env"],
                        source_phase=source,historical_task=pool["task"],pair=name,episodes=len(pool["episodes"]),
                        min_gap=gap,participates_in_loss=participates,**values,
                        loss_pairs=values["gap_kept_pairs"] if participates else 0,
                        non_tie_violation_rate=ratio(values["all_violations"],values["non_tie_pairs"]),
                        gap_kept_violation_rate=ratio(values["kept_violations"],values["gap_kept_pairs"]),
                        loss_violation_rate=ratio(values["kept_violations"],values["gap_kept_pairs"]) if participates else None))
    write(out,"cross_group_offline",offline)
    usages=defaultdict(Counter);logged=Counter();updates=Counter();log=run/(method+"_updates.jsonl")
    with log.open() as stream:
        for line in stream:
            row=json.loads(line);phase=int(row["phase"]);updates[phase]+=1;logged[phase]+=row["valid_pairs"]
            for historical in row["historical_tasks"]:
                source=int(historical["source_phase"])
                usages[phase,source].update(historical["episode_ids"])
    sampled=[];summaries=[];histograms=[];reconstructed=Counter()
    for (phase,source),usage in sorted(usages.items()):
        pool=pools[source];weights=[usage[i] for i in range(len(pool["episodes"]))]
        common=dict(phase=phase,training_task=protocol["stages"][phase]["task"],source_phase=source,
                    historical_task=pool["task"],sampled_episodes=sum(weights),min_gap=gap)
        for name,_,_ in PAIRS:
            parts=[ep["parts"][name] for ep in pool["episodes"]];totals=Counter()
            for part,weight in zip(parts,weights):
                totals.update(candidate_pairs=weight*part["candidate_pairs"],non_tie_pairs=weight*len(part["raw"]),
                    gap_kept_pairs=weight*int(part["kept"].sum()),
                    loss_pairs=weight*int(part["kept"].sum()) if part["participates"] else 0)
            reconstructed[phase]+=totals["loss_pairs"]
            sampled.append(dict(common,pair=name,participates_in_loss=not adjacent or name!="G1-G3",**totals))
            for population in ("non_tie","gap_kept","loss"):
                for scale in ("raw","normalized"):
                    stats,hist=distribution(parts,weights,population,scale)
                    identity=dict(common,pair=name,population=population,q_gap_scale=scale)
                    summaries.append(dict(identity,**stats))
                    for low,high,count in zip(BINS[scale],BINS[scale][1:],hist):
                        histograms.append(dict(identity,bin_low=low,bin_high=high,count=count))
    write(out,"sampled_pairs",sampled);write(out,"q_gap_summary",summaries);write(out,"q_gap_histogram",histograms)
    write(out,"sampling_audit",[dict(phase=phase,updates=updates[phase],logged_valid_pairs=logged[phase],
        reconstructed_loss_pairs=reconstructed[phase],counts_match=logged[phase]==reconstructed[phase]) for phase in sorted(logged)])
    (out/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(run=run.name,output=str(out),offline_rows=len(offline),sampled_rows=len(sampled),
        all_pair_counts_match=all(logged[p]==reconstructed[p] for p in logged)),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();diagnose(args.run.resolve(),args.output.resolve())
