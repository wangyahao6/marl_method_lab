"""Generate and independently audit fixed masks, then apply the requested pretraining gate."""
import argparse
import csv
import gzip
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'drivers/continual')]
from learners.pair_tc import TCPPreference, LEVELS, quotas, digest_masks
from learners.action_set_preference import batch_episodes, decision_masks
from learners.ranking_preference import reference_order
from common.runtime import episode_spec
from drivers.continual.memory import PreferenceHistory


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit_episode(ep, prepared, scheme, groups):
    batch,_=batch_episodes([ep],scheme,groups,'cpu',include_good=False,include_old_q=True)
    legal,decision=decision_masks(batch)
    q=batch['old_q_values'][:,:-1]
    _,scale,_,_=reference_order(q,legal,.1)
    length=ep['length'];size=legal.size(-1)
    left,right=th.triu_indices(size,size,1)
    mask=prepared['pair_mask']
    counts=prepared['pair_counts']
    valid=counts[...,0]>0
    strict=legal[0,...,left]&legal[0,...,right]&(q[0,...,left]!=q[0,...,right])
    expected=decision[0]&(legal[0].sum(-1)>=4)&~((legal[0,...,left]&legal[0,...,right])&~strict).any(-1)
    if not th.equal(valid,expected[:length]):
        raise ValueError('valid state set differs from the common eligibility rule')
    if (mask & ~strict[:length]).any() or mask[~valid].any():
        raise ValueError('selected illegal, tied, or invalid-state pair')
    order=th.sort(q[0].masked_fill(~legal[0],-th.inf),descending=True,stable=True).indices
    rank=order.argsort(-1)
    top=(left==order[...,:1])|(right==order[...,:1])
    top=top[:length]&strict[:length]&valid[...,None]
    if not th.equal(mask&top,top):
        raise ValueError('missing K1 constraint')
    n=legal[0,:length].sum(-1)
    if not th.equal(mask.sum(-1)[valid],(2*n-3)[valid]):
        raise ValueError('per-state pair budget mismatch')
    distance=(rank[...,left]-rank[...,right]).abs()[:length]
    extra=mask&~top
    bucket=(distance-1).clamp(0,2)
    quota_counts=th.stack([(extra&(bucket==k)).sum(-1) for k in range(3)],-1)
    for count in n[valid].unique().tolist():
        use=valid&(n==count)
        if not np.array_equal(quota_counts[use].numpy(),np.broadcast_to(quotas(count),(int(use.sum()),3))):
            raise ValueError('per-state rank-distance quota mismatch')
    # Independent Boolean Floyd-Warshall, not the selection algorithm's bitset update.
    shape=mask.shape[:-1]
    reach=np.zeros((*shape,size,size),dtype=bool)
    lo,hi=left.numpy(),right.numpy()
    forward=(q[0,:length,...,left]>q[0,:length,...,right]).numpy()
    reach[...,lo,hi]=mask.numpy()&forward
    reach[...,hi,lo]=mask.numpy()&~forward
    for k in range(size):
        reach |= reach[..., :,k:k+1]&reach[..., k:k+1,:]
    reached=reach.sum(axis=(-1,-2))
    candidates=counts[...,0].numpy()
    gap=((q[0,...,left]-q[0,...,right]).abs()/scale[0])[:length].numpy()
    rows=[]
    for t,agent in valid.nonzero().tolist():
        direct=int(mask[t,agent].sum());den=int(candidates[t,agent]);tc=int(reached[t,agent])
        rows.append(dict(step=t,agent=agent,n_actions=int(n[t,agent]),direct_pairs=direct,
                         possible_pairs=den,closure_pairs=tc,direct_coverage=direct/den,
                         tc=tc/den,complete_order=int(tc==den),closure_gain=(tc-direct)/den,
                         extra_near=int(quota_counts[t,agent,0]),extra_mid=int(quota_counts[t,agent,1]),
                         extra_far=int(quota_counts[t,agent,2])))
    return rows,gap[mask.numpy()],gap[extra.numpy()],quota_counts.sum((0,1)).tolist(),digest_masks([valid]),digest_masks([top])


def generate(parent):
    output=parent/'offline_tc';output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((parent/'protocol.json').read_text())
    info=json.loads((parent/'env_info.json').read_text())
    scheme,groups,preprocess=episode_spec(info)
    from components.episode_buffer import ReplayBuffer
    scheme=ReplayBuffer(scheme,groups,1,2,preprocess=preprocess).scheme
    stats=[];provenance=[]
    seed=json.loads((parent/'config.json').read_text())['seed']
    fields=['seed','phase','level','episode','step','agent','n_actions','direct_pairs','possible_pairs',
            'closure_pairs','direct_coverage','tc','complete_order','closure_gain','extra_near','extra_mid','extra_far']
    with gzip.open(output/'per_state.csv.gz','wt',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for phase in range(len(protocol['stages'])-1):
            source=PreferenceHistory.phase_path(parent,phase)/'success.th'
            pool=th.load(source,map_location='cpu')
            if not pool['episodes']:
                raise ValueError('empty historical success pool: '+str(source))
            checkpoint=parent/pool['reference_checkpoint']
            provenance.append(dict(phase=phase,task=pool['task'],source_pool=str(source),
                                   pool_sha256=sha(source),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
            controls=[]
            for level in LEVELS:
                settings=dict(weight=1.,temperature=1.,min_gap=0,q_scale_floor=.1,pair_seed=seed,batch_episodes=8)
                pref=TCPPreference(level,settings,np.random.default_rng(seed),scheme,groups,lambda x:None)
                prepared,audit=pref.prepare_pool(pool,prior_budget=0)
                all_rows=[];gaps=[];extra_gaps=[];distance=np.zeros(3,dtype=np.int64)
                states_hash=hashlib.sha256();top_hash=hashlib.sha256();budget_hash=hashlib.sha256()
                for index,(episode,selected) in enumerate(zip(pool['episodes'],prepared)):
                    rows,gap,extra,quota,state_digest,top_digest=audit_episode(episode,selected,scheme,groups)
                    enriched=[dict(seed=seed,phase=phase,level=level,episode=index,**row) for row in rows]
                    writer.writerows(enriched);all_rows.extend(rows);gaps.extend(gap);extra_gaps.extend(extra);distance+=quota
                    states_hash.update(state_digest.encode());top_hash.update(top_digest.encode())
                    budget_hash.update(selected['pair_counts'].numpy().tobytes())
                if not all_rows:
                    raise ValueError('no eligible decision states: '+str(source))
                totals={k:sum(row[k] for row in all_rows) for k in ('direct_pairs','possible_pairs','closure_pairs','complete_order')}
                tc=np.array([row['tc'] for row in all_rows]);gaps=np.asarray(gaps);extra_gaps=np.asarray(extra_gaps)
                row=dict(seed=seed,phase=phase,task=pool['task'],level=level,states=len(all_rows),**totals,
                         direct_edge_coverage=totals['direct_pairs']/totals['possible_pairs'],
                         micro_tc=totals['closure_pairs']/totals['possible_pairs'],macro_tc=float(tc.mean()),
                         complete_order_coverage=totals['complete_order']/len(all_rows),
                         closure_gain=(totals['closure_pairs']-totals['direct_pairs'])/totals['possible_pairs'],
                         macro_closure_gain=float(np.mean([r['closure_gain'] for r in all_rows])),
                         tc_min=float(tc.min()),tc_p10=float(np.quantile(tc,.1)),tc_median=float(np.median(tc)),
                         tc_p90=float(np.quantile(tc,.9)),tc_max=float(tc.max()),
                         extra_near=int(distance[0]),extra_mid=int(distance[1]),extra_far=int(distance[2]),
                         total_near=int(distance[0])+len(all_rows),total_mid=int(distance[1])+len(all_rows),
                         total_far=int(distance[2])+sum(r['n_actions']-3 for r in all_rows),
                         mean_pairs_per_state=totals['direct_pairs']/len(all_rows),
                         qgap_mean=float(gaps.mean()),qgap_median=float(np.median(gaps)),qgap_std=float(gaps.std()),
                         extra_qgap_mean=float(extra_gaps.mean()),extra_qgap_median=float(np.median(extra_gaps)),
                         extra_qgap_std=float(extra_gaps.std()),valid_state_sha256=states_hash.hexdigest(),
                         top1_sha256=top_hash.hexdigest(),counts_sha256=budget_hash.hexdigest(),
                         pair_mask_sha256=audit['pair_mask_sha256'])
                # Preserve distributions for overlap / Q-gap analyses without imposing filters.
                directory=output/level;directory.mkdir(exist_ok=True)
                np.savez_compressed(directory/('phase%04d_distributions.npz'%phase),qgap=gaps,extra_qgap=extra_gaps,tc=tc)
                path=directory/('phase%04d.th'%phase)
                th.save(dict(masks=[ep['pair_mask'] for ep in prepared],counts=[ep['pair_counts'] for ep in prepared],
                             audit=audit,source_sha256=sha(source)),path)
                row['mask_file_sha256']=sha(path);stats.append(row)
                controls.append((row['valid_state_sha256'],row['top1_sha256'],row['counts_sha256'],tuple(distance)))
                if audit['closure_pairs'] != totals['closure_pairs']:
                    raise ValueError('closure audit disagrees with independent Floyd-Warshall')
            if len(set(controls))!=1:
                raise ValueError('strict mask controls do not match')
    write_csv(output/'statistics.csv',stats)
    launch=json.loads((parent/'launch.json').read_text())
    (output/'provenance.json').write_text(json.dumps(dict(seed=seed,training_commit=launch['git_commit'],
          training_source_sha256=launch['source_sha256'],config_sha256=sha(parent/'config.json'),
          return_state_sha256=sha(parent/'return_state.th'),environment=info,
          runtime=dict(python=sys.version,platform=platform.platform(),torch=th.__version__,numpy=np.__version__,
                       cuda=th.version.cuda,cudnn=th.backends.cudnn.version()),pools=provenance),indent=2)+'\n')
    (output/'audit.json').write_text(json.dumps(dict(strict_controls_passed=True,statistics=stats),indent=2)+'\n')
    print(json.dumps(dict(parent=str(parent),states=sum(r['states'] for r in stats),offline_complete=True)))


def gate(plan_path):
    plan=json.loads(plan_path.read_text());out=plan_path.parent;all_stats=[];summaries=[]
    for identity in plan['reference_ids']:
        parent=ROOT/'runs'/'tc_coverage'/identity
        audit=json.loads((parent/'offline_tc/audit.json').read_text())
        if not audit['strict_controls_passed']:
            raise ValueError('strict controls failed')
        all_stats.extend(audit['statistics'])
    passed=True
    for seed in plan['seeds']:
        for level in LEVELS:
            rows=[r for r in all_stats if r['seed']==seed and r['level']==level]
            states=sum(r['states'] for r in rows);pairs=sum(r['possible_pairs'] for r in rows)
            summaries.append(dict(seed=seed,level=level,states=states,
                 micro_tc=sum(r['closure_pairs'] for r in rows)/pairs,
                 macro_tc=sum(r['macro_tc']*r['states'] for r in rows)/states))
        rows=[r for r in summaries if r['seed']==seed]
        for metric in ('micro_tc','macro_tc'):
            values=[r[metric] for r in rows]
            passed &= values[1]-values[0]>=.01 and values[2]-values[1]>=.01 and values[2]-values[0]>=.05
    # No requirement on Q-gap agreement: discrepancies are retained and reported.
    write_csv(out/'offline_statistics.csv',all_stats);write_csv(out/'offline_seed_summary.csv',summaries)
    verdict=dict(passed=bool(passed),strict_controls_passed=True,
                 criterion='For each seed, pooled across four history pools: adjacent arms >=1 percentage point and High-Low >=5 points in both Micro and Macro TC. No Q-gap gate.',
                 seed_summary=summaries)
    (out/'feasibility.json').write_text(json.dumps(verdict,indent=2)+'\n')
    lines=['# TS5-1M TC offline feasibility','',verdict['criterion'],'',
           'Strict controls: identical within-seed source pools, valid states, all K1 edges, per-state N-2 extra edges, and Near/Mid/Far quotas. Near=distance 1; Mid=2; Far>=3.',
           'Q-gap is only a secondary tie break among equal closure gains. Actual distributions are retained; no gap filtering or exclusion.',
           'Direct edge coverage = |E| / choose(N,2). TC = |closure(E)| / choose(N,2). Micro pools numerators/denominators; Macro averages states. Complete order coverage = fraction of states with TC=1. Closure gain = TC minus direct coverage.',
           'States with fewer than four legal actions or exact old-Q ties are excluded identically. They are not removed to improve separation. Low action counts may have identical attainable TC.',
           'Because distance quotas reserve Mid/Far edges, complete-order coverage may be zero in all arms; that outcome cannot answer whether full ordering improves learning.',
           '', '| Seed | Arm | Micro TC | Macro TC |','|---|---|---:|---:|']
    lines += ['| {seed} | {level} | {micro_tc:.4f} | {macro_tc:.4f} |'.format(**r) for r in summaries]
    lines += ['', 'Verdict: '+('PASS; dependent training may start.' if passed else 'STOP; no formal branches may start. Inspect attainable graph structures and pool composition.'),
              '', 'See offline_statistics.csv for poolwise controls, Q-gap means/medians/std, complete-order coverage, and TC quantiles. Raw per-state distributions, masks, hashes and pool provenance remain in each reference run/offline_tc/.']
    (out/'OFFLINE_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verdict))
    if not passed:
        raise SystemExit(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent',type=Path,nargs='?');parser.add_argument('--gate',type=Path)
    args=parser.parse_args();th.set_num_threads(1)
    if args.gate:gate(args.gate)
    else:generate(args.parent)
