"""Generate K1 gap0 and all four gap0.5 masks from immutable TS5 FT pools."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch as th

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'drivers/continual'),str(ROOT/'scripts')]
from learners.pair_tc import TCPPreference, digest_masks, quotas
from learners.pair_ablation import pair_statistics
from learners.ranking_preference import reference_order,full_rank_pairs
from common.runtime import episode_spec
from components.episode_buffer import ReplayBuffer
from scripts.summarize_szc_pair_distance import selector_data

BATCH='ts5_tc_gap_extension_20261125'
OLD='ts5_tc_coverage_20261123'


def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)


def analyze(old,legal,decisions,masks,valid,counts,level,gap):
    n=legal.sum(-1);size=legal.size(-1);left,right=th.triu_indices(size,size,1)
    q,scale,_,_=reference_order(old,legal,.1)
    _,_,_,filtered=full_rank_pairs(q,legal,decisions,dict(min_gap=gap));filtered=filtered[...,left,right]
    legal_pairs=legal[...,left]&legal[...,right];strict=legal_pairs&(old[...,left]!=old[...,right])
    original=decisions&(n>=4)&~(legal_pairs&~strict).any(-1)
    order=th.sort(old.masked_fill(~legal,-th.inf),descending=True,stable=True).indices
    top=(left==order[...,:1])|(right==order[...,:1])
    base=filtered&top
    ranks=order.argsort(-1);distance=(ranks[...,left]-ranks[...,right]).abs();bucket=(distance-1).clamp(0,2)
    expected=original.clone()
    capacity=th.stack([(filtered&~top&(bucket==k)).sum(-1) for k in range(3)],-1)
    if gap:
        expected &= base.any(-1)
        for count in range(4,size+1):
            for k,quota in enumerate(quotas(count)):expected &= (n!=count)|(capacity[...,k]>=int(quota))
    assert th.equal(valid,expected),'common state set mismatch'
    assert not (masks&~filtered).any(),'selected pair fails standard gap filter'
    base &= expected[...,None]
    assert th.equal(masks&top,base),'K1 base mismatch'
    extra=masks&~top;extras=extra.sum(-1)
    assert th.equal(extras[valid],(th.zeros_like(n) if level=='k1' else n-2)[valid]),'extra budget mismatch'
    qc=th.stack([(extra&(bucket==k)).sum(-1) for k in range(3)],-1)
    for count in n[valid].unique().tolist():
        target=np.zeros(3,dtype=int) if level=='k1' else quotas(count)
        assert np.array_equal(qc[valid&(n==count)].numpy(),np.broadcast_to(target,(int((valid&(n==count)).sum()),3))), 'quota mismatch'
    # Independent Floyd-Warshall audit of directed original-action edges.
    reach=np.zeros((*masks.shape[:-1],size,size),dtype=bool)
    direction=(old[...,left]>old[...,right]).numpy()
    reach[...,left.numpy(),right.numpy()]=masks.numpy()&direction
    reach[...,right.numpy(),left.numpy()]=masks.numpy()&~direction
    for k in range(size):reach |= reach[..., :,k:k+1]&reach[..., k:k+1,:]
    closure=reach.sum((-1,-2));den=(n*(n-1)//2)[valid].numpy();tc=closure[valid.numpy()]/den
    selected=masks.sum(-1)[valid].numpy();states=int(valid.sum())
    gaps=(q[...,left]-q[...,right]).abs().numpy()
    def distribution(prefix,values):
        return {prefix+'_'+name:float(fn(values)) if values.size else None for name,fn in (
            ('mean',np.mean),('median',np.median),('std',np.std),('min',np.min),('max',np.max))}
    original_counts=original.sum()
    row=dict(original_states=int(original_counts),valid_states=states,excluded_states=int(original_counts)-states,
       retained_fraction=states/int(original_counts) if original_counts else None,
       direct_pairs=int(masks.sum()),possible_pairs=int(den.sum()),closure_pairs=int(closure[valid.numpy()].sum()),
       base_pairs=int(base.sum()),extra_pairs=int(extra.sum()),complete_order_states=int((tc==1).sum()),
       direct_coverage=float(selected.sum()/den.sum()) if states else None,
       micro_tc=float(closure[valid.numpy()].sum()/den.sum()) if states else None,macro_tc=float(tc.mean()) if states else None,
       closure_gain=float((closure[valid.numpy()].sum()-selected.sum())/den.sum()) if states else None,
       complete_order_coverage=float((tc==1).mean()) if states else None,
       **{name:int((masks&(bucket==k)).sum()) for k,name in enumerate(('near_pairs','mid_pairs','far_pairs'))},
       **{name:int(qc[...,k].sum()) for k,name in enumerate(('extra_near','extra_mid','extra_far'))},
       **distribution('qgap',gaps[masks.numpy()]),**distribution('base_qgap',gaps[base.numpy()]),
       **distribution('extra_qgap',gaps[extra.numpy()]))
    return row,base,closure,gaps,original


def generate(out):
    out.mkdir(parents=True,exist_ok=False)
    statistics=[];provenance=[];histogram=[];gate_pass=True
    for seed in (2222,3535):
        parent=ROOT/'runs/tc_coverage'/('%s_reference_seed%d'%(OLD,seed))
        info=read(parent/'env_info.json');scheme,groups,pre=episode_spec(info)
        scheme=ReplayBuffer(scheme,groups,1,2,preprocess=pre).scheme
        source_hashes=[]
        for phase in range(4):
            source=parent/'action_set_memory'/('phase%04d'%phase)/'success.th'
            pool=th.load(source,map_location='cpu');old,legal,decisions=selector_data(pool['episodes'])
            source_hashes.append(dict(phase=phase,pool=str(source),pool_sha256=sha(source)))
            for gap in (0,.5):
                controls=[]
                for level in ('k1',) if gap==0 else ('k1','low','medium','high'):
                    pref=TCPPreference(level,dict(weight=1.,temperature=1.,min_gap=gap,q_scale_floor=.1,pair_seed=3535,batch_episodes=8),np.random.default_rng(seed),scheme,groups,lambda r:None)
                    prepared,audit=pref.prepare_pool(pool,prior_budget=0)
                    masks=th.zeros((*old.shape[:-1],old.size(-1)*(old.size(-1)-1)//2),dtype=th.bool)
                    counts=th.zeros((*old.shape[:-1],3),dtype=th.long)
                    for i,ep in enumerate(prepared):
                        masks[i,:ep['length']]=ep['pair_mask'];counts[i,:ep['length']]=ep['pair_counts']
                    valid=counts[...,0]>0
                    row,base,closure,gaps,original=analyze(old,legal,decisions,masks,valid,counts,level,gap)
                    assert audit['closure_pairs']==row['closure_pairs']
                    base_masks=[base[i,:ep['length']] for i,ep in enumerate(prepared)]
                    valid_masks=[valid[i,:ep['length']] for i,ep in enumerate(prepared)]
                    base_digest=digest_masks(base_masks);valid_digest=digest_masks(valid_masks)
                    if gap==0:
                        old_offline=th.load(parent/'offline_tc/low'/('phase%04d.th'%phase),map_location='cpu')
                        assert valid_digest==old_offline['audit']['valid_state_sha256'],'gap0 effective states differ from existing TC'
                        old_masks=old_offline['masks'];l,r=th.triu_indices(old.size(-1),old.size(-1),1)
                        order=th.sort(old.masked_fill(~legal,-th.inf),descending=True,stable=True).indices
                        top=(l==order[...,:1])|(r==order[...,:1])
                        assert all(th.equal(ep['pair_mask'],old_masks[i]&top[i,:ep['length']]) for i,ep in enumerate(prepared))
                    identity=dict(seed=seed,min_gap=gap,phase=phase,task=pool['task'],level=level)
                    row=dict(identity,**row,valid_state_sha256=valid_digest,base_mask_sha256=base_digest,pair_mask_sha256=audit['pair_mask_sha256'])
                    statistics.append(row);gate_pass &= row['valid_states']>0
                    controls.append((valid_digest,base_digest,row['base_pairs']))
                    path=out/'masks'/('seed%d'%seed)/('gap0' if gap==0 else 'gap05')/level
                    path.mkdir(parents=True,exist_ok=True)
                    th.save(dict(masks=[e['pair_mask'] for e in prepared],counts=[e['pair_counts'] for e in prepared],audit=audit,source_sha256=sha(source)),path/('phase%04d.th'%phase))
                    # Exact state TC histogram, including action count and directed pair counts.
                    counter={}
                    for direct,possible,reached in zip(masks.sum(-1)[valid].tolist(),counts[...,0][valid].tolist(),closure[valid.numpy()].tolist()):
                        key=(int(direct),int(possible),int(reached));counter[key]=counter.get(key,0)+1
                    histogram += [dict(identity,direct_pairs=k[0],possible_pairs=k[1],closure_pairs=k[2],states=v,tc=k[2]/k[1]) for k,v in counter.items()]
                    np.savez_compressed(path/('phase%04d_distributions.npz'%phase),qgap=gaps[masks.numpy()],base_qgap=gaps[base.numpy()],extra_qgap=gaps[(masks&~base).numpy()],tc=closure[valid.numpy()]/counts[...,0][valid].numpy())
                assert len(set(controls))==1,'four methods differ in base pairs or effective states'
        provenance.append(dict(seed=seed,parent=str(parent),return_state_sha256=sha(parent/'return_state.th'),pools=source_hashes))
    write(out/'offline_statistics.csv',statistics);write(out/'tc_state_histogram.csv',histogram)
    (out/'source_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    seed_summary=[]
    for seed in (2222,3535):
        for gap in (0,.5):
            for level in ('k1',) if gap==0 else ('k1','low','medium','high'):
                rows=[r for r in statistics if r['seed']==seed and r['min_gap']==gap and r['level']==level]
                states=sum(r['valid_states'] for r in rows);pairs=sum(r['possible_pairs'] for r in rows)
                seed_summary.append(dict(seed=seed,min_gap=gap,level=level,states=states,
                    micro_tc=sum(r['closure_pairs'] for r in rows)/pairs if pairs else None,
                    macro_tc=sum(r['macro_tc']*r['valid_states'] for r in rows)/states if states else None))
    write(out/'offline_seed_summary.csv',seed_summary)
    (out/'feasibility.json').write_text(json.dumps(dict(passed=bool(gate_pass),strict_controls_passed=True,
        reason='All eight historical pools have nonempty common gap05 states with unchanged N-2 extra budgets and original Near/Mid/Far quotas. Gap0 K1 uses exactly the old valid states. No threshold, quota or budget relaxation.',seed_summary=seed_summary),indent=2)+'\n')
    lines=['# TC K1 / min_gap extension offline report','',
       'Standard filter reuses reference_order + full_rank_pairs: population std of legal old Q (floor 0.1), non-tied upper-triangle legal pairs, abs(standardized difference) >= min_gap. All selected edges, including K1, use the same standard. This leaves fewer than N-1 K1 pairs at gap05 in some states; no filtered pair is restored.',
       'Gap0 K1 uses the same old valid states and exact N-1 old Top-1 edges. Gap05 common states are the old valid population intersected with at least one surviving K1 edge and sufficient threshold-passing non-Top-1 capacity in EACH original distance bucket. K1-only uses that same common set although it needs no extras. No silent threshold or budget change.',
       'Counts and Q-gap distributions: offline_statistics.csv. TC = directed transitive closure / choose(N,2); Micro pools pairs, Macro averages states. Closure Gain = Micro TC minus Direct Coverage.',
       '', '| Seed | Phase | Old states | Gap05 common | Retained % |','|---|---:|---:|---:|---:|']
    lines += ['| %d | %d | %d | %d | %.2f |'%(r['seed'],r['phase'],r['original_states'],r['valid_states'],100*r['retained_fraction']) for r in statistics if r['min_gap']==.5 and r['level']=='k1']
    lines += ['', '| Seed | Gap | Arm | Micro TC | Macro TC |','|---|---:|---|---:|---:|']
    lines += ['| {seed} | {min_gap} | {level} | {micro_tc:.5f} | {macro_tc:.5f} |'.format(**r) for r in seed_summary]
    lines += ['', 'PASS: training may start.' if gate_pass else 'STOP: formal training remains unsubmitted.',
        'Gap0 and gap05 use different common populations; cross-gap performance differences combine gap screening and state selection. Seed 4646 is preserved only in the original dataset and excluded from this two-seed comparison. No TC gap-size or retained-state-fraction threshold is added.']
    (out/'OFFLINE_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=bool(gate_pass),output=str(out),seed_summary=seed_summary)))
    if not gate_pass:raise SystemExit(2)


def verify(out):
    verdict=read(out/'feasibility.json')
    if not verdict['passed']:raise ValueError('Offline feasibility did not pass')
    checked=0
    for source in read(out/'source_provenance.json'):
        for pool in source['pools']:
            if sha(Path(pool['pool']))!=pool['pool_sha256']:
                raise ValueError('Historical pool changed after offline verification')
    for path in (out/'masks').glob('seed*/gap*/*/phase*.th'):
        saved=th.load(path,map_location='cpu')
        if digest_masks(saved['masks'])!=saved['audit']['pair_mask_sha256']:
            raise ValueError('Frozen mask changed after offline verification')
        checked+=1
    if checked!=40:raise ValueError('Expected 40 frozen history masks')
    print(json.dumps(dict(passed=True,verified_history_masks=checked)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'results'/BATCH)
    parser.add_argument('--verify',type=Path)
    args=parser.parse_args();th.set_num_threads(1)
    if args.verify:verify(args.verify)
    else:generate(args.output)
