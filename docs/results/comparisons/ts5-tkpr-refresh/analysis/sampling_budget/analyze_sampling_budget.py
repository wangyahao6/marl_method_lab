# -*- coding: utf-8 -*-
"""Fixed checkpoint/pool TS5 TKPR sampling-budget probe; CPU only, no environments.
From repository root: PYTHONPATH=src:. python <this-script> --repo .
Only analysis output files are written. Agent parameters and input artifacts stay unchanged.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th

BUDGETS = ('K1', 'Refresh-2', 'Refresh-4', 'Refresh-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('.'))
    parser.add_argument('--seed', type=int, default=86431)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    root = args.repo.resolve()
    sys.path[:0] = [str(root / 'src'), str(root)]
    from controllers import REGISTRY
    from components.episode_buffer import ReplayBuffer
    from drivers.continual.common.runtime import episode_spec
    from drivers.continual.memory import PreferenceHistory
    from learners.ranking_preference import RankingPreference
    from learners.action_set_preference import ActionSetPreference

    out = root / 'docs/results/comparisons/ts5-tkpr-refresh/analysis/sampling_budget'
    out.mkdir(parents=True, exist_ok=True)
    run = root / 'runs/ts5_3535_tkpr_k1_20261122_attempt001'
    read = lambda p: json.loads(p.read_text())
    cfg = read(run / 'config.json')
    cfg.update(device='cpu', use_cuda=False)
    th.set_num_threads(2)
    th.manual_seed(args.seed)
    spec, groups, preprocess = episode_spec(read(run / 'env_info.json'))
    scheme = ReplayBuffer(spec, groups, 1, 2, preprocess=preprocess).scheme
    mac = REGISTRY[cfg['mac']](scheme, groups, SimpleNamespace(**cfg))
    mac.agent.eval()
    mac.init_hidden(1)
    parameters = tuple(p for p in mac.agent.parameters() if p.requires_grad)
    preference = RankingPreference('topk_preference', cfg['topk_preference'], None, scheme, groups, None)
    assert cfg['topk_preference']['k'] == 1
    assert cfg['topk_preference']['weight'] == 1

    def save_csv(name, rows):
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (out / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1048576), b''):
                h.update(chunk)
        return h.hexdigest()

    def state_hash():
        h = hashlib.sha256()
        for name, tensor in mac.agent.state_dict().items():
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def grad(loss):
        derivatives = th.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        return th.cat([(g if g is not None else th.zeros_like(p)).detach().reshape(-1).double()
                       for p, g in zip(parameters, derivatives)])

    def cosine(x, y):
        norm = float(x.norm() * y.norm())
        return float((x.dot(y) / norm).clamp(-1, 1)) if norm else None

    def aggregation_weights(counts):
        leaves = [th.zeros((), requires_grad=True) for _ in counts]
        scalar = ActionSetPreference.reduce_task_losses(leaves, th.tensor(counts))
        return [float(w) for w in th.autograd.grad(scalar, leaves)]

    pools = {}
    paths = [run / 'config.json', run / 'protocol.json', run / 'env_info.json']
    for phase in range(4):
        path = run / f'action_set_memory/phase{phase:04d}/success.th'
        pool = th.load(path, map_location='cpu')
        assert pool['task'] == 'ABCDE'[phase] and pool['phase'] == phase
        assert len(pool['episodes']) == 100
        assert pool['reference_checkpoint'].startswith(f'models/phase{phase:04d}_')
        for episode in pool['episodes']:
            data = episode['data']
            assert data['old_q_values'].shape == data['avail_actions'].shape
            for key in ('obs', 'actions_onehot', 'filled', 'terminated', 'old_q_values'):
                assert len(data[key]) == episode['length'] + 1
        pools[pool['task']] = pool
        paths.append(path)
    checkpoints = {int(p.parent.name[5:9]): p for p in run.glob('models/*/agent.th')}
    assert all(phase in checkpoints for phase in (2, 3, 4))
    paths += [checkpoints[phase] for phase in (2, 3, 4)]
    inputs = [dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=sha(path))
              for path in paths]
    manifest = dict(run_id=run.name, training_commit='9db0ed1951ded62c00eb1ff27a80c71ae5465dc5',
                    analysis_base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                    seed=args.seed, repeats=args.repeats, python=sys.version, torch=th.__version__,
                    numpy=np.__version__, device='cpu',threads=2, settings=cfg['topk_preference'],
                    gradient_parameters=[dict(name=n, shape=list(p.shape)) for n,p in mac.agent.named_parameters() if p.requires_grad],
                    files=inputs)
    (out / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    raw, full_rows, summaries, paired, checks = [], [], [], [], []
    for phase in (2, 3, 4):
        mac.agent.load_state_dict(th.load(checkpoints[phase], map_location='cpu'))
        before = state_hash()
        history = PreferenceHistory({i: pools['ABCDE'[i]] for i in range(phase)})
        tasks = list('ABCDE'[:phase])
        # Full 100-episode task gradient, with original task loss; independent whole-sequence RNN.
        refs, ref_counts, ref_pairs, ref_losses = {}, {}, {}, {}
        for task in tasks:
            loss, count, stats = preference.task_loss(mac, pools[task], pools[task]['episodes'])
            stat = preference.statistics(stats.detach().tolist())
            refs[task] = grad(loss)
            ref_counts[task] = int(count)
            ref_pairs[task] = stat['valid_pairs']
            ref_losses[task] = float(loss.detach())
            full_rows.append(dict(checkpoint_phase=phase, scope=task, episodes=100,
                                  loss=float(loss.detach()), valid_states=int(count), valid_pairs=stat['valid_pairs'],
                                  grad_norm=float(refs[task].norm())))
            del loss, count, stats
        full_weights = aggregation_weights([ref_counts[t] for t in tasks])
        refs['ALL'] = sum(refs[t] * w for t,w in zip(tasks,full_weights))
        ref_counts['ALL'] = sum(ref_counts[t] for t in tasks)
        ref_pairs['ALL'] = sum(ref_pairs[t] for t in tasks)
        full_rows.append(dict(checkpoint_phase=phase,scope='ALL',episodes=100*phase,
                              loss=sum(ref_losses[t]*w for t,w in zip(tasks,full_weights)),
                              valid_states=ref_counts['ALL'],valid_pairs=ref_pairs['ALL'],grad_norm=float(refs['ALL'].norm())))
        vectors = defaultdict(list)
        # Separate independent K1 stream and paired nested Refresh stream. Other phases independent.
        k1_rng = np.random.default_rng(np.random.SeedSequence([args.seed, phase, 0]))
        refresh_rng = np.random.default_rng(np.random.SeedSequence([args.seed, phase, 1]))
        for repeat in range(args.repeats):
            nested = {task: refresh_rng.choice(100, 8, replace=False).tolist() for task in tasks}
            k1 = {pool['task']: ids for pool,ids in history.sample('ABCDE'[phase],8,k1_rng)}
            sample_vectors = {}
            for budget in BUDGETS:
                size = None if budget == 'K1' else int(budget.rsplit('-',1)[1])
                ids_by_task = k1 if budget=='K1' else {t:nested[t][:size] for t in tasks}
                losses, counts, stats, gradients = [], [], {}, {}
                for task in tasks:
                    selected = [pools[task]['episodes'][i] for i in ids_by_task[task]]
                    loss,count,stat = preference.task_loss(mac,pools[task],selected)
                    losses.append(loss);counts.append(count)
                    stats[task] = preference.statistics(stat.detach().tolist())
                    gradients[task] = grad(loss)
                combined_loss = ActionSetPreference.reduce_task_losses(losses,th.stack(counts))
                gradients['ALL'] = grad(combined_loss)
                weights = aggregation_weights([int(c) for c in counts])
                reconstruction = sum(gradients[t]*w for t,w in zip(tasks,weights))
                error = float((gradients['ALL']-reconstruction).abs().max())
                assert error<=2e-6, 'Task aggregation derivative mismatch'
                checks.append(dict(checkpoint_phase=phase,budget=budget,repeat=repeat,
                                   aggregation_max_abs_error=error,parameters_unchanged=state_hash()==before,
                                   nested_2_4_8=True))
                for scope in tasks+['ALL']:
                    is_all = scope=='ALL'
                    valid = sum(int(c) for c in counts) if is_all else int(counts[tasks.index(scope)])
                    pairs = sum(stats[t]['valid_pairs'] for t in tasks) if is_all else stats[scope]['valid_pairs']
                    decisions = sum(stats[t]['decision_states'] for t in tasks) if is_all else stats[scope]['decision_states']
                    participation = sum(int(c)>0 for c in counts)/phase if is_all else int(valid>0)
                    v,ref=gradients[scope],refs[scope]
                    vectors[budget,scope].append(v)
                    raw.append(dict(checkpoint_phase=phase,phase_label='ABCDE'[phase],budget=budget,repeat=repeat,scope=scope,
                                    loss=float(combined_loss.detach()) if is_all else float(losses[tasks.index(scope)].detach()),
                                    sampled_episodes=sum(len(x) for x in ids_by_task.values()) if is_all else len(ids_by_task[scope]),
                                    episode_ids=json.dumps(ids_by_task if is_all else ids_by_task[scope],sort_keys=True),
                                    grad_norm=float(v.norm()),cosine_full=cosine(v,ref),relative_gradient_error=float((v-ref).norm()/ref.norm()),
                                    decision_states=decisions,valid_states=valid,valid_pairs=pairs,
                                    valid_state_fraction=valid/decisions if decisions else None,
                                    state_coverage_of_full=valid/ref_counts[scope],pair_coverage_of_full=pairs/ref_pairs[scope],
                                    task_participation=participation,no_valid_constraint_fraction=1-participation,
                                    full_grad_norm=float(ref.norm()),task_weight=None if is_all else weights[tasks.index(scope)]))
                sample_vectors[budget]=gradients['ALL']
                del losses,counts,combined_loss,gradients
            for left,right in itertools.combinations(BUDGETS,2):
                c=cosine(sample_vectors[left],sample_vectors[right])
                paired.append(dict(checkpoint_phase=phase,repeat=repeat,budget_a=left,budget_b=right,
                                   cosine=c,angle_degrees=math.degrees(math.acos(c)) if c is not None else None,
                                   cosine_full_delta=cosine(sample_vectors[right],refs['ALL'])-cosine(sample_vectors[left],refs['ALL'])))
        assert state_hash()==before
        for (budget,scope),vecs in vectors.items():
            group=[r for r in raw if r['checkpoint_phase']==phase and r['budget']==budget and r['scope']==scope]
            row=dict(checkpoint_phase=phase,phase_label='ABCDE'[phase],budget=budget,scope=scope,repeats=len(group))
            for metric in ('grad_norm','cosine_full','relative_gradient_error','decision_states','valid_states','valid_pairs',
                           'state_coverage_of_full','pair_coverage_of_full','task_participation','no_valid_constraint_fraction','task_weight','sampled_episodes'):
                vals=[r[metric] for r in group if r[metric] is not None]
                row[metric+'_mean']=float(np.mean(vals)) if vals else None
                row[metric+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
            matrix=th.stack(vecs);center=matrix.mean(0);ref=refs[scope]
            variance=float(((matrix-center).square().sum(1)).sum()/(len(vecs)-1))
            row.update(gradient_variance_trace=variance,normalized_gradient_variance_trace=variance/float(ref.square().sum()),
                       relative_bias_norm=float((center-ref).norm()/ref.norm()),
                       mean_relative_squared_error=float((matrix-ref).square().sum(1).mean()/ref.square().sum()))
            summaries.append(row)
        print('phase',phase,'complete; full norms',[(t,float(refs[t].norm())) for t in tasks+['ALL']],flush=True)
    save_csv('sampling_budget.csv',raw)
    save_csv('sampling_budget_summary.csv',summaries)
    save_csv('full_pool_reference.csv',full_rows)
    save_csv('paired_directions.csv',paired)
    save_csv('verification_checks.csv',checks)
    assert all(r['parameters_unchanged'] for r in checks)
    for r in manifest['files']:
        assert sha(root/r['path'])==r['sha256'], 'Original input changed'
    print('COMPLETE',len(raw),'raw rows;',len(checks),'aggregation/parameter checks',flush=True)


if __name__ == '__main__':
    main()
