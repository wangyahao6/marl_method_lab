"""Static matrices and descriptive report for fixed-network comparisons."""
import itertools,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_and_report(out,manifest,missing,errors,reps,outputs,summary):
    for metric,rows,fname in [('linear_cka',reps,'representation_similarity'),('greedy_agreement',outputs,'decision_agreement')]:
        layers=['A1','A2'] if metric=='linear_cka' else ['decision']
        for layer in layers:
            fig,axes=plt.subplots(3,3,figsize=(19,17))
            for r,target in enumerate('SZC'):
                prior=[c for c in 'SZC' if c!=target]
                paths=[target]+[p+target for p in prior]+[''.join(p)+target for p in itertools.permutations(prior)]+prior+[''.join(p) for p in itertools.permutations(prior)]
                for c,probe in enumerate('SZC'):
                    ax=axes[r,c];m=np.full((len(paths),len(paths)),np.nan);np.fill_diagonal(m,1);counts=[]
                    for i,j in itertools.combinations(range(len(paths)),2):
                        vals=[row[metric] for row in rows if row['target']==target and row['probe']==probe and row['path_a']==paths[i] and row['path_b']==paths[j] and (metric!='linear_cka' or row['layer']==layer) and row[metric] is not None]
                        if vals:m[i,j]=m[j,i]=np.mean(vals);counts.append(len(vals))
                    im=ax.imshow(np.ma.masked_invalid(m),vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(len(paths)));ax.set_xticklabels(paths,rotation=60);ax.set_yticks(range(len(paths)));ax.set_yticklabels(paths)
                    ax.set_title(f'Target {target}, probe {probe}; n={min(counts) if counts else 0}-{max(counts) if counts else 0} seeds')
            fig.suptitle(f'{layer}: {metric} (equal-weight seed means; diagonal = identity)',fontsize=17);fig.tight_layout(rect=[0,0,.94,.96]);fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.02,pad=.02)
            fig.savefig(out/f'{fname}_{layer}.png',dpi=150);plt.close(fig)
    lines=['# Independent experiment: VDN Agent network similarity','',f'Frozen snapshot: {len(manifest)}/45 completed endpoint models. Unavailable at snapshot: {len(missing)}; analysis errors: {len(errors)}. Exact model hashes: manifest.json; fixed input hashes/counts: probes.json. No new training or environment interaction.','', '## Design','', 'Target T in S/Z/C. Nine model roles: T; P->T and Q->T; P->Q->T and Q->P->T; P, Q, P->Q, Q->P before learning T. All within-seed pairs reported. Target exposure is2M for target-trained models; total budgets are2/4/6M. The two prior orders at stage3 have matched total budget.','', 'Three fixed probes per seed: last32 completed training episodes from scratch S/Z/C endpoints. Each compared model replays the SAME observations, recorded previous actions, agent IDs, episode ordering and masks; GRU starts at zero each episode and evolves with that model. This is teacher-forced history replay, not independent on-policy evaluation. No model/probe regeneration.','', 'Samples are valid agent decision-time rows, excluding padding, bootstrap states and dead/forced no-op agents (no available action beyond no-op). CKA pools agent/time rows on each probe separately. Action agreement further restricts to states with more than one legal action. No pooled row-level significance inference.','', '## Metrics','', 'Primary: linear centered CKA of A1=net[2] and A2=net[4] outputs. Center feature columns; use ||X^T Y||_F^2/(||X^T X||_F ||Y^T Y||_F). Undefined zero-variance cases remain empty.','', 'Parameters: flatten incoming A1 net.0 and incoming A2 net.3 separately, GRU, Q head and full used network; cosine and symmetric normalized L2 = ||a-b||/sqrt((||a||^2+||b||^2)/2). Agent fc1 is defined but unused in actual forward and excluded; full used network includes the A1 gate. These groups do not claim neuron alignment.','', 'Outputs: legal greedy action agreement, raw legal Q RMSE and per-state legal-action-mean-centered Q RMSE. Agreement reports actual legal choices under shared histories, not win-rate equivalence.','', 'Each seed is one training replicate. seed_summary.csv gives equal-weight seed mean/sample std/n per comparison; shared prefixes/probes and reused pair rows are not independent replicates. No cross-seed neuron index pairing.','', '## Selected comparisons on target-task probes','', '| Target | Models | Layer | Mean CKA | Seeds |','|---|---|---|---:|---:|']
    for target in 'SZC':
        for row in summary:
            if row['family']=='representations' and row['target']==target and row['condition'].startswith(target+'/') and row['path_a']==target and row['path_b'].endswith(target):
                lines.append(f"| {target} | {row['path_a']} vs {row['path_b']} | {row['condition'].split('/')[-1]} | {row['mean']:.4f} | {row['n_seeds']} |" if row['mean'] is not None else f"| {target} | {row['path_a']} vs {row['path_b']} | - | undefined | 0 |")
    lines+=['','## Interpretation limits','', 'A higher CKA indicates more similar linear representation geometry on the stated probe, not equal neuron functions or causal task knowledge. Weight distance, representation similarity and decision similarity answer different questions. Scratch vs transfer differs in total updates, replay and exploration; prior-only checkpoints differ in latest task. Probe dependence and single-behavior-source bias remain; fixed-probe comparisons control input but do not establish environmental performance. No causal protection/forgetting conclusions. Missing triples limit seed coverage and are not treated as completed.','', 'Artifacts: parameters.csv, representations.csv, outputs.csv, coverage.csv, seed_summary.csv, manifest.json, probes.json, errors.json, implementation_checks.json and three PNG matrices. Existing training results and models were read only.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
