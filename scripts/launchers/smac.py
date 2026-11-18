"""Launch plan for the smac suite."""
import sys
import yaml
from .common import ROOT, LaunchSpec, merge


def build(a, p, out, env, trhr, trhr_overrides):
    if not a.map or a.steps is None or a.steps<=0: p.error('smac requires --map and positive --steps')
    if a.scenario or a.method=='softprotect': p.error('softprotect currently uses the continual driver')
    suffix={'baseline':'','reborn':'_b','redo':'_r','reset':'_s','trhr':''}[a.method]
    preset=a.algorithm+suffix
    f=ROOT/'src/config/algs'/f'{preset}.yaml'
    if not f.exists():p.error('No upstream preset for this combination; see docs/METHODS.md')
    cfg={}
    for path in ['src/config/default.yaml','src/config/envs/sc2.yaml',str(f.relative_to(ROOT))]:
        merge(cfg,yaml.safe_load((ROOT/path).read_text()))
    # Immutable experiment identity is kept separate from upstream method dispatch name.
    cfg.update(seed=a.seed,t_max=a.steps,local_results_path=str(out),save_model=True)
    cfg['env_args'].update(map_name=a.map,seed=a.seed)
    for item in a.set:
        if item in trhr_overrides:
            continue
        if '=' not in item:p.error('--set requires KEY=VALUE')
        key,value=item.split('=',1);d=cfg
        if key.split('.')[0] in ['name','seed','t_max','local_results_path','unique_token','evaluate','checkpoint_path']:p.error('use the dedicated training arguments; evaluation is a separate command')
        if key in ['env_args.map_name','env_args.seed']:p.error('use --map / --seed')
        keys=key.split('.')
        for k in keys[:-1]: d=d.setdefault(k,{})
        d[keys[-1]]=yaml.safe_load(value)
    if trhr is not None:
        if (cfg['env'], cfg['mac'], cfg['agent'], cfg['learner'], cfg['mixer']) != (
            'sc2', 'basic_mac', 'rnn', 'q_learner', 'vdn'
        ):
            p.error('TRHR requires the standard SMAC VDN controller, agent and learner')
        cfg['trhr'] = trhr
    if cfg.get('a2_trace') and (
        a.method not in ('baseline', 'trhr') or
        (cfg['env'], cfg['mac'], cfg['agent'], cfg['learner'], cfg['mixer'], cfg['runner'])
        != ('sc2', 'basic_mac', 'rnn', 'q_learner', 'vdn', 'episode')
    ):
        p.error('a2_trace requires static SMAC two-layer VDN baseline or TRHR')
    # Supply a fully merged config to Sacred through a simple dedicated entrypoint.
    command=[sys.executable,str(ROOT/'scripts/smac_entry.py'),str(out/'config.json')]
    details={'config':cfg,'preset':preset}
    return LaunchSpec(command, env, details)
