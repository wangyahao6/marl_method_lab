"""Exact-roster sequence training with shared prefixes and optional soft protection."""
import argparse,copy,json,logging,os,random,time,traceback
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import hetero_train as h
from hetero_environment import TEAMS
from hetero_protection import select as qmix_select
from ts3_vdn_detection import select as vdn_select
from ts3_vdn_type_detection import select as vdn_type_select
from ts3_protection import TS3Protection
import vdn_chi
from protection import cpu_episode,rng_state,digest
import numpy as np
import torch as th
import yaml
from pysc2.lib.protocol import ConnectionError as SC2ConnectionError
from pysc2.lib.remote_controller import ConnectError as SC2ConnectError, RequestError as SC2RequestError
from s2clientprotocol import sc2api_pb2 as sc_pb
base=h.base; ROOT=base.ROOT; write=base.write; append=base.append
P=base.PROTOCOL
EXTENDED = P['id'] in ('ts5', 'ts6', 'ts7')
FACTORIAL = P.get('intervention') == 'single_ff_trhr_factorial'
CHI = P.get('intervention') == 'vdn_chi'
CHI_COMPARISON = P.get('chi_comparison', False)
SINGLE_FF_CONTROL = P.get('intervention') == 'single_ff_softprotect'
if not EXTENDED:
    TEAMS.update({key:tuple(value) for key,value in P['tasks'].items()})

def select_scope(result):
    if FACTORIAL or CHI_COMPARISON:
        item = result['A']
        scores = item.get('scores')
        item['sets']['overall'] = None if scores is None else [
            i for i, score in enumerate(scores) if score > P['threshold_beta']
        ]
        item['selection'] = 'overall_threshold_union'
        item['defined'] = scores is not None
    elif P.get('selection') == 'overall_threshold_union':
        layer = result['A2']
        chosen = layer['sets']['overall_threshold']
        layer['sets']['overall'] = chosen
        layer['selection'] = 'overall_threshold_union'
        layer['defined'] = chosen is not None
    return result

def sequence_metrics(order, sources):
    """Join immutable prefix and current evaluation rows, then calculate stage metrics."""
    stages, initial, before, curves = {}, {}, {}, {}
    for source, labels in sources:
        for line in (source/'evaluations.jsonl').read_text().splitlines():
            row = json.loads(line)
            task, label, win_rate = (row[k] for k in ('task', 'label', 'win_rate'))
            if label not in labels:
                continue
            if label == 'stage_end':
                stages.setdefault(order[row['phase']], {})[task] = win_rate
            elif label == 'initial_reference':
                initial[task] = win_rate
            elif label == 'next_unseen':
                before[task] = win_rate
            if label in ('phase_start', 'periodic', 'stage_end') and task == order[row['phase']]:
                curves.setdefault(task, {})[row['t_env']] = win_rate
    metrics = {}
    for i, task in enumerate(order):
        row = stages.get(task, {})
        seen, old = order[:i+1], order[:i]
        acc = sum(row[j] for j in seen) / len(seen) if all(j in row for j in seen) else None
        bwt = [row[j] - stages[j][j] for j in old if j in row and j in stages.get(j, {})]
        fwt_tasks = [j for j in old + [task] if j != order[0] and j in initial and j in before]
        fwt = [before[j] - initial[j] for j in fwt_tasks]
        metrics[task] = dict(
            acc_percent=None if acc is None else 100 * acc,
            bwt_pp=100 * sum(bwt) / len(bwt) if bwt and len(bwt) == len(old) else None,
            fwt_pp=100 * sum(fwt) / len(fwt) if fwt and len(fwt) == i else None,
            fwt_tasks=fwt_tasks)
        if FACTORIAL:
            points = sorted(curves.get(task, {}).items())
            metrics[task]['old_mean_percent'] = (100 * sum(row[j] for j in old) / len(old)
                if old and all(j in row for j in old) else None)
            metrics[task]['learning_auc_percent'] = (
                100 * float(np.trapz([y for _, y in points], [x for x, _ in points]))
                / (points[-1][0] - points[0][0]) if task in row and len(points) > 1 else None)
            metrics[task]['learning_curve'] = points
    return dict(stage_win_rates=stages, initial_reference=initial, before_first_training=before,
                metrics=metrics, sources=[str(source/'evaluations.jsonl') for source, _ in sources])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['prefix','branch','eval_fwt','resume','full'],required=True)
    parser.add_argument('--parent');parser.add_argument('--group',choices=P['groups'],default='N');parser.add_argument('--seed',type=int,required=True);o=parser.parse_args()
    out=Path(os.environ['MARL_RUN_DIR']);started=time.time();runner=None;updates=episode=phase=0
    track_retention = EXTENDED or FACTORIAL or CHI_COMPARISON or o.role == 'full'
    try:
        th.set_num_threads(1);random.seed(o.seed);np.random.seed(o.seed);th.manual_seed(o.seed);th.cuda.manual_seed_all(o.seed)
        cfg=h.config(o.seed,out)
        algorithm=P.get('algorithm','qmix')
        detection = P['single_ff_detection'] if o.role == 'full' and not CHI else None
        if algorithm != 'qmix':
            base.merge(cfg, yaml.safe_load((ROOT/'src/config/algs'/f'{algorithm}.yaml').read_text()))
        cfg['algorithm']=algorithm;cfg['top_k']=P.get('top_k')
        if FACTORIAL:
            cfg.update(a2_trace=True, trace_stages=len(P['order']),
                       state_weighted_screening=P.get('state_weighted_screening', True))
            if o.group in ('TRHR', 'A_union_TRHR'):
                cfg['trhr'] = P['trhr_settings']
        elif 'trhr' in P:
            cfg['trhr'] = P['trhr']
        cfg.update(t_max=P['thresholds'][-1],protocol_id=P['id'],role=o.role,parent=o.parent,group=o.group,screening=P['screening'],phase_boundary=P['phase_boundary'],tasks=P['tasks'],thresholds=P['thresholds'])
        if SINGLE_FF_CONTROL:
            cfg['intervention'] = 'single_ff_softprotect'
        if CHI:
            cfg.update(intervention='vdn_chi', chi_probe_episodes=vdn_chi.PROBE_EPISODES,
                       chi_threshold=P.get('threshold_beta'))
        if 'gem_score_threshold' in P:
            cfg['gem_score_threshold'] = P['gem_score_threshold']
        if 'sc2_game_version' in P:
            cfg['env_args']['game_version'] = P['sc2_game_version']
        if EXTENDED:
            cfg['env_args']['rosters'] = P['tasks']
        args=SimpleNamespace(**cfg)
        logging.basicConfig(level=logging.INFO);logger=base.Logger(logging.getLogger(out.name));original=logger.log_stat
        def log(k,v,t,**kwargs):append(out/'metrics.jsonl',dict(key=k,value=float(v),t_env=int(t)));original(k,v,t,**kwargs)
        logger.log_stat=log
        runner=base.EpisodeRunner(args,logger);info=runner.get_env_info();args.n_agents=info['n_agents'];args.n_actions=info['n_actions'];args.state_shape=info['state_shape']
        write(out/'config.json',vars(args));write(out/'env_info.json',info)
        scheme=dict(state=dict(vshape=info['state_shape']),obs=dict(vshape=info['obs_shape'],group='agents'),actions=dict(vshape=(1,),group='agents',dtype=th.long),avail_actions=dict(vshape=(info['n_actions'],),group='agents',dtype=th.int),reward=dict(vshape=(1,)),terminated=dict(vshape=(1,),dtype=th.uint8))
        if FACTORIAL:
            scheme['source_phase'] = dict(vshape=(1,), dtype=th.long, episode_const=True)
        groups=dict(agents=args.n_agents);preprocess=dict(actions=('actions_onehot',[base.OneHot(out_dim=args.n_actions)]))
        buffer=base.ReplayBuffer(scheme,groups,args.buffer_size,info['episode_limit']+1,preprocess=preprocess,device='cpu')
        mac=base.controllers[args.mac](buffer.scheme,groups,args);runner.setup(scheme,groups,preprocess,mac);learner=base.learners[args.learner](mac,buffer.scheme,logger,args)
        for network in [mac.agent,learner.target_mac.agent,learner.mixer,learner.target_mixer]:
            for module in network.modules():module._forward_hooks.clear()
        learner.cuda();selection={}
        gem_mode=P.get('intervention')=='a2_local_gem'
        initial_agent={k:v.detach().cpu().clone() for k,v in mac.agent.state_dict().items()} if gem_mode or (track_retention and o.role in ('prefix', 'resume', 'full')) else None
        if o.role=='branch':
            parent=Path(o.parent);state=th.load(parent/'prefix_state.th',map_location='cpu')
            for network,key in [(mac.agent,'agent'),(learner.mixer,'mixer'),(learner.target_mac.agent,'target_agent'),(learner.target_mixer,'target_mixer')]:network.load_state_dict(state[key])
            learner.optimiser.load_state_dict(state['optimizer']);buffer=state['buffer'];episode=state['episode'];updates=state['updates'];learner.last_target_update_episode=state['last_target_update_episode'];learner.log_stats_t=state['log_stats_t'];runner.t_env=state['t_env']
            mac.action_selector.epsilon=state['epsilon'];runner.env.restore_distribution(state['environment_distributions']);h.restore_rng(state['rng'])
            if 'trhr' not in P and not CHI:
                selection_path = ROOT / P['prefix_selection_override'] if P.get('prefix_selection_override') else parent/'selection.json'
                selection=select_scope(json.loads(selection_path.read_text()))
                if P.get('prefix_selection_override'):
                    import hashlib
                    write(out/'prefix_selection.json', dict(selection=selection, source=str(selection_path), source_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(), parent_original=str(parent/'selection.json'), parent_original_sha256=hashlib.sha256((parent/'selection.json').read_bytes()).hexdigest()))
            write(out/'branch_restore.json',dict(parent=str(parent),t_env=runner.t_env,episode=episode,updates=updates,buffer_episodes=buffer.episodes_in_buffer,buffer_index=buffer.buffer_index,model_hash=base.parameter_hash(learner),parent_model_hash=state['model_hash'],rng_match=digest(rng_state())==digest(state['rng']),engine='new SC2 process; recoverable distribution RNG restored'))
            if FACTORIAL:
                names = [name for name, _ in mac.agent.named_parameters()]
                write(out/'branch_parameter_identity.json', dict(
                    online_matches_prefix=all(th.equal(p.detach().cpu(), state['agent'][k])
                        for k, p in mac.agent.named_parameters()),
                    target_matches_prefix=all(th.equal(p.detach().cpu(), state['target_agent'][k])
                        for k, p in learner.target_mac.agent.named_parameters()),
                    parameters=names))
            del state
        if o.role == 'resume':
            parent = Path(o.parent)
            learner.load_models(str(parent))
            state = th.load(parent/'targets.th', map_location='cpu')
            learner.target_mac.agent.load_state_dict(state['target_agent'])
            learner.target_mixer.load_state_dict(state['target_mixer'])
            runner.t_env, episode, updates = (state[k] for k in ('t_env', 'episode', 'updates'))
            interval = args.target_update_interval
            learner.last_target_update_episode = ((episode - 1) // interval) * interval
            learner.log_stats_t = runner.t_env
            mac.action_selector.epsilon = mac.action_selector.schedule.eval(runner.t_env)
            phase = P['order'].index('E')
            write(out/'checkpoint_restore.json', dict(
                checkpoint=str(parent), t_env=runner.t_env, episode=episode, updates=updates,
                last_target_update_episode_inferred=learner.last_target_update_episode,
                epsilon=mac.action_selector.epsilon, replay='empty', rng='reseeded',
                parameter_hash=base.parameter_hash(learner)))
        gem_band=o.group.split('_',1)[1] if o.group.startswith('GEM_') else None
        if FACTORIAL:
            layer, mode = ('A', 'union') if o.group in ('A_union', 'A_union_TRHR') else (None, 'baseline')
        else:
            layer,mode=(None,'baseline') if o.group=='N' or gem_band else o.group.split('_')
        historical = set()

        def remember(result):
            chosen = result.get('A', {}).get('sets', {}).get('overall')
            if chosen is not None:
                historical.update(chosen)

        if FACTORIAL and o.role == 'branch':
            remember(selection)
        protector=TS3Protection(learner,layer,mode,lambda x:append(out/'update_audit.jsonl',x))
        gem=None
        if gem_band:
            from a2_local_gem import A2LocalGEM
            gem=A2LocalGEM(learner,gem_band,lambda x:append(out/'gem_audit.jsonl',x),
                           lambda x:append(out/'gem_errors.jsonl',x),
                           score_threshold=P.get('gem_score_threshold'))
            source=next((Path(o.parent)/'detections').glob('A_*/screen_input.th'))
            source_batch=th.load(source,map_location='cpu')['batch']
            append(out/'gem_selection_history.jsonl',gem.update('A',selection,source_batch,runner.t_env))
            del source_batch
        chi_probes = {}
        if CHI and o.role == 'branch':
            chi_probes[0] = parent/'chi_probes'/'phase0.th'

        def refresh_chi(phase, task, *, episodes=None):
            try:
                if episodes is not None:
                    path = out/'chi_probes'/f'phase{phase}.th'
                    chi_probes[phase] = path
                    vdn_chi.save_probe(mac, episodes, path)
                if not layer:
                    return None
                result = vdn_chi.reassess(mac, chi_probes, layer, top_k=args.top_k,
                                          threshold=args.chi_threshold)
                append(out/'chi_assessments.jsonl', dict(phase=phase, task=task,
                       t_env=runner.t_env, layers=result))
                return result
            except Exception:
                append(out/'chi_errors.jsonl', dict(phase=phase, task=task, t_env=runner.t_env,
                       error=traceback.format_exc()))
                return {layer: dict(sets={'overall': []}, defined=False)} if layer else None

        if CHI and o.role == 'branch':
            selection = refresh_chi(0, P['order'][0])
        if o.role=='branch' and layer:append(out/'selection_history.jsonl',dict(source_task='A',next_task='B',**protector.update(selection,runner.t_env)))
        def status(state='running',**extra):write(out/'status.json',dict(status=state,pid=os.getpid(),t_env=runner.t_env,phase=phase,task=P['order'][phase],updates=updates,episode=episode,elapsed_seconds=time.time()-started,**extra))
        def evaluate_block(task, envseed, per, agent_state):
            ea = copy.deepcopy(args)
            ea.device = 'cpu'
            ea.use_cuda = False
            ea.env_args['seed'] = envseed
            emac = base.controllers[args.mac](buffer.scheme, groups, ea)
            emac.agent.load_state_dict(mac.agent.state_dict() if agent_state is None else agent_state)
            emac.agent.eval()
            er = base.EpisodeRunner(ea, base.Logger(logging.getLogger('eval')))
            episodes = []
            try:
                er.env.set_task(task)
                er.setup(scheme, groups, preprocess, emac)
                with th.no_grad():
                    for j in range(per):
                        er.run(test_mode=True)
                        episodes.append(dict(env_seed=envseed, episode=j,
                            won=bool(er.env.last_info.get('battle_won', False)), length=er.t))
            finally:
                er.close_env()
            return episodes

        def evaluate(task, focus, label, agent_state=None, reference_phase=None):
            saved = rng_state()
            idx = P['order'].index(task)
            blocks = [91101 + idx*100 + j for j in range(4)] if focus else [
                91001 + idx*100 + j for j in range(2)]
            per = 32 if focus else 16
            retries = P.get('evaluation_sc2_retries', 0)
            common = dict(t_env=runner.t_env, phase=phase, trained_task=P['order'][phase],
                          reference_phase=reference_phase, task=task, label=label, focus=focus)
            episodes = []
            try:
                for envseed in blocks:
                    block_rng = rng_state() if retries else None
                    for attempt in range(retries + 1):
                        try:
                            episodes.extend(evaluate_block(task, envseed, per, agent_state))
                            break
                        except (SC2ConnectionError, SC2ConnectError, SC2RequestError) as error:
                            if isinstance(error, SC2RequestError) and not (
                                    isinstance(error.res, sc_pb.ResponseCreateGame)
                                    and error.res.error == sc_pb.ResponseCreateGame.InvalidMapData):
                                raise
                            will_retry = attempt < retries
                            if retries:
                                append(out/'evaluation_sc2_errors.jsonl', dict(
                                    **common, env_seed=envseed, attempt=attempt+1,
                                    will_retry=will_retry, error=str(error)))
                            if will_retry:
                                h.restore_rng(block_rng)
                            elif retries and label == 'periodic':
                                status()
                                return
                            else:
                                raise
                for episode_result in episodes:
                    append(out/'test_episodes.jsonl', dict(**common, **episode_result))
                wins = sum(result['won'] for result in episodes)
                append(out/'evaluations.jsonl', dict(
                    **common, wins=wins, episodes=len(episodes), win_rate=wins/len(episodes)))
            finally:
                h.restore_rng(saved)
            status()
        def screen(task):
            dest=out/'detections'/f'{task}_{runner.t_env}';dest.mkdir(parents=True,exist_ok=False)
            try:
                if FACTORIAL or CHI_COMPARISON:
                    result = vdn_select(learner, args, fifo, dest,
                                        state_weighted=getattr(args, 'state_weighted_screening', True))
                elif algorithm == 'vdn':
                    detector = vdn_type_select if P.get('selection') in ('type_threshold_union', 'overall_threshold_union') else vdn_select
                    result = detector(learner, args, fifo, dest)
                else:
                    result = qmix_select(learner, args, fifo, dest)
                result = select_scope(result)
            except Exception:
                write(dest/'error.json',dict(error=traceback.format_exc()));result={key:dict(sets={'overall':None}) for key in ['A','A1','A2']}
            write(dest/'result.json',result)
            return result
        def detect_single_ff(task, nominal, boundary):
            record = dict(task=task, phase=phase, t_env=runner.t_env, nominal_t_env=nominal,
                          stage_end=boundary, episodes=len(fifo), layer='A', layer_path='agent.net[2]',
                          width=args.rnn_hidden_dim, threshold=detection['threshold'])
            try:
                layer = vdn_select(learner, args, fifo, save_input=False)['A']
                scores = layer['scores']
                layer['sets']['above_threshold'] = (
                    None if scores is None else [i for i, value in enumerate(scores)
                                                  if value > detection['threshold']])
                append(out/'neuron_detections.jsonl', dict(record, **layer))
            except Exception:
                append(out/'neuron_detection_errors.jsonl', dict(record, error=traceback.format_exc()))

        if track_retention:
            if o.role in ('prefix', 'full'):
                for unseen in P['order'][1:]:
                    evaluate(unseen, True, 'initial_reference', initial_agent, reference_phase=-1)
            elif o.role == 'resume':
                t_resume = runner.t_env
                runner.t_env = 0
                # Same initializer and seed as TS3. The original initial checkpoint was not archived.
                write(out/'initial_reference_identity.json', dict(
                    seed=o.seed, reconstructed_model_hash=digest(initial_agent)))
                history = P['history'][str(o.seed)]
                historical_fwt = ROOT/history['baseline_fwt']
                historical_identity = json.loads((historical_fwt/'initial_reference_identity.json').read_text())
                matching_reference = digest(initial_agent) == historical_identity['reconstructed_model_hash']
                # On a changed initializer, collect every initial reference using the same model.
                for unseen in P['order'][5 if matching_reference else 1:]:
                    evaluate(unseen, True, 'initial_reference', initial_agent, reference_phase=-1)
                if layer:
                    # B's reference comes from the shared A prefix. C-E use this branch's own models.
                    for phase in range(1, 4):
                        saved_model = next(parent.parent.glob(f'phase{phase}_*'))
                        runner.t_env = int(saved_model.name.rsplit('_', 1)[1])
                        agent = th.load(saved_model/'agent.th', map_location='cpu')
                        evaluate(P['order'][phase+1], True, 'next_unseen', agent, reference_phase=phase)
                phase = P['order'].index('E')
                runner.t_env = t_resume
                if layer:
                    saved_protection = json.loads((parent/'protection.json').read_text())
                    protector.schedule.replace(saved_protection['protected'], t_resume)
                    sample_seed = o.seed + 500000
                    saved_rng = rng_state()
                    fifo = deque(maxlen=P['resume_screen_episodes'])
                    try:
                        sample_args = copy.deepcopy(args)
                        sample_args.env_args['seed'] = sample_seed
                        sample_runner = base.EpisodeRunner(sample_args, logger)
                        try:
                            sample_runner.env.set_task('E')
                            sample_runner.setup(scheme, groups, preprocess, mac)
                            with th.no_grad():
                                for _ in range(fifo.maxlen):
                                    sample_runner.t_env = t_resume
                                    fifo.append((cpu_episode(sample_runner.run(test_mode=False)), None))
                        finally:
                            sample_runner.close_env()
                    finally:
                        h.restore_rng(saved_rng)
                    selection = screen('E')
                    append(out/'selection_history.jsonl', dict(
                        source_task='E', next_task='F', sample_seed=sample_seed,
                        resampled_episodes=len(fifo), source_checkpoint=str(parent),
                        **protector.update(selection, t_resume)))
                evaluate('F', True, 'next_unseen', reference_phase=phase)
        if gem_mode:
            # Shared per-seed initial reference (not claimed as an archived original initializer).
            # Evaluation RNG is restored; it never touches replay or takes gradient steps.
            cache=ROOT/P['initial_reference_cache']/f'seed{o.seed}.json'
            initial_hash=digest(initial_agent)
            write(out/'initial_reference_identity.json',dict(seed=o.seed,reconstructed_model_hash=initial_hash,meaning='same-seed fresh untrained reference, not an archived initial checkpoint'))
            if cache.exists():
                cached=json.loads(cache.read_text())
                write(out/'initial_reference_reuse.json',dict(source=str(cache),cached_model_hash=cached.get('model_hash'),current_model_hash=initial_hash,hashes_match=cached.get('model_hash')==initial_hash))
                for item in json.loads(cache.read_text())['evaluations']:append(out/'evaluations.jsonl',item)
            else:
                th.save(initial_agent,out/'initial_reference_agent.th')
                for unseen in P['order'][1:]:evaluate(unseen,True,'initial_reference',initial_agent,reference_phase=-1)
                data=[json.loads(x) for x in (out/'evaluations.jsonl').read_text().splitlines() if json.loads(x)['label']=='initial_reference']
                cache.parent.mkdir(parents=True,exist_ok=True)
                temp=cache.with_name(cache.name+f'.{os.getpid()}.tmp')
                temp.write_text(json.dumps(dict(seed=o.seed,model_hash=initial_hash,model_source=str(out/'initial_reference_agent.th'),meaning='untrained same-seed architecture reference reconstructed by current initializer; not an archived historical initializer',evaluations=data),indent=2));temp.replace(cache)
            if o.role=='eval_fwt':
                source=P['sources'][str(o.seed)];phase=0
                prefix=ROOT/'runs'/source['prefix'];base_run=ROOT/'runs'/source['baseline']
                for completed in range(4):
                    phase=completed
                    directory=next(((prefix if phase==0 else base_run)/'models').glob(f'phase{phase}_*'))
                    model=th.load(directory/'agent.th',map_location='cpu')
                    target_file=directory/'targets.th';runner.t_env=th.load(target_file,map_location='cpu')['t_env']
                    evaluate(P['order'][phase+1],True,'next_unseen',model,reference_phase=phase)
                status('completed',finished=time.time(),evaluation_only=True,new_training_steps=0);return
            if o.role=='branch':
                phase=0;evaluate('B',True,'next_unseen',reference_phase=0)
        if o.role == 'prefix':
            phases = range(1)
        elif o.role == 'full':
            phases = range(len(P['order']))
        else:
            phases = range(phase + 1, len(P['order']))
        last_phase = len(P['order']) - 1
        for phase in phases:
            end=P['thresholds'][phase];task=P['order'][phase];begin=runner.t_env;runner.env.set_task(task);fifo=deque(maxlen=vdn_chi.PROBE_EPISODES if CHI else P['screen_episodes'])
            if FACTORIAL:
                trace = learner.a2_trace
                trace.phase = phase
                trace.history = sorted(historical)
                trace.protected = sorted(protector.schedule.protected[layer]) if layer else []
                trace.streak.zero_()
                if learner.trhr is not None:
                    learner.trhr.protected.zero_()
                    learner.trhr.protected[trace.protected] = True
            base.log_task_epsilon(runner, out, phase, task, 'start')
            append(out/'transitions.jsonl',dict(phase=phase,task=task,actual_start=begin,nominal_end=end,buffer_episodes=buffer.episodes_in_buffer,protection=None if not layer else protector.schedule.state(begin),gem_band=gem_band))
            audited=[False]
            def audit(x):
                if not audited[0]:append(out/'environment_audit.jsonl',dict(phase=phase,t_env=runner.t_env,**x));audited[0]=True
            runner.env.audit=audit;status();evaluate(task,False,'phase_start');next_test=(runner.t_env//25000+1)*25000
            if detection:
                interval = detection['interval']
                next_detection = (begin // interval + 1) * interval
            while runner.t_env<end:
                with th.no_grad():batch=runner.run(test_mode=False)
                if FACTORIAL:
                    batch.update({'source_phase': [[phase]]}, mark_filled=False)
                if CHI or FACTORIAL or o.role=='prefix' or layer or gem or detection:
                    types = None
                    if not CHI and not EXTENDED and not FACTORIAL:
                        raw = runner.env.wrapped.env
                        types = ['stalker' if raw.agents[i].unit_type==raw.stalker_id else 'zealot' for i in range(args.n_agents)]
                    fifo.append((cpu_episode(batch), types))
                buffer.insert_episode_batch(batch)
                if buffer.can_sample(args.batch_size):
                    sample=buffer.sample(args.batch_size);sample=sample[:,:sample.max_t_filled()];sample.to(args.device);protector.t_env=runner.t_env
                    if gem:gem.phase=phase
                    learner.train(sample,runner.t_env,episode);updates+=1;del sample
                episode+=1
                if episode%25==0:status()
                if detection and runner.t_env >= min(next_detection, end):
                    detect_single_ff(task, min(next_detection, end), runner.t_env >= end)
                    next_detection += interval
                if runner.t_env>=end:break
                if runner.t_env>=next_test:
                    evaluate(task,False,'periodic')
                    while next_test<=runner.t_env:next_test+=25000
            if FACTORIAL:
                trace.flush(runner.t_env)
            base.log_task_epsilon(runner, out, phase, task, 'end')
            append(out/'phase_end.jsonl',dict(phase=phase,task=task,actual_start=begin,actual_end=runner.t_env,nominal_end=end,overshoot=runner.t_env-end))
            dest=out/'models'/f'phase{phase}_{task}_{runner.t_env}';dest.mkdir(parents=True,exist_ok=False);learner.save_models(str(dest))
            th.save(dict(target_agent=learner.target_mac.agent.state_dict(),target_mixer=learner.target_mixer.state_dict(),t_env=runner.t_env,episode=episode,updates=updates),dest/'targets.th')
            if layer:write(dest/'protection.json',protector.schedule.state(runner.t_env))
            if gem:gem.save(dest/'gem_state.th')
            for seen in P['order'][:phase+1]:evaluate(seen,True,'stage_end')
            if (gem or track_retention) and phase<last_phase:evaluate(P['order'][phase+1],True,'next_unseen',reference_phase=phase)
            if CHI:
                selection = refresh_chi(phase, task, episodes=[entry[0] for entry in fifo])
                if CHI_COMPARISON and o.role == 'prefix':
                    selection = screen(task)
                if phase < last_phase and layer:
                    append(out/'selection_history.jsonl', dict(source_task=task, next_task=P['order'][phase+1],
                           **protector.update(selection, runner.t_env)))
            elif phase<last_phase and (FACTORIAL or o.role=='prefix' or layer or gem):
                selection=screen(task)
                if FACTORIAL:
                    remember(selection)
                    append(out/'historical_selection.jsonl', dict(task=task, t_env=runner.t_env,
                           selected=selection['A']['sets']['overall'], union=sorted(historical)))
                if gem:
                    from protection import combine
                    append(out/'gem_selection_history.jsonl',gem.update(task,selection,combine([x[0] for x in fifo]),runner.t_env))
                if layer:append(out/'selection_history.jsonl',dict(source_task=task,next_task=P['order'][phase+1],**protector.update(selection,runner.t_env)))
        if o.role=='prefix':
            if not CHI or CHI_COMPARISON:
                write(out/'selection.json',selection)
            state=dict(agent=mac.agent.state_dict(),mixer=learner.mixer.state_dict(),target_agent=learner.target_mac.agent.state_dict(),target_mixer=learner.target_mixer.state_dict(),optimizer=learner.optimiser.state_dict(),buffer=buffer,t_env=runner.t_env,episode=episode,updates=updates,last_target_update_episode=learner.last_target_update_episode,log_stats_t=learner.log_stats_t,epsilon=mac.action_selector.epsilon,rng=rng_state(),environment_distributions=runner.env.distribution_state(),model_hash=base.parameter_hash(learner))
            th.save(state,out/'prefix_state.th')
        if gem:gem.flush()
        if track_retention and o.role != 'prefix':
            all_labels = ('stage_end', 'initial_reference', 'next_unseen')
            if FACTORIAL:
                all_labels += ('phase_start', 'periodic')
            if o.role == 'resume':
                sources = [(ROOT/history['prefix'], ('stage_end',)),
                           (parent.parent.parent, ('stage_end',)),
                           (historical_fwt, tuple(label for label, use in (
                               ('initial_reference', matching_reference), ('next_unseen', True)) if use))]
            elif o.role == 'full':
                sources = []
            else:
                sources = [(Path(o.parent), all_labels)]
            write(out/'retention.json', sequence_metrics(P['order'], sources + [(out, all_labels)]))
        status('completed',finished=time.time())
    except BaseException:
        write(out/'status.json',dict(status='failed',t_env=runner.t_env if runner else 0,phase=phase,updates=updates,error=traceback.format_exc()));raise
    finally:
        if runner:runner.close_env()

if __name__=='__main__':main()
