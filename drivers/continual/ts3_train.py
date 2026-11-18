"""Exact-roster sequence training with shared prefixes and optional soft protection."""
import argparse,copy,json,logging,os,random,time,traceback
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import hetero_train as h
from common.checkpoints import (continuation_state, cpu_snapshot, parent_hash_fields,
                                restore_networks_and_optimizer,
                                save_stage_models)
from common.evaluation import evaluate_block, fixed_seed_plan
from common.metrics import Metrics, method_for_group, record_evaluation
from common.records import attach_legacy_metrics
from common.runtime import begin_environment_audit, episode_spec, log_phase_end, restore_rng, td_sample, write_sequence_status
from hetero_environment import TEAMS
from hetero_protection import select as qmix_select
from ts3_vdn_detection import select as vdn_select
from ts3_vdn_type_detection import select as vdn_type_select
from ts3_protection import TS3Protection
from ts4_diagnostics import score as vdn_score_layers
import vdn_chi
from protection import rng_state,digest
from memory import History
from memory.episodes import combine, cpu_episode
from components.task_replay import (log_first_replay_batch, make_replay_buffer,
                                    restore_replay_buffer, start_replay_task)
import numpy as np
import torch as th
import yaml
from pysc2.lib.protocol import ConnectionError as SC2ConnectionError
from pysc2.lib.remote_controller import ConnectError as SC2ConnectError, RequestError as SC2RequestError
from s2clientprotocol import sc2api_pb2 as sc_pb
base=h.base; ROOT=base.ROOT; write=base.write; append=base.append
P=base.PROTOCOL
EXTENDED = P['id'] in ('ts5', 'ts6', 'ts7', 'ts8')
FACTORIAL = P.get('intervention') == 'single_ff_trhr_factorial'
CHI = P.get('intervention') == 'vdn_chi'
CHI_TRHR = CHI and 'trhr' in P
TRACE = FACTORIAL or CHI_TRHR
CHI_COMPARISON = P.get('chi_comparison', False)
SINGLE_FF_CONTROL = P.get('intervention') == 'single_ff_softprotect'
TS4 = P.get('suite') == 'ts4'
TS4_FACTORIAL = FACTORIAL and TS4
RETURN_SEQUENCE = TS4 or P.get('suite') == 'ts8'
if TS4:
    import ts4_train  # registers the TS4 environment
elif not EXTENDED:
    TEAMS.update({key:tuple(value) for key,value in P['tasks'].items()})

def select_scope(result):
    if FACTORIAL or CHI_COMPARISON:
        item = result['A']
        if item.get('selection') == 'fixed_mask':
            return result
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

def sequence_metrics(order, sources, *, stage_end_steps=None):
    """Join immutable prefix and current evaluation rows, then calculate stage metrics."""
    stages, initial, before, curves = {}, {}, {}, {}
    stage_labels = P.get('stage_labels', order)
    for source, labels in sources:
        for line in (source/'evaluations.jsonl').read_text().splitlines():
            row = json.loads(line)
            task, label, win_rate = (row[k] for k in ('task', 'label', 'win_rate'))
            if label not in labels:
                continue
            if label == 'stage_end':
                stages.setdefault(stage_labels[row['phase']], {})[task] = win_rate
                if stage_end_steps is not None:
                    stage_end_steps[row['phase']] = (row['t_env'], source.name)
            elif label == 'initial_reference':
                initial[task] = win_rate
            elif label == 'next_unseen':
                before[task] = win_rate
            if label in ('phase_start', 'periodic', 'stage_end') and task == order[row['phase']]:
                curves.setdefault(stage_labels[row['phase']], {})[row['t_env']] = win_rate
    metrics = {}
    for i, (task, stage) in enumerate(zip(order, stage_labels)):
        row = stages.get(stage, {})
        seen = list(dict.fromkeys(order[:i+1]))
        old = [j for j in seen if j != task]
        previous = dict(zip(order[:i], stage_labels[:i]))
        acc = sum(row[j] for j in seen) / len(seen) if all(j in row for j in seen) else None
        bwt = [row[j] - stages[previous[j]][j] for j in old
               if j in row and j in stages.get(previous[j], {})]
        fwt_tasks = [j for j in seen[1:] if j in initial and j in before]
        fwt = [before[j] - initial[j] for j in fwt_tasks]
        metrics[stage] = dict(
            acc_percent=None if acc is None else 100 * acc,
            bwt_pp=100 * sum(bwt) / len(bwt) if bwt and len(bwt) == len(old) else None,
            fwt_pp=100 * sum(fwt) / len(fwt) if fwt and len(fwt) == len(seen)-1 else None,
            fwt_tasks=fwt_tasks)
        if FACTORIAL or RETURN_SEQUENCE:
            points = sorted(curves.get(stage, {}).items())
            metrics[stage]['old_mean_percent'] = (100 * sum(row[j] for j in old) / len(old)
                if old and all(j in row for j in old) else None)
            metrics[stage]['learning_auc_percent'] = (
                100 * float(np.trapz([y for _, y in points], [x for x, _ in points]))
                / (points[-1][0] - points[0][0]) if task in row and len(points) > 1 else None)
            metrics[stage]['learning_curve'] = points
            if task in previous:
                pre_return = stages.get(stage_labels[i-1], {}).get(task)
                metrics[stage]['before_revisit_percent'] = None if pre_return is None else 100 * pre_return
                metrics[stage]['revisit_gain_pp'] = (
                    100 * (row[task] - pre_return) if pre_return is not None and task in row else None)
    return dict(stage_win_rates=stages, initial_reference=initial, before_first_training=before,
                metrics=metrics, sources=[str(source/'evaluations.jsonl') for source, _ in sources])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['prefix','branch','eval_fwt','resume','full'],required=True)
    parser.add_argument('--parent');parser.add_argument('--group',choices=P['groups'],default='N');parser.add_argument('--seed',type=int,required=True);o=parser.parse_args()
    out=Path(os.environ['MARL_RUN_DIR']);started=time.time();runner=None;updates=episode=phase=0
    save_branch_prefix = o.role == 'branch' and P.get('save_branch_prefix', False)
    release_search = P.get('chi_release_search', False)
    release_trial = P.get('chi_release_trial', False)
    release_event = None
    if o.role == 'branch':
        phase = P.get('branch_start_phase', 1) - 1
    if release_trial:
        release_event = json.loads((Path(o.parent)/'release_event.json').read_text())
        if release_event['phase'] is None:
            write(out/'status.json', dict(status='skipped', reason='no CHI exit in TS8'))
            return
        phase = release_event['phase']
    track_retention = EXTENDED or FACTORIAL or TS4 or CHI_COMPARISON or o.role == 'full'
    try:
        th.set_num_threads(1);random.seed(o.seed);np.random.seed(o.seed);th.manual_seed(o.seed);th.cuda.manual_seed_all(o.seed)
        cfg=h.config(o.seed,out)
        algorithm=P.get('algorithm','qmix')
        detection = P['single_ff_detection'] if o.role == 'full' and not CHI and not FACTORIAL else None
        if algorithm != 'qmix':
            base.merge(cfg, yaml.safe_load((ROOT/'src/config/algs'/f'{algorithm}.yaml').read_text()))
        cfg['algorithm']=algorithm;cfg['top_k']=P.get('top_k')
        if FACTORIAL:
            cfg.update(a2_trace=True, trace_stages=len(P['order']),
                       state_weighted_screening=(o.group != 'A_equal_union' and P.get('state_weighted_screening', True)))
            if o.group in ('TRHR', 'A_union_TRHR'):
                cfg['trhr'] = P['trhr_settings']
        elif 'trhr' in P:
            cfg['trhr'] = P['trhr']
        if CHI_TRHR:
            cfg.update(a2_trace=True, trace_stages=len(P['order']))
        cfg.update(t_max=P['thresholds'][-1],protocol_id=P['id'],role=o.role,parent=o.parent,group=o.group,screening=P['screening'],phase_boundary=P['phase_boundary'],tasks=P['tasks'],thresholds=P['thresholds'])
        if SINGLE_FF_CONTROL:
            cfg['intervention'] = 'single_ff_softprotect'
        if CHI:
            cfg.update(intervention='vdn_chi', chi_probe_episodes=vdn_chi.PROBE_EPISODES,
                       chi_threshold=P.get('threshold_beta'),
                       chi_protect_output=not o.group.endswith('_input'))
        if 'gem_score_threshold' in P:
            cfg['gem_score_threshold'] = P['gem_score_threshold']
        if 'sc2_game_version' in P:
            cfg['env_args']['game_version'] = P['sc2_game_version']
        if EXTENDED:
            cfg['env_args']['rosters'] = P['tasks']
        if TS4:
            cfg['env'] = 'ts4_protoss'
            cfg['env_args']['capability_config']['team_gen']['weights'] = [1., 0., 0.]
        if P.get('replay_on_switch') == 'er' and o.role == 'resume':
            raise ValueError('VDN-ER requires historical replay; the TS6 weight-only resume has none')
        if 'replay_on_switch' in P:
            cfg['replay_on_switch'] = P['replay_on_switch']
        args=SimpleNamespace(**cfg)
        logging.basicConfig(level=logging.INFO)
        logger = base.Logger(logging.getLogger(out.name))
        attach_legacy_metrics(logger, out)
        method = method_for_group(o.group, chi=CHI, trhr='trhr' in cfg)
        metrics = Metrics(out, algorithm, method, o.seed, group=o.group, role=o.role)
        runner=base.EpisodeRunner(args,logger);info=runner.get_env_info();args.n_agents=info['n_agents'];args.n_actions=info['n_actions'];args.state_shape=info['state_shape']
        write(out/'config.json',vars(args));write(out/'env_info.json',info)
        scheme, groups, preprocess = spec = episode_spec(info)
        if FACTORIAL or (TS4 and (CHI or CHI_COMPARISON)):
            scheme['source_phase'] = dict(vshape=(1,), dtype=th.long, episode_const=True)
        buffer = make_replay_buffer(args, P['order'], scheme, groups,
                                    info['episode_limit'] + 1, preprocess)
        mac=base.controllers[args.mac](buffer.scheme,groups,args);runner.setup(scheme,groups,preprocess,mac);learner=base.learners[args.learner](mac,buffer.scheme,logger,args)
        for network in [mac.agent,learner.target_mac.agent,learner.mixer,learner.target_mixer]:
            for module in network.modules():module._forward_hooks.clear()
        learner.cuda();selection={}
        if TRACE:
            def record_trace(row):
                trace_phase = row['phase']
                trace_layer = 'A' if args.agent == 'single_ff_rnn' else 'A2'
                metrics.neurons(trace_phase, P['order'][trace_phase], row['t_env'], trace_layer,
                                {'activation_rate': row['activation_rate']},
                                event='activation_window', start_t_env=row['start_t_env'])
            learner.a2_trace.record = record_trace
        gem_mode=P.get('intervention')=='a2_local_gem'
        initial_agent=cpu_snapshot(mac.agent) if gem_mode or (track_retention and o.role in ('prefix', 'resume', 'full')) else None
        initial_hashes = set()
        if gem_mode or (track_retention and o.role == 'resume'):
            initial_hashes.add(digest(initial_agent))
            if hasattr(mac.agent, 'legacy_initial_state'):
                initial_hashes.add(digest(mac.agent.legacy_initial_state(initial_agent)))
        if o.role=='branch':
            parent=Path(o.parent);state=th.load(parent/'prefix_state.th',map_location='cpu')
            restore_networks_and_optimizer(learner, state)
            buffer=restore_replay_buffer(buffer,state['buffer'],P['order'][:phase+1]);episode=state['episode'];updates=state['updates'];learner.last_target_update_episode=state['last_target_update_episode'];learner.log_stats_t=state['log_stats_t'];runner.t_env=state['t_env']
            mac.action_selector.epsilon=state['epsilon'];runner.env.restore_distribution(state['environment_distributions']);restore_rng(state['rng'])
            if 'trhr' not in P and not CHI:
                selection_path = ROOT / P['prefix_selection_override'] if P.get('prefix_selection_override') else parent/'selection.json'
                if TS4_FACTORIAL and o.group == 'A_equal_union':
                    selection_path = parent/'selection_equal.json'
                selection=select_scope(json.loads(selection_path.read_text()))
                if P.get('prefix_selection_override'):
                    import hashlib
                    write(out/'prefix_selection.json', dict(selection=selection, source=str(selection_path), source_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(), parent_original=str(parent/'selection.json'), parent_original_sha256=hashlib.sha256((parent/'selection.json').read_bytes()).hexdigest()))
            write(out/'branch_restore.json',dict(parent=str(parent),t_env=runner.t_env,episode=episode,updates=updates,buffer_episodes=buffer.episodes_in_buffer,buffer_index=buffer.buffer_index,model_hash=base.parameter_hash(learner),**parent_hash_fields(state),rng_match=digest(rng_state())==digest(state['rng']),engine='new SC2 process; recoverable distribution RNG restored'))
            if TRACE:
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
            layer, mode = ('A', 'union') if o.group in ('A_union', 'A_union_TRHR', 'A_equal_union', 'A_union_input') else (None, 'baseline')
        else:
            layer,mode=(None,'baseline') if o.group=='N' or gem_band else o.group.split('_', 1)
        if CHI and mode.endswith('_input'):
            mode = mode[:-len('_input')]
        historical = set()

        def remember(result):
            chosen = result.get('A', {}).get('sets', {}).get('overall')
            if chosen is not None:
                historical.update(chosen)

        if FACTORIAL and o.role == 'branch':
            remember(selection)
        protect_output = (args.chi_protect_output if CHI else o.group != 'A_union_input')
        protector=TS3Protection(learner,layer,mode,lambda x:append(out/'update_audit.jsonl',x),
                                protect_output=protect_output,
                                recovery_steps=P['recovery_steps'] if release_trial else 100000)
        gem=None
        if gem_band:
            from a2_local_gem import A2LocalGEM
            def record_gem(row):
                append(out/'gem_audit.jsonl', row)
                phase_index = row['phase']
                keys = ('updates', 'projections', 'fallbacks', 'zero_or_invalid_reference_rows',
                        'correction_norm_sum')
                metrics.record('Mechanism', phase_index, P['order'][phase_index], row['t_env'],
                               {key: row[key] for key in keys}, event='gem_window', band=row['band'])
                for source_task, task_stats in row.get('tasks', {}).items():
                    metrics.record('Mechanism', phase_index, P['order'][phase_index], row['t_env'],
                                   {'max_actual_local_dot': task_stats['max_actual_local_dot']},
                                   event='gem_window', band=row['band'], source_task=source_task)

            gem=A2LocalGEM(learner,gem_band,record_gem,
                           lambda x:append(out/'gem_errors.jsonl',x),
                           score_threshold=P.get('gem_score_threshold'))
            source=next((Path(o.parent)/'detections').glob('A_*/screen_input.th'))
            source_batch = History.load(source)['batch']
            append(out/'gem_selection_history.jsonl',gem.update('A',selection,source_batch,runner.t_env))
            del source_batch
        chi_probes = History()
        if CHI and o.role == 'branch':
            if release_trial:
                chi_probes.update((int(i), Path(path)) for i, path in release_event['probes'].items())
            else:
                chi_probes[0] = (ROOT/P['chi_prefix_probe'] if P.get('chi_prefix_probe')
                                 else parent/'chi_probes'/'phase0.th')

        def refresh_chi(phase, task, *, episodes=None):
            try:
                if episodes is not None:
                    path = out/'chi_probes'/f'phase{phase}.th'
                    vdn_chi.save_probe(mac, episodes, path)
                    chi_probes[phase] = path
                if not layer:
                    return None
                result = vdn_chi.reassess(mac, chi_probes, layer, top_k=args.top_k,
                                          threshold=args.chi_threshold)
                append(out/'chi_assessments.jsonl', dict(phase=phase, task=task,
                       t_env=runner.t_env, layers=result))
                metrics.neurons(phase, task, runner.t_env, layer,
                                {'normalized_score': result[layer]['scores']}, event='chi_reassessment')
                return result
            except Exception:
                append(out/'chi_errors.jsonl', dict(phase=phase, task=task, t_env=runner.t_env,
                       error=traceback.format_exc()))
                return {layer: dict(sets={'overall': None}, defined=False,
                                    skip_reason='detection_error')} if layer else None

        if CHI and o.role == 'branch':
            if release_trial:
                selection = {layer: dict(sets={'overall': release_event['selected']},
                                         defined=release_event['defined'])}
                protector.schedule.replace({layer: release_event['previous']}, runner.t_env)
                if P.get('chi_keep_exiting'):
                    selection[layer]['sets']['overall'] = sorted(
                        set(release_event['selected']) | set(release_event['exiting']))
            else:
                selection = refresh_chi(0, P['order'][0])
            if CHI_TRHR:
                remember(selection)
        if o.role == 'branch' and layer:
            append(out/'selection_history.jsonl', dict(source_task=P['order'][phase],
                   next_task=P['order'][phase+1], **protector.update(selection, runner.t_env)))
        def status(state='running', **extra):
            write_sequence_status(out, state, runner.t_env, phase, P['order'][phase],
                                  updates, episode, started, **extra)
        def evaluate(task, focus, label, agent_state=None, reference_phase=None):
            saved = rng_state()
            idx = P['order'].index(task)
            blocks, per = fixed_seed_plan(idx, focus)
            retries = P.get('evaluation_sc2_retries', 0)
            common = dict(t_env=runner.t_env, phase=phase, trained_task=P['order'][phase],
                          reference_phase=reference_phase, task=task, label=label, focus=focus)
            episodes = []
            try:
                for envseed in blocks:
                    block_rng = rng_state() if retries else None
                    for attempt in range(retries + 1):
                        try:
                            reference = mac.agent.state_dict() if agent_state is None else agent_state
                            episodes.extend(evaluate_block(
                                args, spec, buffer.scheme, reference, task, envseed, per,
                                make_controller=base.controllers[args.mac], make_runner=base.EpisodeRunner,
                                make_logger=base.Logger))
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
                                restore_rng(block_rng)
                            elif retries and label in ('periodic', 'release_window', 'release_old'):
                                status()
                                return
                            else:
                                raise
                for episode_result in episodes:
                    append(out/'test_episodes.jsonl', dict(**common, **episode_result))
                wins = sum(result['won'] for result in episodes)
                record_evaluation(out, metrics, dict(
                    **common, wins=wins, episodes=len(episodes), win_rate=wins/len(episodes)))
            finally:
                restore_rng(saved)
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
            for name, item in result.items():
                scores = item.get('scores')
                score_groups = scores if isinstance(scores, dict) else item.get('group_statistics', {})
                if score_groups:
                    for group, values in score_groups.items():
                        metrics.neurons(phase, task, runner.t_env, name,
                                        {'weighted_activity': values.get('activity'),
                                         'normalized_score': values.get('scores', values.get('score'))},
                                        event='boundary_detection', agent_group=group)
                else:
                    metrics.neurons(phase, task, runner.t_env, name,
                                    {'weighted_activity': item.get('activity'), 'normalized_score': scores},
                                    event='boundary_detection')
            return result
        def detect_layers(task, nominal, boundary):
            record = dict(task=task, phase=phase, t_env=runner.t_env, nominal_t_env=nominal,
                          stage_end=boundary, episodes=len(fifo), width=args.rnn_hidden_dim,
                          threshold=detection['threshold'])
            try:
                if args.agent == 'single_ff_rnn':
                    layers = vdn_select(learner, args, fifo, save_input=False)
                else:
                    layers = vdn_score_layers(learner, args, combine([entry[0] for entry in fifo]))
                for name, layer in layers.items():
                    scores = layer['scores']
                    if name != 'A':
                        activity = layer['activity']
                        layer['sets']['overall'] = (None if activity is None else sorted(
                            sorted(range(len(activity)), key=lambda i: (-activity[i], i))[:args.top_k]))
                    layer['sets']['above_threshold'] = (
                        None if scores is None else [i for i, value in enumerate(scores)
                                                      if value > detection['threshold']])
                    path = 'agent.net[4]' if name == 'A2' else 'agent.net[2]'
                    append(out/'neuron_detections.jsonl', dict(record, layer=name, layer_path=path, **layer))
                    metrics.neurons(phase, task, runner.t_env, name,
                                    {'weighted_activity': layer['activity'], 'normalized_score': scores},
                                    event='stage_end' if boundary else 'periodic_detection')
            except Exception:
                append(out/'neuron_detection_errors.jsonl', dict(record, error=traceback.format_exc()))

        if track_retention:
            if o.role in ('prefix', 'full'):
                for unseen in list(dict.fromkeys(P['order']))[1:]:
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
                matching_reference = historical_identity['reconstructed_model_hash'] in initial_hashes
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
                        restore_rng(saved_rng)
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
                write(out/'initial_reference_reuse.json',dict(source=str(cache),cached_model_hash=cached.get('model_hash'),current_model_hash=initial_hash,hashes_match=cached.get('model_hash') in initial_hashes))
                for item in json.loads(cache.read_text())['evaluations']:
                    record_evaluation(out, metrics, item)
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
                phase=0;evaluate(P['order'][1],True,'next_unseen',reference_phase=0)
        if o.role == 'prefix':
            phases = range(1)
        elif o.role == 'full':
            phases = range(len(P['order']))
        else:
            phases = range(phase + 1, len(P['order']))
        last_phase = len(P['order']) - 1

        def save_continuation():
            state = continuation_state(learner, runner, buffer, episode, updates)
            th.save(state, out/'prefix_state.th')

        for phase in phases:
            end=P['thresholds'][phase];task=P['order'][phase];begin=runner.t_env;runner.env.set_task(task);fifo=deque(maxlen=vdn_chi.PROBE_EPISODES if CHI else P['screen_episodes'])
            if TRACE:
                trace = learner.a2_trace
                trace.phase = phase
                trace.history = sorted(historical)
                trace.protected = sorted(protector.schedule.protected[layer]) if layer else []
                trace.streak.zero_()
                if learner.trhr is not None:
                    learner.trhr.protected.zero_()
                    learner.trhr.protected[trace.protected] = True
            replay_state = start_replay_task(buffer, task)
            base.log_task_epsilon(runner, out, phase, task, 'start')
            schedule = protector.schedule.state(begin) if layer else None
            append(out/'transitions.jsonl',dict(phase=phase,task=task,actual_start=begin,nominal_end=end,buffer_episodes=buffer.episodes_in_buffer,protection=schedule,gem_band=gem_band,**replay_state))
            if schedule is not None:
                metrics.protection(phase, task, begin, schedule)
            begin_environment_audit(runner, out, phase)
            status()
            evaluate(task, release_trial, 'phase_start')
            next_test = begin + 25000 if release_trial else (begin//25000+1)*25000
            if release_trial:
                old_tasks = [seen for seen in dict.fromkeys(P['order'][:phase]) if seen != task]
            if detection:
                interval = detection['interval']
                next_detection = (begin // interval + 1) * interval
            while runner.t_env<end:
                with th.no_grad():batch=runner.run(test_mode=False)
                if FACTORIAL or (TS4 and (CHI or CHI_COMPARISON)):
                    batch.update({'source_phase': [[phase]]}, mark_filled=False)
                if CHI or FACTORIAL or o.role=='prefix' or layer or gem or detection:
                    types = None
                    if not CHI and not EXTENDED and not FACTORIAL:
                        raw = runner.env.wrapped.env
                        types = ['stalker' if raw.agents[i].unit_type==raw.stalker_id else 'zealot' for i in range(args.n_agents)]
                    fifo.append((cpu_episode(batch), types))
                buffer.insert_episode_batch(batch)
                if buffer.can_sample(args.batch_size):
                    sample=td_sample(buffer, args.batch_size, args.device);protector.t_env=runner.t_env
                    if gem:gem.phase=phase
                    learner.train(sample,runner.t_env,episode);updates+=1;del sample
                    if replay_state:
                        log_first_replay_batch(buffer, append, out, phase, task, runner.t_env)
                        replay_state = {}
                episode+=1
                if episode%25==0:status()
                if detection and runner.t_env >= min(next_detection, end):
                    detect_layers(task, min(next_detection, end), runner.t_env >= end)
                    next_detection += interval
                if runner.t_env>=end:break
                if runner.t_env>=next_test:
                    elapsed = next_test - begin
                    if release_trial and elapsed <= 200000:
                        evaluate(task, True, 'release_window', reference_phase=elapsed)
                        if elapsed in (50000, 100000, 200000):
                            for seen in old_tasks:
                                evaluate(seen, True, 'release_old', reference_phase=elapsed)
                    else:
                        evaluate(task, False, 'periodic')
                    while next_test <= runner.t_env:
                        next_test += 25000
            if TRACE:
                trace.flush(runner.t_env)
            base.log_task_epsilon(runner, out, phase, task, 'end')
            log_phase_end(out, phase, task, begin, end, runner.t_env)
            dest = out/'models'/f'phase{phase}_{task}_{runner.t_env}'
            save_stage_models(learner, dest, runner.t_env, episode, updates)
            if layer:write(dest/'protection.json',protector.schedule.state(runner.t_env))
            if gem:gem.save(dest/'gem_state.th')
            seen_tasks = list(dict.fromkeys(P['order'][:phase+1]))
            for seen in seen_tasks:
                evaluate(seen, True, 'stage_end')
            if (gem or track_retention) and phase < last_phase and P['order'][phase+1] not in seen_tasks:
                evaluate(P['order'][phase+1], True, 'next_unseen', reference_phase=phase)
            if release_trial:
                break
            if CHI:
                selection = refresh_chi(phase, task, episodes=[entry[0] for entry in fifo])
                if release_search and phase < last_phase and selection[layer]['defined']:
                    previous = protector.schedule.protected[layer]
                    item = selection[layer]
                    chosen = set(item['sets']['overall'])
                    exiting = previous - chosen
                    if exiting:
                        save_continuation()
                        write(out/'release_event.json', dict(
                            phase=phase, t_env=runner.t_env, previous=sorted(previous),
                            selected=sorted(chosen), exiting=sorted(exiting),
                            defined=item.get('defined', True),
                            probes={i: str(path) for i, path in chi_probes.items()}))
                        break
                if CHI_TRHR:
                    remember(selection)
                if CHI_COMPARISON and o.role == 'prefix':
                    selection = screen(task)
                if phase < last_phase and layer:
                    append(out/'selection_history.jsonl', dict(source_task=task, next_task=P['order'][phase+1],
                           **protector.update(selection, runner.t_env)))
            elif ((phase < last_phase and (FACTORIAL or o.role == 'prefix' or layer or gem))
                  or (phase == last_phase and save_branch_prefix)):
                selection=screen(task)
                if TS4_FACTORIAL and o.role == 'prefix':
                    try:
                        equal = select_scope(vdn_select(learner, args, fifo,
                                                        save_input=False, state_weighted=False))
                    except Exception:
                        write(out/'selection_equal_error.json', dict(error=traceback.format_exc()))
                        equal = {'A': dict(sets={'overall': None})}
                    write(out/'selection_equal.json', equal)
                if FACTORIAL:
                    remember(selection)
                    append(out/'historical_selection.jsonl', dict(task=task, t_env=runner.t_env,
                           selected=selection['A']['sets']['overall'], union=sorted(historical)))
                if gem:
                                        append(out/'gem_selection_history.jsonl',gem.update(task,selection,combine([x[0] for x in fifo]),runner.t_env))
                if layer:append(out/'selection_history.jsonl',dict(source_task=task,next_task=P['order'][phase+1],**protector.update(selection,runner.t_env)))
        if o.role == 'prefix' or save_branch_prefix:
            if not CHI or CHI_COMPARISON:
                write(out/'selection.json',selection)
            save_continuation()
        if release_search and not (out/'release_event.json').exists():
            write(out/'release_event.json', dict(phase=None, t_env=runner.t_env))
        if gem:gem.flush()
        if track_retention and o.role != 'prefix':
            all_labels = ('stage_end', 'initial_reference', 'next_unseen')
            if FACTORIAL or RETURN_SEQUENCE:
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
                if P.get('earlier_evaluation_source'):
                    sources.insert(0, (ROOT/P['earlier_evaluation_source'], all_labels))
                if release_trial:
                    launch = json.loads((parent/'launch.json').read_text())
                    earlier_parent = launch['request'].get('parent')
                    if earlier_parent:
                        sources.insert(0, (Path(earlier_parent), all_labels))
            stage_end_steps = {}
            report = sequence_metrics(P['order'], sources + [(out, all_labels)], stage_end_steps=stage_end_steps)
            write(out/'retention.json', report)
            for index, (stage, row) in enumerate(report['metrics'].items()):
                if index in stage_end_steps:
                    step, source_run_id = stage_end_steps[index]
                    metrics.retention(index, P['order'][index], step, row,
                                      stage_label=stage, source_run_id=source_run_id)
        status('completed',finished=time.time())
    except BaseException:
        write(out/'status.json',dict(status='failed',t_env=runner.t_env if runner else 0,phase=phase,updates=updates,error=traceback.format_exc()));raise
    finally:
        if runner:runner.close_env()

if __name__=='__main__':main()
