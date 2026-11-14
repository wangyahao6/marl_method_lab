"""TS4 VDN, single-layer soft protection and observational neuron tracking."""
import argparse,copy,json,logging,os,random,time,traceback
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import hetero_train as h
from common.checkpoints import cpu_snapshot, save_stage_models
from common.evaluation import evaluate_block, fixed_seed_plan
from common.metrics import Metrics, record_evaluation
from common.records import attach_legacy_metrics
from memory import History
from memory.episodes import combine, cpu_episode
from common.runtime import begin_environment_audit, episode_spec, log_phase_end, restore_rng, td_sample, write_sequence_status
from ts4_diagnostics import score,compare,vectors
from environment import SequentialProtoss
from ts3_protection import TS3Protection
from protection import rng_state
from components.task_replay import log_first_replay_batch, make_replay_buffer, start_replay_task
import numpy as np
import torch as th
base=h.base; ROOT=base.ROOT; write=base.write; append=base.append
P=base.PROTOCOL
class TS4Protoss(SequentialProtoss):
    def distribution_state(self):
        return copy.deepcopy(self.wrapped.env_key_to_distribution_map)

    def restore_distribution(self, state):
        self.wrapped.env_key_to_distribution_map = copy.deepcopy(state)
        self.set_task(self.unit)

    def reset(self):
        audit=self.audit;self.audit=None
        try:result=super().reset()
        finally:self.audit=audit
        raw=self.wrapped.env
        raw.max_reward=raw.n_enemies*raw.reward_death_value+raw.reward_win+sum(u.health_max+u.shield_max for u in raw.enemies.values())
        if audit:audit(dict(unit=self.unit,ally_types=[int(u.unit_type) for u in raw.agents.values()],enemy_types=[int(u.unit_type) for u in raw.enemies.values()],obs_shape=self.get_obs_size(),state_shape=self.get_state_size(),max_reward=float(raw.max_reward)))
        return result
base.environments['ts4_protoss']=TS4Protoss

def main():
    parser=argparse.ArgumentParser();
    parser.add_argument('--group',choices=P['groups'],default='N');parser.add_argument('--seed',type=int,required=True);o=parser.parse_args()
    out=Path(os.environ['MARL_RUN_DIR']);started=time.time();runner=None;updates=episode=phase=0
    try:
        th.set_num_threads(1);random.seed(o.seed);np.random.seed(o.seed);th.manual_seed(o.seed);th.cuda.manual_seed_all(o.seed)
        cfg=h.config(o.seed,out)
        base.merge(cfg,__import__('yaml').safe_load((ROOT/'src/config/algs/vdn.yaml').read_text()))
        cfg.update(env='ts4_protoss',t_max=P['thresholds'][-1],protocol_id=P['id'],group=o.group,screening=P['screening'],phase_boundary=P['phase_boundary'],tasks=P['order'],thresholds=P['thresholds'])
        cfg['env_args']['capability_config']['team_gen']['weights']=[1.,0.,0.]
        if 'replay_on_switch' in P:
            cfg['replay_on_switch'] = P['replay_on_switch']
        args=SimpleNamespace(**cfg)
        logging.basicConfig(level=logging.INFO)
        logger = base.Logger(logging.getLogger(out.name))
        attach_legacy_metrics(logger, out)
        metrics = Metrics(out, 'vdn', ('baseline' if o.group == 'N' else 'softprotect'), o.seed, group=o.group)
        runner=base.EpisodeRunner(args,logger);info=runner.get_env_info();args.n_agents=info['n_agents'];args.n_actions=info['n_actions'];args.state_shape=info['state_shape']
        write(out/'config.json',vars(args));write(out/'env_info.json',info)
        scheme, groups, preprocess = spec = episode_spec(info)
        buffer = make_replay_buffer(args, P['order'], scheme, groups,
                                    info['episode_limit'] + 1, preprocess)
        mac=base.controllers[args.mac](buffer.scheme,groups,args);runner.setup(scheme,groups,preprocess,mac);learner=base.learners[args.learner](mac,buffer.scheme,logger,args)
        for network in [mac.agent,learner.target_mac.agent,learner.mixer,learner.target_mixer]:
            for module in network.modules():module._forward_hooks.clear()
        learner.cuda();selection={}
        layer,mode=(None,'baseline') if o.group=='N' else o.group.split('_')
        protector=TS3Protection(learner,layer,mode,lambda x:append(out/'update_audit.jsonl',x))
        anchors = History()
        def status(state='running', **extra):
            write_sequence_status(out, state, runner.t_env, phase, P['order'][phase],
                                  updates, episode, started, **extra)
        def evaluate(task,focus,label):
            saved = rng_state()
            wins = 0
            idx = ['stalker', 'zealot', 'colossus'].index(task)
            blocks, per = fixed_seed_plan(idx, focus)
            context = dict(t_env=runner.t_env, phase=phase, trained_task=P['order'][phase], task=task, label=label, focus=focus)
            try:
                for envseed in blocks:
                    rows = evaluate_block(
                        args, spec, buffer.scheme, mac.agent.state_dict(), task, envseed, per,
                        make_controller=base.controllers[args.mac], make_runner=base.EpisodeRunner,
                        make_logger=base.Logger,
                        record=lambda row: append(out/'test_episodes.jsonl', dict(context, **row)))
                    wins += sum(row['won'] for row in rows)
                record_evaluation(out, metrics, dict(
                    context, wins=int(wins), episodes=per*len(blocks), win_rate=wins/(per*len(blocks))))
            finally:restore_rng(saved)
            status()
        def diagnose(label,anchor=False):
            dest=out/'detections'/f'phase{phase}_{label}_{runner.t_env}';dest.mkdir(parents=True,exist_ok=False)
            try:
                cpu=combine(list(fifo));result=score(learner,args,cpu)
                write(dest/'result.json',result)
                History.save(dest/'screen_input.th', dict(batch=cpu, agent=cpu_snapshot(mac.agent), scores=result))
                current=vectors(mac.agent)
                if anchors:
                    last=list(anchors.values())[-1]
                    append(out/'neuron_changes.jsonl',dict(t_env=runner.t_env,phase=phase,label=label,kind='different_task_data',source_phase=last['phase'],layers=compare(last['scores'],result,last['vectors'],current)))
                for key,ref in anchors.items():
                    fixed=score(learner,args,ref['batch'])
                    write(dest/f'fixed_phase{key}.json',fixed)
                    append(out/'neuron_changes.jsonl',dict(t_env=runner.t_env,phase=phase,label=label,kind='fixed_old_task_histories',source_phase=key,layers=compare(ref['scores'],fixed,ref['vectors'],current)))
                append(out/'detection_counts.jsonl',dict(t_env=runner.t_env,phase=phase,label=label,counts={k:v['count'] for k,v in result.items()},ids={k:v['sets']['overall'] for k,v in result.items()}))
                if anchor:anchors[phase]=dict(phase=phase,batch=cpu,scores=result,vectors=current)
                for layer_name, item in result.items():
                    metrics.neurons(phase, P['order'][phase], runner.t_env, layer_name,
                                    {'weighted_activity': item['activity'], 'normalized_score': item['scores']},
                                    event=label)
                return result
            except Exception:
                write(dest/'error.json',dict(error=traceback.format_exc()))
                return {key:dict(sets={'overall':None}) for key in ['A1','A2']}
        phases=range(4)
        for phase in phases:
            end=P['thresholds'][phase];task=P['order'][phase];begin=runner.t_env;runner.env.set_task(task);fifo=deque(maxlen=P['screen_episodes'])
            replay_state = start_replay_task(buffer, task)
            base.log_task_epsilon(runner, out, phase, task, 'start')
            schedule = protector.schedule.state(begin) if layer else None
            append(out/'transitions.jsonl',dict(phase=phase,task=task,actual_start=begin,nominal_end=end,buffer_episodes=buffer.episodes_in_buffer,protection=schedule,**replay_state))
            if schedule is not None:
                metrics.protection(phase, task, begin, schedule)
            begin_environment_audit(runner, out, phase)
            status();evaluate(task,False,'phase_start');next_test=(runner.t_env//25000+1)*25000;next_old=(runner.t_env//100000+1)*100000;early_done=False
            if phase>0:
                for old_task in dict.fromkeys(P['order'][:phase]):
                    evaluate(old_task,True,'switch_before_training')
            while runner.t_env<end:
                with th.no_grad():batch=runner.run(test_mode=False)
                fifo.append(cpu_episode(batch))
                buffer.insert_episode_batch(batch)
                if buffer.can_sample(args.batch_size):
                    sample=td_sample(buffer, args.batch_size, args.device);protector.t_env=runner.t_env;learner.train(sample,runner.t_env,episode);updates+=1;del sample
                    if replay_state:
                        log_first_replay_batch(buffer, append, out, phase, task, runner.t_env)
                        replay_state = {}
                episode+=1
                if phase>0 and len(fifo)==P['screen_episodes'] and not early_done:
                    diagnose('first32_new_task');early_done=True
                if phase>0 and runner.t_env>=next_old and runner.t_env<end:
                    for old_task in dict.fromkeys(P['order'][:phase]):
                        if old_task!=task:evaluate(old_task,False,'retention')
                    while next_old<=runner.t_env:next_old+=100000
                if episode%25==0:status()
                if runner.t_env>=end:break
                if runner.t_env>=next_test:
                    evaluate(task,False,'periodic')
                    while next_test<=runner.t_env:next_test+=25000
            base.log_task_epsilon(runner, out, phase, task, 'end')
            log_phase_end(out, phase, task, begin, end, runner.t_env)
            dest = out/'models'/f'phase{phase}_{task}_{runner.t_env}'
            save_stage_models(learner, dest, runner.t_env, episode, updates)
            if layer:write(dest/'protection.json',protector.schedule.state(runner.t_env))
            for seen in dict.fromkeys(P['order'][:phase+1]):evaluate(seen,True,'stage_end')
            selection=diagnose('boundary',anchor=True)
            if phase<3 and layer:append(out/'selection_history.jsonl',dict(source_phase=phase,next_phase=phase+1,**protector.update(selection,runner.t_env)))
        status('completed',finished=time.time())
    except BaseException:
        write(out/'status.json',dict(status='failed',t_env=runner.t_env if runner else 0,phase=phase,updates=updates,error=traceback.format_exc()));raise
    finally:
        if runner:runner.close_env()

if __name__=='__main__':main()
