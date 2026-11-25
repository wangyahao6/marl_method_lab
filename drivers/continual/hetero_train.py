"""Shared A prefix + resumed B/A branches for exact heterogeneous Protoss."""
import argparse,json,logging,os,random,time,traceback
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import train as base
from common.checkpoints import (continuation_state, parent_hash_fields, restore_networks_and_optimizer,
                                save_stage_models)
from common.evaluation import evaluate_block, fixed_seed_plan
from common.metrics import Metrics, record_evaluation
from common.records import attach_legacy_metrics
from common.exploration import epsilon_on_switch
from common.runtime import begin_environment_audit, episode_spec, log_phase_end, restore_rng, td_sample
import numpy as np
import torch as th
import yaml
from hetero_environment import ExactProtoss
from hetero_protection import AgentProtection,select
from protection import cpu_episode,rng_state,digest
ROOT=base.ROOT;write=base.write;append=base.append
base.environments['hetero_protoss']=ExactProtoss


def config(seed,out):
    cfg={}
    for f in ['src/config/default.yaml','vendor/smacv2/smacv2/examples/configs/sc2_gen_protoss.yaml','src/config/algs/qmix.yaml']:base.merge(cfg,yaml.safe_load((ROOT/f).read_text()))
    cfg.update(seed=seed,device='cuda:0',use_cuda=True,env='hetero_protoss',batch_size=32,buffer_size=5000,lr=.0005,target_update_interval=200,hypernet_embed=64,test_nepisode=32,test_interval=25000,t_max=2500000,local_results_path=str(out),unique_token=out.name,save_model=False)
    cfg['epsilon_on_switch'] = epsilon_on_switch(base.PROTOCOL)
    cfg['env_args'].update(seed=seed,game_version='4.10');return cfg


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['prefix','branch'],required=True);p.add_argument('--parent');p.add_argument('--task',choices=['R','M'],default='R');p.add_argument('--group',default='N');p.add_argument('--seed',type=int,default=3535);o=p.parse_args()
    out=Path(os.environ['MARL_RUN_DIR']);out.mkdir(exist_ok=True,parents=True);started=time.time();runner=None;t=0;updates=0;episode=0;phase=0
    try:
        th.set_num_threads(1);random.seed(o.seed);np.random.seed(o.seed);th.manual_seed(o.seed);th.cuda.manual_seed_all(o.seed)
        cfg=config(o.seed,out);cfg.update(role=o.role,parent=o.parent,branch_task=o.task,group=o.group,protocol_id=base.PROTOCOL['id']);args=SimpleNamespace(**cfg)
        logging.basicConfig(level=logging.INFO)
        logger = base.Logger(logging.getLogger(out.name))
        attach_legacy_metrics(logger, out)
        metrics = Metrics(out, 'qmix', ('baseline' if o.group == 'N' else 'softprotect'), o.seed, group=o.group, role=o.role)
        runner=base.EpisodeRunner(args,logger);info=runner.get_env_info();args.n_agents=info['n_agents'];args.n_actions=info['n_actions'];args.state_shape=info['state_shape']
        write(out/'config.json',vars(args));write(out/'env_info.json',info)
        scheme, groups, preprocess = spec = episode_spec(info)
        buffer=base.ReplayBuffer(scheme,groups,args.buffer_size,info['episode_limit']+1,preprocess=preprocess,device='cpu')
        mac=base.controllers[args.mac](buffer.scheme,groups,args);runner.setup(scheme,groups,preprocess,mac);learner=base.learners[args.learner](mac,buffer.scheme,logger,args)
        learner.remove_activation_hooks()
        learner.cuda();fifo=deque(maxlen=32);selection={}
        if o.role=='branch':
            parent=Path(o.parent);state=th.load(parent/'prefix_state.th',map_location='cpu')
            restore_networks_and_optimizer(learner, state)
            buffer=state['buffer'];t=state['t_env'];episode=state['episode'];updates=state['updates'];learner.last_target_update_episode=state['last_target_update_episode'];learner.log_stats_t=state['log_stats_t'];runner.t_env=t
            mac.action_selector.epsilon=state['epsilon'];runner.env.restore_distribution(state['environment_distributions']);restore_rng(state['rng'])
            selection=json.loads((parent/'selection.json').read_text())
            write(out/'branch_restore.json',dict(parent=str(parent),t_env=t,episode=episode,updates=updates,buffer_episodes=buffer.episodes_in_buffer,buffer_index=buffer.buffer_index,optimizer_entries=len(learner.optimiser.state),last_target_update_episode=learner.last_target_update_episode,model_hash=base.parameter_hash(learner),**parent_hash_fields(state),rng_match=digest(rng_state())==digest(state['rng']),engine='new process, common seed; distribution RNG restored'))
            del state
        protection=AgentProtection(learner,selection,o.group if o.role=='branch' else 'N',lambda r:append(out/'update_audit.jsonl',r))
        write(out/'protection.json',dict(ids=protection.ids,unique_parameters=protection.unique_parameters,group=o.group))
        def status(state='running',**extra):write(out/'status.json',dict(status=state,t_env=runner.t_env,phase=phase,updates=updates,episode=episode,elapsed_seconds=time.time()-started,**extra))
        def evaluate(task,focus,label):
            saved = rng_state()
            wins = 0
            idx = 0 if task == 'A' else 1
            blocks, per = fixed_seed_plan(idx, focus, stride=1000)
            context = dict(t_env=runner.t_env, phase=phase, task=task, label=label, focus=focus)
            try:
                for envseed in blocks:
                    rows = evaluate_block(
                        args, spec, buffer.scheme, mac.agent.state_dict(), task, envseed, per,
                        make_controller=base.controllers[args.mac], make_runner=base.EpisodeRunner,
                        make_logger=base.Logger, include_return=True,
                        record=lambda row: append(out/'test_episodes.jsonl', dict(context, **row)))
                    wins += sum(row['won'] for row in rows)
                record_evaluation(out, metrics, dict(
                    context, wins=int(wins), episodes=per*len(blocks), win_rate=wins/(per*len(blocks))))
            finally:restore_rng(saved)
            status()
        if o.role=='prefix':phases=[(1000000,'A',0)]
        else:phases=[(1500000,o.task,1),(2500000,'A',2)]
        for end,task,phase in phases:
            begin=runner.t_env;runner.env.set_task(task);protection.active=phase==1 and o.group!='N'
            base.log_task_epsilon(runner, out, phase, task, 'start')
            append(out/'transitions.jsonl',dict(phase=phase,task=task,actual_start=begin,nominal_end=end,active=protection.active,scale=.1 if protection.active else 1.,buffer_episodes=buffer.episodes_in_buffer,model_hash=base.parameter_hash(learner)))
            begin_environment_audit(runner, out, phase)
            for layer, ids in protection.ids.items():
                width = mac.agent.net[0 if layer == 'A1' else 3].out_features
                metrics.record('Mechanism', phase, task, begin,
                               {'protected_neuron_fraction': (len(ids) if protection.active else 0) / width},
                               event='phase_start', layer=layer)
            if phase!=2:evaluate(task,False,'phase_start')
            next_test=(runner.t_env//25000+1)*25000;next_old=(runner.t_env//100000+1)*100000
            while runner.t_env<end:
                with th.no_grad():batch=runner.run(test_mode=False)
                if o.role=='prefix':
                    raw=runner.env.wrapped.env;types=['stalker' if raw.agents[i].unit_type==raw.stalker_id else 'zealot' for i in range(args.n_agents)];fifo.append((cpu_episode(batch),types))
                buffer.insert_episode_batch(batch)
                if buffer.can_sample(args.batch_size):
                    sample=td_sample(buffer, args.batch_size, args.device);protection.t_env=runner.t_env;learner.train(sample,runner.t_env,episode);updates+=1;del sample
                episode+=1;t=runner.t_env
                if t>=end:break
                if t>=next_test:
                    evaluate(task,False,'periodic')
                    while next_test<=t:next_test+=25000
                if phase==1 and t>=next_old:
                    evaluate('A',False,'retention')
                    while next_old<=t:next_old+=100000
                if episode%100==0:status()
            base.log_task_epsilon(runner, out, phase, task, 'end')
            log_phase_end(out, phase, task, begin, end, runner.t_env)
            save_stage_models(learner, out/'models'/f'phase{phase}_{runner.t_env}',
                              runner.t_env, episode, updates)
            if phase==1:evaluate(task,True,'B_end');evaluate('A',True,'return_A_before_update')
            else:evaluate('A',True,'A1_end' if phase==0 else 'A2_end')
        if o.role=='prefix':
            before_screen=digest([base.parameter_hash(learner),learner.optimiser.state_dict(),rng_state()])
            try:selection=select(learner,args,fifo,out);write(out/'selection.json',selection)
            except Exception:
                write(out/'selection_error.json',dict(error=traceback.format_exc()));selection={k:dict(T=[],G=[],R=[],k=0,defined=False) for k in ['A1','A2']};write(out/'selection.json',selection)
            for layer, item in selection.items():
                for group_name, scores in item.get('scores', {}).items():
                    metrics.neurons(phase, task, runner.t_env, layer,
                                    {'weighted_activity': scores['activity'], 'normalized_score': scores['score']},
                                    event='boundary_detection', agent_group=group_name)
            write(out/'screen_audit.json',dict(training_state_unchanged=before_screen==digest([base.parameter_hash(learner),learner.optimiser.state_dict(),rng_state()])))
            state = continuation_state(learner, runner, buffer, episode, updates)
            th.save(state,out/'prefix_state.th');write(out/'prefix_identity.json',dict(t_env=runner.t_env,episode=episode,updates=updates,buffer_index=buffer.buffer_index,buffer_episodes=buffer.episodes_in_buffer,model_hash=state['model_hash']))
        status('completed',finished=time.time())
    except BaseException:
        write(out/'status.json',dict(status='failed',t_env=runner.t_env if runner else t,updates=updates,error=traceback.format_exc()));raise
    finally:
        if runner:runner.close_env()

if __name__=='__main__':main()
