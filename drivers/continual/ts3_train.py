"""TS3 five-task training with shared A prefix and single-layer protection branches."""
import argparse,copy,json,logging,os,random,time,traceback
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import hetero_train as h
from hetero_environment import TEAMS
from hetero_protection import select as qmix_select
from ts3_vdn_detection import select as vdn_select
from ts3_protection import TS3Protection
from protection import cpu_episode,rng_state,digest
import numpy as np
import torch as th
base=h.base; ROOT=base.ROOT; write=base.write; append=base.append
P=base.PROTOCOL
TEAMS.update({key:tuple(value) for key,value in P['tasks'].items()})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['prefix','branch'],required=True)
    parser.add_argument('--parent');parser.add_argument('--group',choices=P['groups'],default='N');parser.add_argument('--seed',type=int,required=True);o=parser.parse_args()
    out=Path(os.environ['MARL_RUN_DIR']);started=time.time();runner=None;updates=episode=phase=0
    try:
        th.set_num_threads(1);random.seed(o.seed);np.random.seed(o.seed);th.manual_seed(o.seed);th.cuda.manual_seed_all(o.seed)
        cfg=h.config(o.seed,out)
        algorithm=P.get('algorithm','qmix')
        if algorithm=='vdn':base.merge(cfg,__import__('yaml').safe_load((ROOT/'src/config/algs/vdn.yaml').read_text()))
        cfg['algorithm']=algorithm;cfg['top_k']=P.get('top_k')
        cfg.update(t_max=P['thresholds'][-1],protocol_id=P['id'],role=o.role,parent=o.parent,group=o.group,screening=P['screening'],phase_boundary=P['phase_boundary'],tasks=P['tasks'],thresholds=P['thresholds'])
        args=SimpleNamespace(**cfg)
        logging.basicConfig(level=logging.INFO);logger=base.Logger(logging.getLogger(out.name));original=logger.log_stat
        def log(k,v,t,**kwargs):append(out/'metrics.jsonl',dict(key=k,value=float(v),t_env=int(t)));original(k,v,t,**kwargs)
        logger.log_stat=log
        runner=base.EpisodeRunner(args,logger);info=runner.get_env_info();args.n_agents=info['n_agents'];args.n_actions=info['n_actions'];args.state_shape=info['state_shape']
        write(out/'config.json',vars(args));write(out/'env_info.json',info)
        scheme=dict(state=dict(vshape=info['state_shape']),obs=dict(vshape=info['obs_shape'],group='agents'),actions=dict(vshape=(1,),group='agents',dtype=th.long),avail_actions=dict(vshape=(info['n_actions'],),group='agents',dtype=th.int),reward=dict(vshape=(1,)),terminated=dict(vshape=(1,),dtype=th.uint8))
        groups=dict(agents=args.n_agents);preprocess=dict(actions=('actions_onehot',[base.OneHot(out_dim=args.n_actions)]))
        buffer=base.ReplayBuffer(scheme,groups,args.buffer_size,info['episode_limit']+1,preprocess=preprocess,device='cpu')
        mac=base.controllers[args.mac](buffer.scheme,groups,args);runner.setup(scheme,groups,preprocess,mac);learner=base.learners[args.learner](mac,buffer.scheme,logger,args)
        for network in [mac.agent,learner.target_mac.agent,learner.mixer,learner.target_mixer]:
            for module in network.modules():module._forward_hooks.clear()
        learner.cuda();selection={}
        if o.role=='branch':
            parent=Path(o.parent);state=th.load(parent/'prefix_state.th',map_location='cpu')
            for network,key in [(mac.agent,'agent'),(learner.mixer,'mixer'),(learner.target_mac.agent,'target_agent'),(learner.target_mixer,'target_mixer')]:network.load_state_dict(state[key])
            learner.optimiser.load_state_dict(state['optimizer']);buffer=state['buffer'];episode=state['episode'];updates=state['updates'];learner.last_target_update_episode=state['last_target_update_episode'];learner.log_stats_t=state['log_stats_t'];runner.t_env=state['t_env']
            mac.action_selector.epsilon=state['epsilon'];runner.env.restore_distribution(state['environment_distributions']);h.restore_rng(state['rng'])
            selection=json.loads((parent/'selection.json').read_text())
            write(out/'branch_restore.json',dict(parent=str(parent),t_env=runner.t_env,episode=episode,updates=updates,buffer_episodes=buffer.episodes_in_buffer,buffer_index=buffer.buffer_index,model_hash=base.parameter_hash(learner),parent_model_hash=state['model_hash'],rng_match=digest(rng_state())==digest(state['rng']),engine='new SC2 process; recoverable distribution RNG restored'))
            del state
        layer,mode=(None,'baseline') if o.group=='N' else o.group.split('_')
        protector=TS3Protection(learner,layer,mode,lambda x:append(out/'update_audit.jsonl',x))
        if o.role=='branch' and layer:append(out/'selection_history.jsonl',dict(source_task='A',next_task='B',**protector.update(selection,runner.t_env)))
        def status(state='running',**extra):write(out/'status.json',dict(status=state,pid=os.getpid(),t_env=runner.t_env,phase=phase,task=P['order'][phase],updates=updates,episode=episode,elapsed_seconds=time.time()-started,**extra))
        def evaluate(task,focus,label):
            saved=rng_state();wins=0;idx=P['order'].index(task);blocks=[91101+idx*100+j for j in range(4)] if focus else [91001+idx*100+j for j in range(2)];per=32 if focus else 16
            try:
                for envseed in blocks:
                    ea=copy.deepcopy(args);ea.device='cpu';ea.use_cuda=False;ea.env_args=copy.deepcopy(args.env_args);ea.env_args['seed']=envseed
                    emac=base.controllers[args.mac](buffer.scheme,groups,ea);emac.agent.load_state_dict(mac.agent.state_dict());emac.agent.eval()
                    er=base.EpisodeRunner(ea,base.Logger(logging.getLogger('eval')));er.env.set_task(task);er.setup(scheme,groups,preprocess,emac)
                    try:
                        with th.no_grad():
                            for j in range(per):
                                er.run(test_mode=True);won=bool(er.env.last_info.get('battle_won',False));wins+=won
                                append(out/'test_episodes.jsonl',dict(t_env=runner.t_env,phase=phase,trained_task=P['order'][phase],task=task,label=label,focus=focus,env_seed=envseed,episode=j,won=won,length=er.t))
                    finally:er.close_env()
                append(out/'evaluations.jsonl',dict(t_env=runner.t_env,phase=phase,trained_task=P['order'][phase],task=task,label=label,focus=focus,wins=int(wins),episodes=per*len(blocks),win_rate=wins/(per*len(blocks))))
            finally:h.restore_rng(saved)
            status()
        def screen(task):
            dest=out/'detections'/f'{task}_{runner.t_env}';dest.mkdir(parents=True,exist_ok=False)
            try:result=(vdn_select if algorithm=='vdn' else qmix_select)(learner,args,fifo,dest)
            except Exception:
                write(dest/'error.json',dict(error=traceback.format_exc()));result={key:dict(sets={'overall':None}) for key in ['A1','A2']}
            write(dest/'result.json',result)
            return result
        phases=range(1) if o.role=='prefix' else range(1,5)
        for phase in phases:
            end=P['thresholds'][phase];task=P['order'][phase];begin=runner.t_env;runner.env.set_task(task);fifo=deque(maxlen=P['screen_episodes'])
            append(out/'transitions.jsonl',dict(phase=phase,task=task,actual_start=begin,nominal_end=end,buffer_episodes=buffer.episodes_in_buffer,protection=None if not layer else protector.schedule.state(begin)))
            audited=[False]
            def audit(x):
                if not audited[0]:append(out/'environment_audit.jsonl',dict(phase=phase,t_env=runner.t_env,**x));audited[0]=True
            runner.env.audit=audit;status();evaluate(task,False,'phase_start');next_test=(runner.t_env//25000+1)*25000
            while runner.t_env<end:
                with th.no_grad():batch=runner.run(test_mode=False)
                if o.role=='prefix' or layer:
                    raw=runner.env.wrapped.env;types=['stalker' if raw.agents[i].unit_type==raw.stalker_id else 'zealot' for i in range(args.n_agents)];fifo.append((cpu_episode(batch),types))
                buffer.insert_episode_batch(batch)
                if buffer.can_sample(args.batch_size):
                    sample=buffer.sample(args.batch_size);sample=sample[:,:sample.max_t_filled()];sample.to(args.device);protector.t_env=runner.t_env;learner.train(sample,runner.t_env,episode);updates+=1;del sample
                episode+=1
                if episode%25==0:status()
                if runner.t_env>=end:break
                if runner.t_env>=next_test:
                    evaluate(task,False,'periodic')
                    while next_test<=runner.t_env:next_test+=25000
            append(out/'phase_end.jsonl',dict(phase=phase,task=task,actual_start=begin,actual_end=runner.t_env,nominal_end=end,overshoot=runner.t_env-end))
            dest=out/'models'/f'phase{phase}_{task}_{runner.t_env}';dest.mkdir(parents=True,exist_ok=False);learner.save_models(str(dest))
            th.save(dict(target_agent=learner.target_mac.agent.state_dict(),target_mixer=learner.target_mixer.state_dict(),t_env=runner.t_env,episode=episode,updates=updates),dest/'targets.th')
            if layer:write(dest/'protection.json',protector.schedule.state(runner.t_env))
            for seen in P['order'][:phase+1]:evaluate(seen,True,'stage_end')
            if phase<4 and (o.role=='prefix' or layer):
                selection=screen(task)
                if layer:append(out/'selection_history.jsonl',dict(source_task=task,next_task=P['order'][phase+1],**protector.update(selection,runner.t_env)))
        if o.role=='prefix':
            write(out/'selection.json',selection)
            state=dict(agent=mac.agent.state_dict(),mixer=learner.mixer.state_dict(),target_agent=learner.target_mac.agent.state_dict(),target_mixer=learner.target_mixer.state_dict(),optimizer=learner.optimiser.state_dict(),buffer=buffer,t_env=runner.t_env,episode=episode,updates=updates,last_target_update_episode=learner.last_target_update_episode,log_stats_t=learner.log_stats_t,epsilon=mac.action_selector.epsilon,rng=rng_state(),environment_distributions=runner.env.distribution_state(),model_hash=base.parameter_hash(learner))
            th.save(state,out/'prefix_state.th')
        status('completed',finished=time.time())
    except BaseException:
        write(out/'status.json',dict(status='failed',t_env=runner.t_env if runner else 0,phase=phase,updates=updates,error=traceback.format_exc()));raise
    finally:
        if runner:runner.close_env()

if __name__=='__main__':main()
