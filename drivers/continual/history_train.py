"""VDN task/history endpoint analysis: one resumable 2M stage per prefix node."""
import argparse,copy,json,logging,os,random,time,traceback
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import hetero_train as h
from ts4_diagnostics import score
from protection import combine
from environment import SequentialProtoss
from protection import cpu_episode,rng_state,digest
import numpy as np
import torch as th
base=h.base; ROOT=base.ROOT; write=base.write; append=base.append
P=base.PROTOCOL
class TS4Protoss(SequentialProtoss):
    def reset(self):
        audit=self.audit;self.audit=None
        try:result=super().reset()
        finally:self.audit=audit
        raw=self.wrapped.env
        raw.max_reward=raw.n_enemies*raw.reward_death_value+raw.reward_win+sum(u.health_max+u.shield_max for u in raw.enemies.values())
        if audit:audit(dict(unit=self.unit,ally_types=[int(u.unit_type) for u in raw.agents.values()],enemy_types=[int(u.unit_type) for u in raw.enemies.values()],obs_shape=self.get_obs_size(),state_shape=self.get_state_size(),max_reward=float(raw.max_reward)))
        return result
class HistoryProtoss(TS4Protoss):
    def distribution_state(self):
        return copy.deepcopy(self.wrapped.env_key_to_distribution_map)
    def restore_distribution(self,state):
        self.wrapped.env_key_to_distribution_map=copy.deepcopy(state)
        self.set_task(self.unit)
base.environments['history_protoss']=HistoryProtoss

def main():
    parser=argparse.ArgumentParser();
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--parent');o=parser.parse_args();o.group='N'
    out=Path(os.environ['MARL_RUN_DIR']);started=time.time();runner=None;updates=episode=phase=0
    try:
        th.set_num_threads(1);random.seed(o.seed);np.random.seed(o.seed);th.manual_seed(o.seed);th.cuda.manual_seed_all(o.seed)
        cfg=h.config(o.seed,out)
        base.merge(cfg,__import__('yaml').safe_load((ROOT/'src/config/algs/vdn.yaml').read_text()))
        cfg.update(env='history_protoss',t_max=P['thresholds'][-1],protocol_id=P['id'],group=o.group,screening=P['screening'],phase_boundary=P['phase_boundary'],tasks=P['order'],thresholds=P['thresholds'])
        cfg['env_args']['capability_config']['team_gen']['weights']=[1.,0.,0.]
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
        learner.cuda()
        if o.parent:
            state=th.load(Path(o.parent)/'continuation.th',map_location='cpu')
            for network,key in [(mac.agent,'agent'),(learner.mixer,'mixer'),(learner.target_mac.agent,'target_agent'),(learner.target_mixer,'target_mixer')]:network.load_state_dict(state[key])
            learner.optimiser.load_state_dict(state['optimizer']);buffer=state['buffer'];episode=state['episode'];updates=state['updates']
            learner.last_target_update_episode=state['last_target_update_episode'];learner.log_stats_t=state['log_stats_t'];runner.t_env=state['t_env']
            mac.action_selector.epsilon=state['epsilon'];runner.env.restore_distribution(state['environment_distributions']);h.restore_rng(state['rng'])
            write(out/'branch_restore.json',dict(parent=o.parent,t_env=runner.t_env,episode=episode,updates=updates,buffer_episodes=buffer.episodes_in_buffer,model_hash=base.parameter_hash(learner),parent_model_hash=state['model_hash'],rng_match=digest(rng_state())==digest(state['rng']),engine='new SC2 process, distribution RNG restored'))
            del state
        def status(state='running',**extra):write(out/'status.json',dict(status=state,pid=os.getpid(),t_env=runner.t_env,phase=phase,task=P['order'][phase],updates=updates,episode=episode,elapsed_seconds=time.time()-started,**extra))
        def evaluate(task,focus,label):
            saved=rng_state();wins=0;idx=['stalker','zealot','colossus'].index(task);blocks=[91101+idx*100+j for j in range(4)] if focus else [91001+idx*100+j for j in range(2)];per=32 if focus else 16
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
        def diagnose(label):
            dest=out/'detections'/f'phase{phase}_{label}_{runner.t_env}';dest.mkdir(parents=True,exist_ok=False)
            try:
                cpu=combine(list(fifo));result=score(learner,args,cpu)
                write(dest/'result.json',result)
                th.save(dict(batch=cpu,agent={k:v.detach().cpu().clone() for k,v in mac.agent.state_dict().items()},scores=result),dest/'screen_input.th')
                append(out/'detection_counts.jsonl',dict(t_env=runner.t_env,phase=phase,label=label,counts={k:v['count'] for k,v in result.items()},ids={k:v['sets']['overall'] for k,v in result.items()}))
                return result
            except Exception:
                write(dest/'error.json',dict(error=traceback.format_exc()))
                return {key:dict(sets={'overall':None}) for key in ['A1','A2']}
        phases=[len(P['order'])-1]
        for phase in phases:
            end=P['thresholds'][phase];task=P['order'][phase];begin=runner.t_env;runner.env.set_task(task);fifo=deque(maxlen=P['screen_episodes'])
            append(out/'transitions.jsonl',dict(phase=phase,task=task,actual_start=begin,nominal_end=end,buffer_episodes=buffer.episodes_in_buffer,protection=None))
            audited=[False]
            def audit(x):
                if not audited[0]:append(out/'environment_audit.jsonl',dict(phase=phase,t_env=runner.t_env,**x));audited[0]=True
            runner.env.audit=audit;status();evaluate(task,False,'phase_start');next_test=(runner.t_env//25000+1)*25000
            while runner.t_env<end:
                with th.no_grad():batch=runner.run(test_mode=False)
                fifo.append(cpu_episode(batch))
                buffer.insert_episode_batch(batch)
                if buffer.can_sample(args.batch_size):
                    sample=buffer.sample(args.batch_size);sample=sample[:,:sample.max_t_filled()];sample.to(args.device);learner.train(sample,runner.t_env,episode);updates+=1;del sample
                episode+=1
                if episode%25==0:status()
                if runner.t_env>=end:break
                if runner.t_env>=next_test:
                    evaluate(task,False,'periodic')
                    while next_test<=runner.t_env:next_test+=25000
            append(out/'phase_end.jsonl',dict(phase=phase,task=task,actual_start=begin,actual_end=runner.t_env,nominal_end=end,overshoot=runner.t_env-end))
            dest=out/'models'/f'phase{phase}_{task}_{runner.t_env}';dest.mkdir(parents=True,exist_ok=False);learner.save_models(str(dest))
            th.save(dict(target_agent=learner.target_mac.agent.state_dict(),target_mixer=learner.target_mixer.state_dict(),t_env=runner.t_env,episode=episode,updates=updates),dest/'targets.th')
            for seen in dict.fromkeys(P['order'][:phase+1]):evaluate(seen,True,'stage_end')
            diagnose('boundary')
        if len(P['order'])<3:
            state=dict(agent=mac.agent.state_dict(),mixer=learner.mixer.state_dict(),target_agent=learner.target_mac.agent.state_dict(),target_mixer=learner.target_mixer.state_dict(),optimizer=learner.optimiser.state_dict(),buffer=buffer,t_env=runner.t_env,episode=episode,updates=updates,last_target_update_episode=learner.last_target_update_episode,log_stats_t=learner.log_stats_t,epsilon=mac.action_selector.epsilon,rng=rng_state(),environment_distributions=runner.env.distribution_state(),model_hash=base.parameter_hash(learner))
            th.save(state,out/'continuation.th')
        status('completed',finished=time.time())
    except BaseException:
        write(out/'status.json',dict(status='failed',t_env=runner.t_env if runner else 0,phase=phase,updates=updates,error=traceback.format_exc()));raise
    finally:
        if runner:runner.close_env()

if __name__=='__main__':main()
