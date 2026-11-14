"""Independent fixed-seed retention evaluation; never touches training replay/RNG."""
import argparse
import json
import logging
import random
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
import train as t
from common.metrics import Metrics
from common.runtime import episode_spec
import numpy as np
import torch as th

TASKS = ['stalker', 'zealot', 'colossus']
COMBINATIONS = [(0,1000000,[0]), (1,1500000,[0,1]),
                (2,2000000,[0,1,2]), (3,3000000,[0,1,2])]


def evaluate(run, out, seed, episodes=32, combinations=COMBINATIONS):
    out.mkdir(parents=True, exist_ok=True)
    records = []
    cfg_original = json.loads((run/'config.json').read_text())
    request = json.loads((run/'launch.json').read_text())['request'] if (run/'launch.json').is_file() else {}
    metrics = Metrics(out, request.get('algorithm', cfg_original.get('algorithm', 'qmix')),
                      request.get('method', cfg_original.get('condition', 'baseline')), seed, run_id=run.name)
    t.write(out/'status.json', dict(status='running', started=time.time()))
    for phase, step, task_ids in combinations:
        for task_id in task_ids:
            key = f'{step}_{TASKS[task_id]}'
            target = out/(key+'.json')
            if target.exists():
                existing = json.loads(target.read_text())
                assert existing['episodes'] == episodes and existing['training_seed'] == seed
                records.append(existing); continue
            eval_seed = 100000+100*seed+task_id+1
            random.seed(eval_seed); np.random.seed(eval_seed)
            th.manual_seed(eval_seed); th.cuda.manual_seed_all(eval_seed)
            cfg = json.loads(json.dumps(cfg_original))
            cfg.update(test_nepisode=episodes, seed=eval_seed)
            cfg['env_args']['seed'] = eval_seed
            args = SimpleNamespace(**cfg)
            logger = t.Logger(logging.getLogger('offline'))
            runner = t.EpisodeRunner(args, logger)
            try:
                info = runner.get_env_info()
                args.n_agents, args.n_actions, args.state_shape = info['n_agents'], info['n_actions'], info['state_shape']
                scheme, groups, preprocess = episode_spec(info)
                buffer = t.ReplayBuffer(scheme, groups, 1, info['episode_limit']+1, preprocess=preprocess, device='cpu')
                mac = t.controllers[args.mac](buffer.scheme, groups, args)
                mac.cuda()
                checkpoint = run/'models'/f'phase{phase}_{step}'/'agent.th'
                mac.agent.load_state_dict(th.load(checkpoint, map_location=args.device))
                runner.setup(scheme, groups, preprocess, mac)
                # Initialize with Stalkers, as in continuous training. This reset has
                # no actions; mark it discarded to avoid launching a second SC2.
                runner.env.reset()
                raw = runner.env.wrapped.env
                raw._episode_count += 1
                runner.env.set_task(TASKS[task_id])
                outcomes = []
                for i in range(episodes):
                    with th.no_grad(): runner.run(test_mode=True)
                    outcomes.append(dict(index=i, won=bool(runner.env.last_info.get('battle_won',False)), length=runner.t))
                record = dict(training_seed=seed, evaluation_seed=eval_seed, t_env=step, phase=phase,
                    task=TASKS[task_id], episodes=episodes, wins=sum(x['won'] for x in outcomes),
                    win_rate=sum(x['won'] for x in outcomes)/episodes, outcomes=outcomes,
                    checkpoint=str(checkpoint), max_reward=raw.max_reward)
                t.write(target, record); records.append(record)
                metrics.evaluation(dict(record, label='independent_checkpoint'))
            finally:
                runner.close_env()
    t.write(out/'results.json', records)
    t.write(out/'status.json', dict(status='completed', combinations=len(records),
                                   episodes=sum(r['episodes'] for r in records), finished=time.time()))


if __name__ == '__main__':
    th.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--episodes',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((a.run/'config.json').read_text());state=json.loads((a.run/'status.json').read_text())
    assert state['status']=='completed' and state['t_env']==cfg['t_max']
    TASKS=[cfg['phases'][-1][1]]
    evaluate(a.run, a.output, cfg['seed'],a.episodes,[(len(cfg['phases'])-1,cfg['t_max'],[0])])
