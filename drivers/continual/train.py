"""KNIFE Figure 6 a/c environments; original QMIX versus configurable A/W/V soft protection."""
import argparse
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace, ModuleType

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads(Path(os.environ['MARL_PROTOCOL']).read_text())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'vendor/smacv2'))
os.environ['SC2PATH'] = str(ROOT / 'runtime/SMACv2StarCraftII')
import numpy as np
import torch as th
import yaml
from common.checkpoints import parameter_hash, save_models
from common.metrics import Metrics, record_evaluation
from common.records import append, attach_legacy_metrics, write
from common.runtime import begin_environment_audit, episode_spec, log_task_epsilon, td_sample
from environment import SequentialProtoss
from protection import SoftProtection, NullProtection
# ReBorn envs/__init__.py imports original SMAC unconditionally. Importing both
# SMAC registries creates duplicate PySC2 map names. Supply only the SMACv2
# registry to the unchanged runner in this dedicated process.
environment_module = ModuleType('envs')
environment_module.REGISTRY = {'sequential_protoss': SequentialProtoss}
sys.modules['envs'] = environment_module
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot  # Legacy import used by offline scripts.
from controllers import REGISTRY as controllers
from learners import REGISTRY as learners
from envs import REGISTRY as environments
from runners.episode_runner import EpisodeRunner
from modules.layers.act_layer import ActivateLayer
from utils.logging import Logger


def merge(a, b):
    for k, v in b.items():
        a[k] = merge(a.get(k, {}), v) if isinstance(v, dict) else v
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algorithm', required=True, choices=PROTOCOL['algorithms'])
    p.add_argument('--condition', required=True, choices=['baseline','softprotect'])
    p.add_argument('--scenario', required=True, choices=PROTOCOL['scenarios'])
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--attempt', type=int, default=1)
    o = p.parse_args()
    out = Path(os.environ['MARL_RUN_DIR'])
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runner = None
    t_env = 0
    updates = 0
    try:
        th.set_num_threads(1)
        random.seed(o.seed)
        np.random.seed(o.seed)
        th.manual_seed(o.seed)
        th.cuda.manual_seed_all(o.seed)
        cfg = {}
        for filename in ['default.yaml', f'algs/{o.algorithm}.yaml']:
            merge(cfg, yaml.safe_load((ROOT / 'src/config' / filename).read_text()))
        merge(cfg, yaml.safe_load((ROOT / 'vendor/smacv2/smacv2/examples/configs/sc2_gen_protoss.yaml').read_text()))
        # Algorithm YAML wins for algorithm-specific values, SMACv2 for env.
        merge(cfg, yaml.safe_load((ROOT / 'src/config/algs' / f'{o.algorithm}.yaml').read_text()))
        cfg.update(seed=o.seed, device='cuda:0', env='sequential_protoss',
                   batch_size=32, buffer_size=5000, lr=.0005,
                   target_update_interval=200, hypernet_embed=64,
                   test_interval=10000, test_nepisode=32,
                   t_max=PROTOCOL['scenarios'][o.scenario][-1][0],
                   local_results_path=str(out), unique_token=out.name, save_model=False)
        cfg['env_args'].update(seed=o.seed, game_version='4.10')
        cfg['env_args']['capability_config']['team_gen']['weights'] = [1., 0., 0.]
        phases = PROTOCOL['scenarios'][o.scenario]
        if o.smoke:
            # Exercise all real transitions and all baseline algorithms. Never
            # include these shortened verification runs in formal aggregation.
            phases = [(1000*(i+1), task) for i, (_,task) in enumerate(phases)]
            cfg.update(t_max=phases[-1][0], test_interval=1000, test_nepisode=2, buffer_size=64)
        cfg['soft_protection'] = PROTOCOL['method'] if o.condition=='softprotect' else None
        cfg['scenario'] = o.scenario
        cfg.update(protocol_id=PROTOCOL['id'], condition=o.condition, phases=phases, interventions=PROTOCOL['interventions'] if o.condition=='softprotect' else [],
                   replay_on_switch='retain', optimizer_on_switch='retain',
                   epsilon_on_switch=PROTOCOL.get('epsilon_on_switch', 'per_task'),
                   phase_boundary='exact_step_time_limit_bootstrap')
        write(out / 'config.json', cfg)
        if (ROOT / 'evidence/source_manifest.json').exists():
            shutil.copy2(ROOT / 'evidence/source_manifest.json', out / 'source_manifest.json')
        write(out / 'status.json', dict(status='running', pid=os.getpid(), started=started, t_env=0))
        args = SimpleNamespace(**cfg)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        logger = Logger(logging.getLogger(out.name))
        attach_legacy_metrics(logger, out)
        metrics = Metrics(out, o.algorithm, o.condition, o.seed)
        environments['sequential_protoss'] = SequentialProtoss
        runner = EpisodeRunner(args, logger)
        info = runner.get_env_info()
        write(out / 'env_info.json', info)
        args.n_agents, args.n_actions, args.state_shape = info['n_agents'], info['n_actions'], info['state_shape']
        scheme, groups, preprocess = episode_spec(info)
        buffer = ReplayBuffer(scheme, groups, args.buffer_size, info['episode_limit']+1,
                              preprocess=preprocess, device='cpu')
        mac = controllers[args.mac](buffer.scheme, groups, args)
        runner.setup(scheme, groups, preprocess, mac)
        learner = learners[args.learner](mac, buffer.scheme, logger, args)
        # Native statistics hooks retain graphs and are irrelevant to Figure 2.
        # Remove observers only; network operations and all-one frozen gates stay.
        for network in (mac.agent, learner.target_mac.agent, learner.mixer, learner.target_mixer):
            for module in network.modules():
                if isinstance(module, ActivateLayer):
                    module._forward_hooks.clear()
        learner.cuda()
        protection = SoftProtection(learner, PROTOCOL['method'], out, append, write, metrics=metrics) if o.condition=='softprotect' else NullProtection()
        episode = 0
        initial_hash = parameter_hash(learner)
        def save_model(label):
            dest = out / 'models' / label
            save_models(learner, dest)
            write(dest / 'protection.json', protection.state())
            th.save(dict(target_agent=learner.target_mac.agent.state_dict(),
                         target_mixer=learner.target_mixer.state_dict(),
                         t_env=runner.t_env, episode=episode, updates=updates,
                         last_target_update_episode=learner.last_target_update_episode,
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         torch_rng=th.get_rng_state(), cuda_rng=th.cuda.get_rng_state_all()), dest / 'training_state.th')
        for phase, (end, unit) in enumerate(phases):
            before = parameter_hash(learner)
            protection.enter(unit, phase, runner.t_env)
            runner.env.set_task(unit)
            runner.env.phase = phase
            log_task_epsilon(runner, out, phase, unit, 'start')
            begin_environment_audit(runner, out, phase, all_resets=o.smoke)
            append(out / 'switches.jsonl', dict(phase=phase, unit=unit, t_env=runner.t_env,
                   end=end, model_hash=before, buffer_episodes=int(buffer.episodes_in_buffer),
                   updates=updates, last_target_update_episode=learner.last_target_update_episode,
                   optimizer_state_entries=len(learner.optimiser.state)))
            def evaluate(label):
                runner.env.remaining = None
                # Collect per-episode outcomes; evaluate before any training on
                # the new phase, and at the exact endpoint of the old phase.
                wins = 0
                with th.no_grad():
                    for j in range(args.test_nepisode):
                        runner.run(test_mode=True)
                        won = bool(runner.env.last_info.get('battle_won', False))
                        wins += int(won)
                        append(out / 'test_episodes.jsonl', dict(t_env=runner.t_env, phase=phase,
                               unit=unit, evaluation=label, episode=j, won=won, length=runner.t))
                record_evaluation(out, metrics, dict(t_env=runner.t_env, phase=phase, unit=unit,
                       evaluation=label, wins=wins, episodes=args.test_nepisode,
                       win_rate=wins/args.test_nepisode, wall_seconds=time.time()-started))
                write(out / 'status.json', dict(status='running', pid=os.getpid(), started=started,
                      t_env=runner.t_env, phase=phase, unit=unit, updates=updates,
                      elapsed_seconds=time.time()-started))
            evaluate('phase_start')
            next_test = min(runner.t_env + args.test_interval, end)
            while runner.t_env < end:
                runner.env.remaining = end - runner.t_env
                with th.no_grad():
                    batch = runner.run(test_mode=False)
                protection.remember(batch, episode, unit, runner.t_env-runner.t, runner.t_env, runner.env.last_info)
                buffer.insert_episode_batch(batch)
                if buffer.can_sample(args.batch_size):
                    sample = td_sample(buffer, args.batch_size, args.device)
                    protection.t_env = runner.t_env
                    learner.train(sample, runner.t_env, episode)
                    updates += 1
                    del sample
                episode += 1
                t_env = runner.t_env
                if runner.t_env >= next_test:
                    evaluate('phase_end' if runner.t_env == end else 'periodic')
                    next_test = min(next_test + args.test_interval, end)
            log_task_epsilon(runner, out, phase, unit, 'end')
            save_model(f'phase{phase}_{end}')
            if phase < len(phases)-1 and phase in PROTOCOL['method'].get('detection_phases', range(len(phases)-1)):
                protection.detect(unit, end, args, enumerate_check=o.smoke)
                write(out / 'models' / f'phase{phase}_{end}' / 'protection_after_detection.json', protection.state())
        write(out / 'protection_final.json', protection.state())
        final_hash = parameter_hash(learner)
        write(out / 'status.json', dict(status='completed', started=started, finished=time.time(),
              elapsed_seconds=time.time()-started, t_env=runner.t_env, updates=updates,
              initial_model_hash=initial_hash, final_model_hash=final_hash, interventions=PROTOCOL['interventions'] if o.condition=='softprotect' else []))
    except BaseException:
        write(out / 'status.json', dict(status='failed', started=started, finished=time.time(),
              t_env=runner.t_env if runner else t_env, updates=updates, traceback=traceback.format_exc()))
        raise
    finally:
        if runner:
            runner.close_env()


if __name__ == '__main__':
    main()
