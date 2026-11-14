"""Shared pieces of existing model and continuation formats."""
import hashlib

import torch as th

from .runtime import rng_state


def parameter_hash(learner):
    digest = hashlib.sha256()
    for network in (learner.mac.agent, learner.target_mac.agent,
                    learner.mixer, learner.target_mixer):
        for name, value in network.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def cpu_snapshot(network):
    return {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}


def save_models(learner, dest):
    dest.mkdir(parents=True, exist_ok=False)
    learner.save_models(str(dest))


def save_stage_models(learner, dest, t_env, episode, updates):
    save_models(learner, dest)
    th.save(dict(target_agent=learner.target_mac.agent.state_dict(),
                 target_mixer=learner.target_mixer.state_dict(),
                 t_env=t_env, episode=episode, updates=updates), dest / 'targets.th')


def continuation_state(learner, runner, buffer, episode, updates):
    return dict(agent=learner.mac.agent.state_dict(), mixer=learner.mixer.state_dict(),
                target_agent=learner.target_mac.agent.state_dict(),
                target_mixer=learner.target_mixer.state_dict(),
                optimizer=learner.optimiser.state_dict(), buffer=buffer,
                t_env=runner.t_env, episode=episode, updates=updates,
                last_target_update_episode=learner.last_target_update_episode,
                log_stats_t=learner.log_stats_t, epsilon=learner.mac.action_selector.epsilon,
                rng=rng_state(), environment_distributions=runner.env.distribution_state(),
                model_hash=parameter_hash(learner))


def restore_networks_and_optimizer(learner, state):
    for network, key in ((learner.mac.agent, 'agent'), (learner.mixer, 'mixer'),
                         (learner.target_mac.agent, 'target_agent'),
                         (learner.target_mixer, 'target_mixer')):
        network.load_state_dict(state[key])
    learner.optimiser.load_state_dict(state['optimizer'])
