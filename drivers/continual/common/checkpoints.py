"""Shared pieces of existing model and continuation formats."""
import hashlib

import torch as th

from components.optimizer_checkpoint import load_optimizer_state
from modules.agents.rnn_agent import LEGACY_FC1_KEYS

from .runtime import rng_state


def parameter_hash(learner):
    networks = (learner.mac.agent, learner.target_mac.agent, learner.mixer, learner.target_mixer)
    return hash_model_states(network.state_dict() for network in networks)


def hash_model_states(states, removed_keys=()):
    digest = hashlib.sha256()
    for index, state in enumerate(states):
        for name, value in state.items():
            if index < 2 and name in removed_keys:
                continue
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def parent_hash_fields(state, model_states=None):
    """Compare equivalent network layouts while preserving the recorded archive hash."""
    original = state['model_hash']
    if model_states is None:
        model_states = tuple(state[key] for key in ('agent', 'target_agent', 'mixer', 'target_mixer'))
    legacy = any(all(key in network for key in LEGACY_FC1_KEYS) for network in model_states[:2])
    if legacy and hash_model_states(model_states) == original:
        return dict(parent_model_hash=hash_model_states(model_states, LEGACY_FC1_KEYS),
                    archived_parent_model_hash=original)
    return dict(parent_model_hash=original)


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
    load_optimizer_state(learner.optimiser, state['optimizer'], learner.mac.agent)
