"""Opt-in full shared-prefix artifacts, independent branch copies and provenance."""
import hashlib

from drivers.continual.memory import History, PreferenceHistory
from drivers.continual.memory.cpr import initialize_pool


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reference_input_hash(pool):
    digest = hashlib.sha256()
    for ep in pool['episodes']:
        for key in ('obs', 'actions_onehot', 'avail_actions', 'filled', 'terminated', 'old_q_values'):
            digest.update(key.encode())
            digest.update(ep['data'][key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def copy_prefix_memory(parent, out, method, history=None):
    """Every branch owns its copy of the identical A inputs; only metadata differs."""
    pool = History.load(PreferenceHistory.phase_path(parent, 0) / 'success.th')
    source_hash = reference_input_hash(pool)
    pool['reference_checkpoint'] = str(parent / pool['reference_checkpoint'])
    if method in ('tkpr_cpr', 'tkpr_cpr_anchor'):
        initialize_pool(pool)
    elif method == 'tkpr_refresh':
        pool['reference_checkpoint_role'] = 'original_collection_only'
        for ep in pool['episodes']:
            ep['reference_phase'] = pool['phase']
            ep['reference_checkpoint'] = pool['reference_checkpoint']
    History.save(PreferenceHistory.phase_path(out, 0) / 'success.th', pool)
    if history is not None:
        history[0] = pool
    return dict(parent_memory_input_hash=source_hash, branch_memory_input_hash=reference_input_hash(pool))


def full_state_extra(learner, replay_path):
    def hidden(mac):
        return mac.hidden_states.detach().cpu().clone() if mac.hidden_states is not None else None
    return dict(full_state_version=1, replay_snapshot=replay_path,
                agent=learner.mac.agent.state_dict(), mixer=learner.mixer.state_dict(),
                target_mixer=learner.target_mixer.state_dict(), optimizer=learner.optimiser.state_dict(),
                online_hidden=hidden(learner.mac), target_hidden=hidden(learner.target_mac))
