"""Fixed historical probes for CHI-inspired shared-Agent VDN soft protection."""
import copy

import torch as th

from protection import combine
from ts4_diagnostics import vdn_sensitivity


PROBE_EPISODES = 32
LAYER_INDICES = {'A': 2, 'A1': 2, 'A2': 4}


def decision_masks(batch):
    avail = batch['avail_actions'][:, :-1].bool()
    actions = batch['actions'][:, :-1]
    # SMAC reserves action zero for dead agents. The last filled state is
    # bootstrap-only, including for episodes truncated at the time limit.
    alive = avail[..., 1:].any(-1)
    chosen_available = avail.gather(-1, actions).squeeze(-1)
    agents = alive & chosen_available
    filled = batch['filled'].squeeze(-1).bool()
    lengths = filled.sum(1) - 1
    positions = th.arange(avail.shape[1], device=avail.device)
    states = filled[:, :-1] & (positions < lengths[:, None])
    terminated = batch['terminated'][:, :-1, 0].bool()
    states[:, 1:] &= ~terminated[:, :-1].cummax(1).values
    return states & agents.any(-1), agents


def forward(mac, batch, layer=None):
    # Keep the training controller's recurrent state intact. Each batch row
    # starts from zero and advances along its own recorded history.
    probe_mac = copy.copy(mac)
    was_training = mac.agent.training
    captured = []
    handle = None
    if layer is not None:
        handle = mac.agent.net[LAYER_INDICES[layer]].register_forward_hook(
            lambda module, inputs, output: captured.append(output.detach()))
    mac.agent.eval()
    try:
        probe_mac.init_hidden(batch.batch_size)
        outputs = []
        for t in range(batch.max_seq_length - 1):
            q = probe_mac.forward(batch, t=t)
            if layer is None:
                outputs.append(q)
        if layer is None:
            return th.stack(outputs, dim=1)
        return th.stack(captured, dim=1).reshape(batch.batch_size, -1, mac.n_agents, captured[0].shape[-1])
    finally:
        if handle is not None:
            handle.remove()
        mac.agent.train(was_training)


@th.no_grad()
def save_probe(mac, episodes, path):
    save_probe_batch(mac, combine(episodes), path)


@th.no_grad()
def save_probe_batch(mac, cpu, path):
    batch = copy.deepcopy(cpu)
    batch.to(mac.args.device)
    states, agents = decision_masks(batch)
    weights = th.zeros(states.shape, device=states.device)
    if states.any():
        q = forward(mac, batch)[states]
        sensitivity = vdn_sensitivity(q, batch['actions'][:, :-1, :, 0][states],
                                      batch['avail_actions'][:, :-1][states], agents[states])
        mean = sensitivity.mean()
        if mean > 0:
            weights[states] = sensitivity / mean
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    th.save(dict(batch=cpu, weights=weights.cpu()), temp)
    temp.replace(path)


@th.no_grad()
def reassess(mac, probes, layer, *, top_k, threshold):
    tasks = {}
    current = []
    for phase, path in probes.items():
        probe = th.load(path, map_location='cpu')
        batch = probe['batch']
        batch.to(mac.args.device)
        weights = probe['weights'].to(mac.args.device)
        states, agents = decision_masks(batch)
        scores = None
        if states.any():
            activations = forward(mac, batch, layer)[states].abs()
            valid_agents = agents[states].unsqueeze(-1)
            activity = activations.masked_fill(~valid_agents, 0).sum(1) / valid_agents.sum(1)
            activity = (weights[states][:, None] * activity).mean(0)
            layer_mean = activity.mean()
            if layer_mean > 0:
                scores = (activity / layer_mean).cpu()
                current.append(scores)
        tasks[phase] = dict(probe=str(path), episodes=batch.batch_size, states=int(states.sum()),
                            scores=None if scores is None else scores.tolist())

    scores = th.stack(current).amax(0).tolist() if current else None
    chosen = []
    if scores is not None:
        if top_k is None:
            chosen = [i for i, value in enumerate(scores) if value > threshold]
        else:
            chosen = sorted(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k])
    return {layer: dict(sets={'overall': chosen}, scores=scores, tasks=tasks, defined=scores is not None)}
