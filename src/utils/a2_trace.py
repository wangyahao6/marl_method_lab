"""Windowed, read-only A2 statistics for the single-task TRHR experiment."""
import hashlib
import json
from pathlib import Path

import torch as th

from modules.layers.trhr import valid_a2_rows


class A2Trace:
    def __init__(self, agent, controller, directory, window_updates):
        self.controller = controller
        self.path = Path(directory) / "a2_windows.jsonl"
        self.window_updates = window_updates
        self.layer = agent.net[4]
        self.width = agent.net[3].out_features
        self.linear = agent.net[3]
        self.gru = agent.rnn
        # Store the actual common initialization and fingerprint trainable/frozen parameters.
        state = {k: v.detach().cpu() for k, v in agent.state_dict().items()}
        th.save(state, self.path.parent / "initial_agent.th")
        digest = hashlib.sha256()
        for name, parameter in agent.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
        (self.path.parent / "initial_parameters.sha256").write_text(digest.hexdigest() + "\n")
        self.updates = 0
        self.total_updates = 0

    def _norm(self, rows):
        return th.linalg.vector_norm(rows.flatten(1), dim=1)

    @th.no_grad()
    def before_step(self):
        # Gradients are measured after the existing global gradient clipping.
        incoming, outgoing = self.linear.weight, self.gru.weight_ih
        gradients = th.stack((
            self._norm(incoming.grad),
            self._norm(outgoing.grad.T),
        ))
        return gradients, incoming.detach().clone(), outgoing.detach().clone()

    @th.no_grad()
    def observe(self, signs, mask, batch, snapshot, t_env):
        gradients, incoming, outgoing = snapshot
        deltas = th.stack((
            self._norm(self.linear.weight - incoming),
            self._norm((self.gru.weight_ih - outgoing).T),
        ))
        valid, _, active = valid_a2_rows(signs, mask, batch)
        if not self.updates:
            self.start = int(t_env)
            self.count = th.zeros((), device=valid.device, dtype=th.long)
            self.positives = th.zeros(self.width, device=valid.device, dtype=th.long)
            self.gradient_sum = th.zeros_like(gradients)
            self.gradient_nonzero = th.zeros_like(gradients, dtype=th.long)
            self.update_sum = th.zeros_like(deltas)
            self.update_nonzero = th.zeros_like(deltas, dtype=th.long)
        self.count.add_(valid.sum())
        self.positives.add_((active & valid.unsqueeze(-1)).sum(dim=(0, 1, 2)))
        self.gradient_sum.add_(gradients)
        self.gradient_nonzero.add_(gradients > 0)
        self.update_sum.add_(deltas)
        self.update_nonzero.add_(deltas > 0)
        self.updates += 1
        self.total_updates += 1
        if self.updates == self.window_updates:
            self.flush(t_env)

    @th.no_grad()
    def flush(self, t_env):
        if not self.updates:
            return
        count = self.count.item()
        controller = self.controller
        record = dict(
            start_t_env=self.start, t_env=int(t_env), total_updates=self.total_updates,
            window_updates=self.updates, valid_rows=count,
            activation_rate=(self.positives / count).tolist() if count else None,
            offset=controller.offset.tolist() if controller is not None else [0.] * self.width,
            ema_rate=controller.rate.tolist() if controller is not None else None,
            low_windows=controller.low_windows.tolist() if controller is not None else None,
            gradient_l2=(self.gradient_sum / self.updates).tolist(),
            gradient_nonzero_fraction=(self.gradient_nonzero / self.updates).tolist(),
            update_l2=(self.update_sum / self.updates).tolist(),
            update_nonzero_fraction=(self.update_nonzero / self.updates).tolist(),
        )
        with self.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.updates = 0
