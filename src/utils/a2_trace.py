"""Windowed, read-only diagnostics for the ReLU features entering the GRU."""
import hashlib
import json
from pathlib import Path

import torch as th

from modules.layers.trhr import valid_a2_rows


class A2Trace:
    def __init__(self, agent, controller, directory, window_updates, *, single_ff=False, stages=0):
        self.controller = controller
        self.path = Path(directory) / ("a_windows.jsonl" if single_ff else "a2_windows.jsonl")
        self.stages = stages
        self.phase = 0
        self.history = []
        self.protected = []
        self.adjusted = set()
        self.recovered = set()
        self.window_updates = window_updates
        self.layer = agent.net[2 if single_ff else 4]
        self.linear = agent.net[0 if single_ff else 3]
        self.width = self.linear.out_features
        self.streak = th.zeros(self.width, dtype=th.long)
        self.previous_offset = th.zeros(self.width)
        self.gru = agent.rnn
        if not stages:
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
        valid, target_available, active = valid_a2_rows(signs, mask, batch)
        if not self.updates:
            self.start = int(t_env)
            self.count = th.zeros((), device=valid.device, dtype=th.long)
            self.positives = th.zeros(self.width, device=valid.device, dtype=th.long)
            self.gradient_sum = th.zeros_like(gradients)
            self.gradient_nonzero = th.zeros_like(gradients, dtype=th.long)
            self.update_sum = th.zeros_like(deltas)
            self.update_nonzero = th.zeros_like(deltas, dtype=th.long)
            if self.stages:
                self.source_episodes = th.zeros(self.stages, device=valid.device, dtype=th.long)
                self.source_rows = th.zeros(self.stages, valid.shape[-1], 2, device=valid.device, dtype=th.long)
                self.source_positive = th.zeros(*self.source_rows.shape, self.width, device=valid.device, dtype=th.long)
        if self.stages:
            source = batch["source_phase"].flatten()
            self.source_episodes.index_add_(0, source, th.ones_like(source))
            for condition, rows in enumerate((valid & ~target_available, valid & target_available)):
                self.source_rows[:, :, condition].index_add_(0, source, rows.sum(1))
                self.source_positive[:, :, condition].index_add_(
                    0, source, (rows.unsqueeze(-1) & active).sum(1))
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
        if self.stages:
            offsets = th.tensor(record["offset"])
            increased = (offsets > self.previous_offset).nonzero().flatten().tolist()
            observed = record["activation_rate"]
            new_wakes = []
            if controller is not None and observed is not None and self.updates == self.window_updates:
                seen = th.tensor([i in self.adjusted for i in range(self.width)])
                above = th.tensor(observed) >= controller.settings["target_rate"]
                self.streak = th.where(seen & above, self.streak + 1, 0)
                new_wakes = sorted(set((self.streak >= controller.settings["low_windows"]).nonzero().flatten().tolist()) - self.recovered)
            self.recovered.update(new_wakes)
            self.adjusted.update(increased)
            self.previous_offset = offsets
            record.update(phase=self.phase, historical_units=self.history, protected_units=self.protected,
                          source_episodes=self.source_episodes.tolist(),
                          source_condition_rows=self.source_rows.tolist(),
                          source_condition_positives=self.source_positive.tolist(),
                          adjusted_this_window=increased, newly_woken=new_wakes,
                          adjusted_total=len(self.adjusted), woken_total=len(self.recovered))
        with self.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.updates = 0
