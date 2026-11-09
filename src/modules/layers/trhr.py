"""Target-rate control for the A2 ReLU of the shared VDN agent."""
from contextlib import contextmanager

import torch as th
from torch import nn


@contextmanager
def capture_activity(layer):
    """Capture detached activation signs for one online learner unroll."""
    signs = []

    def collect(_module, _inputs, output):
        signs.append(output.detach() > 0)

    hook = layer.register_forward_hook(collect)
    try:
        yield signs
    finally:
        hook.remove()


def valid_a2_rows(signs, mask, batch):
    available = batch["avail_actions"][:, :-1].bool()
    # The final stored state is only a bootstrap input, including at time limits.
    valid = mask.bool() & batch["filled"][:, 1:].bool()
    alive = available[..., 1:].any(dim=-1)
    valid = valid & alive
    target_available = available[..., 6:].any(dim=-1)
    batch_size, time_steps, n_agents = valid.shape
    active = th.stack(signs[:-1]).reshape(time_steps, batch_size, n_agents, -1)
    active = active.transpose(0, 1)

    return valid, target_available, active


class TRHRReLU(nn.ReLU):
    def __init__(self, width, n_agents, settings):
        super().__init__()
        self.settings = settings
        self.register_buffer("offset", th.zeros(width))
        self.register_buffer("protected", th.zeros(width, dtype=th.bool), persistent=False)
        self.register_buffer("rate", th.zeros(width))
        self.register_buffer("condition_rate", th.zeros(n_agents, 2, width))
        self.register_buffer("condition_seen", th.zeros(n_agents, 2, dtype=th.bool))
        self.register_buffer("low_windows", th.zeros(width, dtype=th.long))
        self.register_buffer("window_updates", th.zeros((), dtype=th.long))
        self.register_buffer("sample_count", th.zeros(n_agents, 2, dtype=th.long))
        self.register_buffer("positive_count", th.zeros(n_agents, 2, width, dtype=th.long))

    def forward(self, inputs):
        return super().forward(inputs + self.offset)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # The baseline ReLU has no state. An old checkpoint starts TRHR at zero.
        if not any(key.startswith(prefix) for key in state_dict):
            for name, buffer in self.named_buffers():
                if name not in self._non_persistent_buffers_set:
                    state_dict[prefix + name] = th.zeros_like(buffer)
        super()._load_from_state_dict(state_dict, prefix, *args)

    @th.no_grad()
    def observe(self, signs, mask, batch, logger, t_env):
        valid, target_available, active = valid_a2_rows(signs, mask, batch)

        for index, condition in enumerate((~target_available, target_available)):
            rows = valid & condition
            self.sample_count[:, index].add_(rows.sum(dim=(0, 1)))
            self.positive_count[:, index].add_((active & rows.unsqueeze(-1)).sum(dim=(0, 1)))

        self.window_updates.add_(1)
        if self.window_updates.item() >= self.settings["window_updates"]:
            self._adjust(logger, t_env)
            self.window_updates.zero_()
            self.sample_count.zero_()
            self.positive_count.zero_()

    def _adjust(self, logger, t_env):
        settings = self.settings
        supported = self.sample_count >= settings["min_condition_samples"]
        enough_data = supported.any()
        beta = settings["ema_weight"]
        total = self.sample_count.sum()
        window_rate = self.positive_count.sum(dim=(0, 1)) / total.clamp_min(1)
        next_rate = th.where(self.condition_seen.any(), self.rate.lerp(window_rate, beta), window_rate)
        self.rate.copy_(th.where(enough_data, next_rate, self.rate))

        group_rate = self.positive_count / self.sample_count.clamp_min(1).unsqueeze(-1)
        next_group_rate = th.where(
            self.condition_seen.unsqueeze(-1), self.condition_rate.lerp(group_rate, beta), group_rate
        )
        self.condition_rate.copy_(
            th.where(supported.unsqueeze(-1), next_group_rate, self.condition_rate)
        )
        self.condition_seen.logical_or_(supported)

        target = settings["target_rate"]
        # A supported condition vetoes adjustment if either the current window
        # or its EMA reaches the target. The current window catches recent changes.
        condition_active = (
            supported.unsqueeze(-1)
            & ((group_rate >= target) | (self.condition_rate >= target))
        ).flatten(0, 1).any(dim=0)
        low = enough_data & (self.rate < target) & (window_rate < target) & ~condition_active & ~self.protected
        self.low_windows.copy_(
            th.where(low, (self.low_windows + 1).clamp_max(settings["low_windows"]), 0)
        )
        eligible = self.low_windows >= settings["low_windows"]
        updated = (self.offset + settings["gain"] * eligible * (target - self.rate).clamp_min(0))
        updated.clamp_(max=settings["max_offset"])
        adjusted = updated > self.offset
        self.offset.copy_(updated)

        for name, value in (
            ("valid_rows", total),
            ("supported_conditions", supported.sum()),
            ("mean_activation_rate", window_rate.mean()),
            ("condition_veto_units", condition_active.logical_and(self.rate < target).sum()),
            ("eligible_units", eligible.sum()),
            ("adjusted_units", adjusted.sum()),
            ("capped_units", (self.offset >= settings["max_offset"]).sum()),
            ("max_offset", self.offset.max()),
        ):
            logger.log_stat("trhr_" + name, value.item(), t_env)
