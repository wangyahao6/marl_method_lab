"""Old-policy value loss of the action picked by a different policy."""


def action_regret(old_q, new_actions, old_best_value, old_scale):
    raw = old_best_value - old_q.gather(-1, new_actions.unsqueeze(-1)).squeeze(-1)
    return raw, raw / old_scale
