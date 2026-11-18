"""Load optimizer state from current or original single-group RNNAgent checkpoints."""


def load_optimizer_state(optimizer, state, agent):
    saved_groups = state['param_groups']
    current_groups = optimizer.param_groups
    if not getattr(agent, 'loaded_legacy_fc1', False):
        optimizer.load_state_dict(state)
        return
    if len(saved_groups) != len(current_groups) or len(current_groups) != 1:
        raise ValueError('legacy fc1 optimizer checkpoint requires one matching parameter group')

    saved_group = saved_groups[0]
    saved_ids = saved_group['params']
    current_params = current_groups[0]['params']
    if len(saved_ids) == len(current_params):
        # A checkpoint explicitly converted to the current parameter layout.
        ids = saved_ids
        converted = state
    else:
        agent_params = list(agent.parameters())
        if (len(saved_ids) != len(current_params) + 2
                or any(a is not b for a, b in zip(agent_params, current_params))
                or len(agent_params) > len(current_params)
                or saved_ids != list(range(len(saved_ids)))
                or not state['state'].keys() <= set(saved_ids)):
            raise ValueError('legacy fc1 optimizer parameter group size or order is incompatible')
        ids = saved_ids[2:]
        converted = dict(state, param_groups=[dict(saved_group, params=ids)],
                         state={key: value for key, value in state['state'].items() if key in ids})

    for key, parameter in zip(ids, current_params):
        for name, value in converted['state'].get(key, {}).items():
            if name != 'step' and hasattr(value, 'shape') and value.shape != parameter.shape:
                raise ValueError(f'legacy fc1 optimizer state {name} for parameter {key} has incompatible shape')
    optimizer.load_state_dict(converted)
