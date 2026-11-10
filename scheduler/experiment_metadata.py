"""Conversation and branch identity carried by submission records."""
GROUP_FIELDS = ('experiment_group_id', 'experiment_group_title', 'conversation_title', 'conversation_url')
BRANCH_FIELDS = ('branch_id', 'branch_title')


def fields(metadata):
    if not isinstance(metadata, dict):
        raise ValueError('metadata must be an object')
    result = {}
    for key in GROUP_FIELDS + BRANCH_FIELDS:
        if key in metadata:
            value = metadata[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(key + ' must be a nonempty string')
            result[key] = value.strip()
    if 'attempt' in metadata:
        value = metadata['attempt']
        if type(value) is not int or value < 1:
            raise ValueError('attempt must be a positive integer')
        result['attempt'] = value
    return result


def inherit(item, parents):
    metadata = item.get('metadata', {})
    explicit = fields(metadata)
    source = metadata.get('inherit_from')
    if source is not None:
        if source not in parents:
            raise ValueError('inherit_from must name a dependency')
        parents = {source: parents[source]}
    elif item['type'] not in ('evaluation', 'summary'):
        return
    values = list(parents.values())
    if not values:
        return
    group = values[0].get('experiment_group_id')
    if not group or any(parent.get('experiment_group_id') != group for parent in values):
        return
    if explicit.get('experiment_group_id', group) != group:
        return
    inherited = {}
    branch = values[0].get('branch_id')
    same_branch = branch and all(parent.get('branch_id') == branch for parent in values)
    keys = GROUP_FIELDS
    if same_branch and explicit.get('branch_id', branch) == branch:
        keys += BRANCH_FIELDS
    for key in keys:
        for parent in values:
            if parent.get(key):
                inherited[key] = parent[key]
                break
    item['metadata'] = dict(inherited, **metadata)
