"""Configuration for the standalone VDN action-set sequence."""
from math import isfinite

ALGORITHMS = ("vdn", "vdn_1ff", "vdn_1ff_128")
DEFAULTS = dict(weight=1.0, margin=0.0, batch_episodes=8, success_episodes=100,
                seed_offsets=[301001, 302001, 303001, 304001], attempts_per_seed=125,
                delta_candidates_by_task={})
EVAL_DEFAULTS = dict(seeds=[91101, 91102], episodes_per_seed=16, interval=25000)


def nonnegative(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


def positive_integer(value):
    return type(value) is int and value > 0


def settings_with_defaults(values, defaults, required=()):
    if not isinstance(values, dict) or values.keys() - (defaults.keys() | set(required)):
        raise ValueError("unknown or malformed action-set protocol settings")
    return {**defaults, **values}


def validate(protocol, algorithm, method):
    if algorithm not in ALGORITHMS or method not in ("baseline", "action_set_preference"):
        raise ValueError("vdn_action_set requires shared recurrent VDN and baseline or action_set_preference")
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    tasks = protocol.get("tasks")
    stages = protocol.get("stages")
    if not isinstance(tasks, dict) or not tasks or not isinstance(stages, list) or not stages:
        raise ValueError("protocol requires named Protoss tasks and a nonempty stage sequence")
    for task, roster in tasks.items():
        if not task or not isinstance(roster, dict) or set(roster) != {"ally", "enemy"}:
            raise ValueError(f"task {task!r} needs ally and enemy rosters")
        for side, counts in roster.items():
            if (not isinstance(counts, list) or len(counts) != 3
                    or any(type(n) is not int or n < 0 for n in counts) or sum(counts) != 5):
                raise ValueError(f"task {task!r} {side} must specify five units as [stalker, zealot, colossus]; "
                                 "changing team size changes the shared Agent inputs and/or action indices")
    for stage in stages:
        if (not isinstance(stage, dict) or set(stage) != {"task", "steps"}
                or not isinstance(stage["task"], str) or stage["task"] not in tasks
                or not positive_integer(stage["steps"])):
            raise ValueError("each stage needs a configured task and a positive integer step budget")

    evaluation = settings_with_defaults(protocol.get("evaluation", {}), EVAL_DEFAULTS)
    if (not valid_seeds(evaluation["seeds"]) or not positive_integer(evaluation["episodes_per_seed"])
            or type(evaluation["interval"]) is not int or evaluation["interval"] < 0):
        raise ValueError("evaluation needs unique seeds, a positive episode count, and a nonnegative interval")

    settings = settings_with_defaults(protocol.get("action_set", {}), DEFAULTS, ("delta_by_task",))
    for key in ("weight", "margin"):
        if not nonnegative(settings[key]):
            raise ValueError(f"action_set.{key} must be finite and nonnegative")
    for key in ("batch_episodes", "success_episodes", "attempts_per_seed"):
        if not positive_integer(settings[key]):
            raise ValueError(f"action_set.{key} must be a positive integer")
    if not valid_seeds(settings["seed_offsets"]) or len(settings["seed_offsets"]) < 2:
        raise ValueError("action_set.seed_offsets requires at least two unique seeds")
    deltas = settings.get("delta_by_task", {})
    if (not isinstance(deltas, dict) or deltas.keys() - tasks.keys()
            or any(not nonnegative(delta) for delta in deltas.values())):
        raise ValueError("action_set.delta_by_task requires finite nonnegative thresholds keyed by task")
    if method == "action_set_preference" and settings["weight"]:
        missing = {stage["task"] for stage in stages} - deltas.keys()
        if missing:
            raise ValueError(f"action_set.delta_by_task is missing {sorted(missing)}")
    candidates = settings["delta_candidates_by_task"]
    if (not isinstance(candidates, dict) or candidates.keys() - tasks.keys()
            or any(not isinstance(values, list) or any(not nonnegative(v) for v in values)
                   for values in candidates.values())):
        raise ValueError("action_set.delta_candidates_by_task requires lists of nonnegative thresholds keyed by task")
    return dict(protocol, algorithm=algorithm, method=method, evaluation=evaluation, action_set=settings)


def valid_seeds(seeds):
    return (isinstance(seeds, list) and bool(seeds)
            and all(type(seed) is int and 0 <= seed < 2**31 for seed in seeds)
            and len(set(seeds)) == len(seeds))
