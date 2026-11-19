"""Configuration for the standalone VDN action-set sequence."""
from math import isfinite

from common.exploration import epsilon_on_switch

TOPK_VARIANTS = ("topk_preference_weighted", "topk_preference_high_regret")
FORK_ROLES = ("prefix", "keep", "release", "shared-prefix", "shared-branch", "pair-reference")

ALGORITHMS = ("vdn", "vdn_1ff", "vdn_1ff_128")
ASPR_MAX_METHODS = ("aspr_max_preference", "action_set_preference")
RANK_METHODS = ("full_rank_preference", "full_rank_preference_relu", "full_rank_preference_unfiltered",
                "full_rank_preference_raw", "full_rank_preference_raw_loss", "top1_preference",
                "top1_max_preference", "top1_unfiltered_preference", "topk_preference", *TOPK_VARIANTS,
                "aspr_preference")
PAIR_STRUCTURES = {f"pair_structure_{selection}": selection for selection in (
    "k1", "adjacent", "k1_adjacent", "frpr", "random_n_minus_1", "random_2n_minus_3")}
PAIR_RATIOS = {f"pair_ratio_{percent}": percent for percent in (0, 20, 40, 60, 80, 100)}
PAIR_ABLATIONS = {"pair_structure": PAIR_STRUCTURES, "pair_ratio": PAIR_RATIOS}
PAIR_METHODS = {**PAIR_STRUCTURES, **PAIR_RATIOS}
PAIR_DEFAULTS = dict(weight=1.0, temperature=1.0, min_gap=0, q_scale_floor=0.1, pair_seed=3535)
PREFERENCE_RANK_METHODS = (*RANK_METHODS, *PAIR_METHODS)
RANK_DEFAULTS = dict(weight=1.0, temperature=1.0, min_gap=0.5, q_scale_floor=0.1)
FRPR_RAW_DEFAULTS = dict(weight=1.0, temperature=1.0, min_gap_raw=0.1)
FRPR_RAW_LOSS_DEFAULTS = dict(RANK_DEFAULTS)
TOPK_DEFAULTS = dict(RANK_DEFAULTS, k=2)
TOPK_WEIGHTED_DEFAULTS = dict(RANK_DEFAULTS, k=1)
TOPK_HIGH_REGRET_DEFAULTS = dict(TOPK_WEIGHTED_DEFAULTS, hr_threshold=0.5,
                               hr_weight=1.0, hr_temperature=1.0)
RANK_RELU_DEFAULTS = {key: value for key, value in RANK_DEFAULTS.items() if key != "temperature"}
TOP1_MAX_DEFAULTS = dict(RANK_RELU_DEFAULTS, margin=0.0)
FRPR_UNFILTERED_DEFAULTS = {key: value for key, value in RANK_DEFAULTS.items() if key != "min_gap"}
TOP1_UNFILTERED_DEFAULTS = {key: value for key, value in RANK_DEFAULTS.items() if key != "min_gap"}
DEFAULTS = dict(weight=1.0, margin=0.0, batch_episodes=8, success_episodes=100,
                seed_offsets=[301001, 302001, 303001, 304001], attempts_per_seed=125,
                delta_candidates_by_task={}, second_gap_quantile=None, probe_episodes_per_seed=0)
EVAL_DEFAULTS = dict(seeds=[91101, 91102], episodes_per_seed=16, interval=25000,
                     focus_seeds=None, focus_episodes_per_seed=None, task_seed_stride=0)


def nonnegative(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


def positive_integer(value):
    return type(value) is int and value > 0


def settings_with_defaults(values, defaults, required=()):
    if not isinstance(values, dict) or values.keys() - (defaults.keys() | set(required)):
        raise ValueError("unknown or malformed action-set protocol settings")
    return {**defaults, **values}


def validate(protocol, algorithm, method):
    if algorithm not in ALGORITHMS or method not in ("baseline", *ASPR_MAX_METHODS, *PREFERENCE_RANK_METHODS):
        raise ValueError("vdn_action_set requires shared recurrent VDN and a supported preference method or baseline")
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    replay = protocol.get("replay_on_switch", "ft")
    if replay != "ft" and (replay != "er" or method != "baseline"):
        raise ValueError("vdn_action_set uses FT replay, with ER supported for the baseline")
    interval = protocol.get("gradient_monitor_interval", 0)
    if type(interval) is not int or interval < 0:
        raise ValueError("gradient_monitor_interval must be a nonnegative integer")
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

    if protocol.get("phase_boundary", "relative") not in ("relative", "cumulative"):
        raise ValueError("phase_boundary must be relative or cumulative")
    exploration = epsilon_on_switch(protocol)

    evaluation = settings_with_defaults(protocol.get("evaluation", {}), EVAL_DEFAULTS)
    if (not valid_seeds(evaluation["seeds"]) or not positive_integer(evaluation["episodes_per_seed"])
            or type(evaluation["interval"]) is not int or evaluation["interval"] < 0):
        raise ValueError("evaluation needs unique seeds, a positive episode count, and a nonnegative interval")

    if (evaluation["focus_seeds"] is None) != (evaluation["focus_episodes_per_seed"] is None):
        raise ValueError("focus_seeds and focus_episodes_per_seed must be set together")
    if evaluation["focus_seeds"] is not None and (not valid_seeds(evaluation["focus_seeds"])
            or not positive_integer(evaluation["focus_episodes_per_seed"])):
        raise ValueError("invalid focus evaluation seeds or episode count")
    if type(evaluation["task_seed_stride"]) is not int or evaluation["task_seed_stride"] < 0:
        raise ValueError("task_seed_stride must be a nonnegative integer")

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
    quantile = settings["second_gap_quantile"]
    if quantile is not None and (not nonnegative(quantile) or quantile > 1):
        raise ValueError("action_set.second_gap_quantile must be between 0 and 1")
    probes = settings["probe_episodes_per_seed"]
    if type(probes) is not int or probes < 0:
        raise ValueError("action_set.probe_episodes_per_seed must be a nonnegative integer")
    settings["delta_by_task"] = deltas
    candidates = settings["delta_candidates_by_task"]
    if (not isinstance(candidates, dict) or candidates.keys() - tasks.keys()
            or any(not isinstance(values, list) or any(not nonnegative(v) for v in values)
                   for values in candidates.values())):
        raise ValueError("action_set.delta_candidates_by_task requires lists of nonnegative thresholds keyed by task")
    ranking = {}
    for name in RANK_METHODS:
        if method == name or name in protocol:
            defaults = {
                "full_rank_preference_relu": RANK_RELU_DEFAULTS,
                "full_rank_preference_unfiltered": FRPR_UNFILTERED_DEFAULTS,
                "full_rank_preference_raw": FRPR_RAW_DEFAULTS,
                "full_rank_preference_raw_loss": FRPR_RAW_LOSS_DEFAULTS,
                "top1_max_preference": TOP1_MAX_DEFAULTS,
                "topk_preference": TOPK_DEFAULTS,
                "topk_preference_weighted": TOPK_WEIGHTED_DEFAULTS,
                "topk_preference_high_regret": TOPK_HIGH_REGRET_DEFAULTS,
                "top1_unfiltered_preference": TOP1_UNFILTERED_DEFAULTS,
            }.get(name, RANK_DEFAULTS)
            options = settings_with_defaults(protocol.get(name, {}), defaults)
            for key, value in options.items():
                if key == "k":
                    if not positive_integer(value):
                        raise ValueError(f"{name}.k must be a positive integer")
                    if name in TOPK_VARIANTS and value != 1:
                        raise ValueError(f"{name}.k must be 1")
                    continue
                positive = key in ("temperature", "q_scale_floor", "hr_temperature")
                if not nonnegative(value) or (positive and value == 0):
                    bound = "positive" if positive else "nonnegative"
                    raise ValueError(f"{name}.{key} must be finite and {bound}")
            ranking[name] = options
    experiment = protocol.get("pair_ablation_experiment")
    if experiment is not None or method in PAIR_METHODS:
        if not isinstance(experiment, str) or experiment not in PAIR_ABLATIONS:
            raise ValueError("pair ablation requires its own experiment protocol")
        methods = PAIR_ABLATIONS[experiment]
        if method not in ("full_rank_preference", *methods):
            raise ValueError("pair ablation requires its own experiment protocol and method")
        if algorithm != "vdn" or len(stages) != 3 or len({stage["task"] for stage in stages}) != 3:
            raise ValueError("pair ablation requires three distinct stages with shared VDN")
        for name in methods:
            if name != method and name not in protocol:
                continue
            options = settings_with_defaults(protocol.get(name, {}), PAIR_DEFAULTS)
            if (any(not nonnegative(value) for value in options.values())
                    or any(options[key] != value for key, value in PAIR_DEFAULTS.items() if key != "pair_seed")
                    or not valid_seeds([options["pair_seed"]])):
                raise ValueError(f"{name} requires the fixed ablation settings and a valid pair_seed")
            ranking[name] = options
    uses_sets = ((method in ASPR_MAX_METHODS and settings["weight"])
                 or (method == "aspr_preference" and ranking[method]["weight"]))
    if uses_sets and quantile is None:
        missing = {stage["task"] for stage in stages} - deltas.keys()
        if missing:
            raise ValueError(f"action_set.delta_by_task is missing {sorted(missing)}")
    return dict(protocol, algorithm=algorithm, method=method, evaluation=evaluation,
                action_set=settings, epsilon_on_switch=exploration, **ranking)


def validate_fork(protocol, role, parent):
    if bool(parent) != (role in ("keep", "release", "shared-branch")):
        raise ValueError("keep, release and shared-branch require --parent; other roles take no parent")
    method = protocol["method"]
    if role == "pair-reference":
        if method != "full_rank_preference" or not protocol.get("pair_ablation_experiment") or not protocol[method]["weight"]:
            raise ValueError("pair-reference requires enabled FRPR on a pair ablation protocol")
        return
    if method in PAIR_METHODS:
        if role != "shared-branch":
            raise ValueError("pair ablation branches require --preference-role shared-branch and the reference run as --parent")
        return
    if protocol.get("pair_ablation_experiment"):
        raise ValueError("pair ablation FRPR is used with --preference-role pair-reference")
    if role in ("shared-prefix", "shared-branch"):
        if protocol.get("replay_on_switch", "ft") != "ft" or len(protocol["stages"]) < 2:
            raise ValueError("a shared S prefix requires FT and at least two stages")
        if role == "shared-prefix" and (method not in RANK_METHODS or not protocol[method]["weight"]):
            raise ValueError("a shared prefix needs an enabled ranking method to collect old Q")
        if role == "shared-branch" and method not in ("baseline", *RANK_METHODS):
            raise ValueError("a shared branch supports the baseline and ranking methods")
        return
    if role:
        if method not in RANK_METHODS or not protocol[method]["weight"]:
            raise ValueError("preference forks require an enabled ranking method")
        stages = protocol["stages"]
        if stages[-1]["task"] not in {stage["task"] for stage in stages[:-1]}:
            raise ValueError("the final stage of a preference fork must revisit a trained task")


def shared_fork_protocol(protocol):
    # All method configurations and common settings match across the shared prefix and branches.
    return {key: value for key, value in protocol.items() if key != "method"}


def valid_seeds(seeds):
    return (isinstance(seeds, list) and bool(seeds)
            and all(type(seed) is int and 0 <= seed < 2**31 for seed in seeds)
            and len(set(seeds)) == len(seeds))


def matching_pair_reference(parent, branch):
    """Compare the TD, environment and history settings across the two ablation protocols."""
    ignored = {"id", "method", "pair_ablation_experiment", *PAIR_METHODS}
    return ({key: value for key, value in parent.items() if key not in ignored}
            == {key: value for key, value in branch.items() if key not in ignored})
