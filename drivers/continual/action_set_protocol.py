"""Configuration for the standalone VDN action-set sequence."""
from math import isfinite

from common.exploration import epsilon_on_switch

TOPK_VARIANTS = ("topk_preference_weighted", "topk_preference_high_regret", "tkpr_refresh", "tkpr_k1_gap0", "tkpr_cpr")
FORK_ROLES = ("prefix", "keep", "release", "shared-prefix", "shared-branch", "pair-reference")

ALGORITHMS = ("vdn", "vdn_1ff", "vdn_1ff_128")
ASPR_MAX_METHODS = ("aspr_max_preference", "action_set_preference")
SELECTED_METHODS = ("shapley_selected_frpr", "random_selected_frpr")
RANK_METHODS = ("full_rank_preference", "full_rank_preference_relu", "full_rank_preference_unfiltered",
                "full_rank_preference_raw", "full_rank_preference_raw_loss", "top1_preference",
                "top1_max_preference", "top1_unfiltered_preference", "topk_preference", *TOPK_VARIANTS,
                "aspr_preference", "aspr_3g_preference", "aspr_3g_adjacent_preference", "aspr_4g_preference")
PAIR_STRUCTURES = {f"pair_structure_{selection}": selection for selection in (
    "k1", "adjacent", "k1_adjacent", "frpr", "random_n_minus_1", "random_2n_minus_3", "random_2n_minus_3_matched")}
PAIR_RATIOS = {f"pair_ratio_{percent}": percent for percent in (0, 20, 40, 60, 80, 100)}
PAIR_DISTANCE = {f"pair_distance_{selection}": selection for selection in (
    "k1", "near", "mid", "far", "far_mid", "random_matched", "maxgap")}
PAIR_GAP_FAR_MID = {f"pair_gap_far_mid_{selection}": f"far_mid_{selection}"
                    for selection in ("unfiltered", "filtered")}
PAIR_GAP_MATCHED = {"pair_gap_matched_non_top1": "gap_matched_non_top1"}
PAIR_TC = {f"pair_tc_{level}": level for level in ("low", "medium", "high")}
PAIR_ABLATIONS = {"tc_coverage": PAIR_TC, "pair_structure": PAIR_STRUCTURES, "pair_ratio": PAIR_RATIOS,
                  "pair_structure_gap05": PAIR_STRUCTURES, "pair_distance": PAIR_DISTANCE,
                  "pair_gap_far_mid": PAIR_GAP_FAR_MID, "pair_gap_matched": PAIR_GAP_MATCHED}
PAIR_METHODS = {**PAIR_STRUCTURES, **PAIR_RATIOS, **PAIR_DISTANCE, **PAIR_GAP_FAR_MID,
                **PAIR_GAP_MATCHED, **PAIR_TC}
PAIR_DEFAULTS = dict(weight=1.0, temperature=1.0, min_gap=0, q_scale_floor=0.1, pair_seed=3535)
PAIR_GAP05_DEFAULTS = dict(PAIR_DEFAULTS, min_gap=0.5)
PAIR_DISTANCE_DEFAULTS = dict(PAIR_DEFAULTS, extra_min_gap=0)
PAIR_EXTRA_GAP_DEFAULTS = dict(PAIR_DEFAULTS, extra_min_gap=0.5)
RANK_POSITION_METHODS = {"tkpr_r2": 2, "tkpr_r3": 3}
HISTORY_RANK_METHODS = (*RANK_METHODS, *SELECTED_METHODS, *RANK_POSITION_METHODS)
PREFERENCE_RANK_METHODS = (*HISTORY_RANK_METHODS, *PAIR_METHODS)
RANK_DEFAULTS = dict(weight=1.0, temperature=1.0, min_gap=0.5, q_scale_floor=0.1)
ASPR_GROUP_DEFAULTS = {"aspr_3g_adjacent_preference": dict(RANK_DEFAULTS, group_count=3, min_gap=0.0),
                       "aspr_3g_preference": dict(RANK_DEFAULTS, group_count=3),
                       "aspr_4g_preference": dict(RANK_DEFAULTS, group_count=4)}
FRPR_RAW_DEFAULTS = dict(weight=1.0, temperature=1.0, min_gap_raw=0.1)
FRPR_RAW_LOSS_DEFAULTS = dict(RANK_DEFAULTS)
TOPK_DEFAULTS = dict(RANK_DEFAULTS, k=2)
TOPK_WEIGHTED_DEFAULTS = dict(RANK_DEFAULTS, k=1)
CPR_DEFAULTS = dict(TOPK_WEIGHTED_DEFAULTS, reference_confidence_threshold=0.5,
                    reference_switch_margin=0.1, task_gate_tolerance=0.05)
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
    if protocol.get("optimizer", "rmsprop") not in ("rmsprop", "adam"):
        raise ValueError("optimizer must be rmsprop or adam")
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

    refresh_defaults = (dict(batch_episodes_per_task=8, refresh_success_episodes=50)
                        if method == "tkpr_refresh" else {})
    settings = settings_with_defaults(protocol.get("action_set", {}),
                                      dict(DEFAULTS, **refresh_defaults), ("delta_by_task",))
    for key in ("weight", "margin"):
        if not nonnegative(settings[key]):
            raise ValueError(f"action_set.{key} must be finite and nonnegative")
    for key in ("batch_episodes", "success_episodes", "attempts_per_seed"):
        if not positive_integer(settings[key]):
            raise ValueError(f"action_set.{key} must be a positive integer")
    if method == "tkpr_refresh":
        for key in refresh_defaults:
            if not positive_integer(settings.get(key)):
                raise ValueError(f"action_set.{key} must be a positive integer")
        if settings["refresh_success_episodes"] > settings["success_episodes"]:
            raise ValueError("refresh_success_episodes exceeds success_episodes")
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
    for name in HISTORY_RANK_METHODS:
        if (method == name or name in protocol
                or (name == "full_rank_preference" and method in SELECTED_METHODS)
                or (name == "topk_preference" and method in RANK_POSITION_METHODS)):
            defaults = {
                **ASPR_GROUP_DEFAULTS,
                "full_rank_preference_relu": RANK_RELU_DEFAULTS,
                "full_rank_preference_unfiltered": FRPR_UNFILTERED_DEFAULTS,
                "full_rank_preference_raw": FRPR_RAW_DEFAULTS,
                "full_rank_preference_raw_loss": FRPR_RAW_LOSS_DEFAULTS,
                "top1_max_preference": TOP1_MAX_DEFAULTS,
                "topk_preference": TOPK_DEFAULTS,
                "tkpr_refresh": TOPK_WEIGHTED_DEFAULTS,
                "tkpr_cpr": CPR_DEFAULTS,
                "tkpr_k1_gap0": dict(TOPK_WEIGHTED_DEFAULTS, min_gap=0.0),
                "topk_preference_weighted": TOPK_WEIGHTED_DEFAULTS,
                "topk_preference_high_regret": TOPK_HIGH_REGRET_DEFAULTS,
                "top1_unfiltered_preference": TOP1_UNFILTERED_DEFAULTS,
            }.get(name, RANK_DEFAULTS)
            if name in SELECTED_METHODS:
                defaults = ranking.get("full_rank_preference", RANK_DEFAULTS)
                options = settings_with_defaults(protocol.get(name, {}), defaults, (
                    "selection_k", "selection_ratio", "selection_seed", "shapley_results", "trace_pools"))
                config = {key: options.pop(key) for key in list(options) if key not in defaults}
                k, ratio = config.get("selection_k"), config.get("selection_ratio")
                sources, pools = config.get("shapley_results"), config.get("trace_pools", {})
                seed = config.get("selection_seed", 0)
                if (k is None) == (ratio is None) or (k is not None and not positive_integer(k)) or (
                        ratio is not None and (not nonnegative(ratio) or not 0 < ratio <= 1)):
                    raise ValueError(f"{name} needs exactly one positive selection_k or selection_ratio in (0, 1]")
                if (not isinstance(sources, list) or not sources
                        or any(not isinstance(path, str) or not path for path in sources)
                        or not valid_seeds([seed]) or not isinstance(pools, dict)
                        or set(pools) - {str(i) for i in range(len(stages))}
                        or any(not isinstance(path, str) or not path for path in pools.values())):
                    raise ValueError(f"Invalid {name} shapley_results, selection_seed, or trace_pools")
                if options != defaults:
                    raise ValueError(f"{name} inherits the FRPR loss settings from full_rank_preference")
                config["selection_seed"] = seed
                ranking[name] = dict(options, **config)
                continue
            if name in RANK_POSITION_METHODS:
                anchor_base = ranking["topk_preference"]
                if anchor_base["k"] != 1:
                    raise ValueError(f"{name} inherits settings from TKPR-K1")
                defaults = {key: value for key, value in anchor_base.items() if key != "k"}
                if settings_with_defaults(protocol.get(name, {}), defaults) != defaults:
                    raise ValueError(f"{name} must use the same loss settings as TKPR-K1")
            options = settings_with_defaults(protocol.get(name, {}), defaults)
            for key, value in options.items():
                if key == "k":
                    if not positive_integer(value):
                        raise ValueError(f"{name}.k must be a positive integer")
                    if name in TOPK_VARIANTS and value != 1:
                        raise ValueError(f"{name}.k must be 1")
                    continue
                if key == "group_count":
                    if value != ASPR_GROUP_DEFAULTS[name]["group_count"]:
                        raise ValueError(f"{name}.group_count must match its method")
                    continue
                positive = key in ("temperature", "q_scale_floor", "hr_temperature")
                if not nonnegative(value) or (positive and value == 0):
                    bound = "positive" if positive else "nonnegative"
                    raise ValueError(f"{name}.{key} must be finite and {bound}")
            ranking[name] = options
    rank_audit = protocol.get("rank_position_diagnostics", False)
    if type(rank_audit) is not bool or (rank_audit and
            (method not in ("topk_preference", *RANK_POSITION_METHODS) or ranking["topk_preference"]["k"] != 1)):
        raise ValueError("rank_position_diagnostics applies to TKPR-K1 and single-rank methods")
    experiment = protocol.get("pair_ablation_experiment")
    if experiment is not None or method in PAIR_METHODS:
        if not isinstance(experiment, str) or experiment not in PAIR_ABLATIONS:
            raise ValueError("pair ablation requires its own experiment protocol")
        methods = PAIR_ABLATIONS[experiment]
        if method not in ("full_rank_preference", *methods):
            raise ValueError("pair ablation requires its own experiment protocol and method")
        if experiment != "tc_coverage" and (algorithm != "vdn" or len(stages) != 3 or len({stage["task"] for stage in stages}) != 3):
            raise ValueError("pair ablation requires three distinct stages with shared VDN")
        for name in methods:
            if name != method and name not in protocol:
                continue
            if experiment == "pair_structure_gap05":
                expected = PAIR_GAP05_DEFAULTS
            elif experiment in ("pair_distance", "pair_gap_far_mid"):
                expected = (PAIR_EXTRA_GAP_DEFAULTS if name == "pair_gap_far_mid_filtered"
                            else PAIR_DISTANCE_DEFAULTS)
            else:
                expected = PAIR_DEFAULTS
            options = settings_with_defaults(protocol.get(name, {}), expected)
            if (any(not nonnegative(value) for value in options.values())
                    or any(options[key] != value for key, value in expected.items() if key != "pair_seed")
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
        if role == "shared-branch" and method not in ("baseline", *HISTORY_RANK_METHODS):
            raise ValueError("a shared branch supports the baseline and ranking methods")
        return
    if role:
        if method not in RANK_METHODS or not protocol[method]["weight"]:
            raise ValueError("preference forks require an enabled ranking method")
        stages = protocol["stages"]
        if stages[-1]["task"] not in {stage["task"] for stage in stages[:-1]}:
            raise ValueError("the final stage of a preference fork must revisit a trained task")


def matching_shared_protocol(parent, branch):
    def shared_settings(protocol):
        return {"optimizer": "rmsprop", **{key: value for key, value in protocol.items() if key not in ("method", "rank_position_diagnostics")}}

    previous, requested = shared_settings(parent), shared_settings(branch)
    # A new branch method can supply its own settings when the prefix had no such method.
    method = branch["method"]
    if method in SELECTED_METHODS and "full_rank_preference" not in previous:
        previous["full_rank_preference"] = RANK_DEFAULTS
    if method not in previous and method in HISTORY_RANK_METHODS and method != parent["method"]:
        previous[method] = requested[method]
    return previous == requested


def valid_seeds(seeds):
    return (isinstance(seeds, list) and bool(seeds)
            and all(type(seed) is int and 0 <= seed < 2**31 for seed in seeds)
            and len(set(seeds)) == len(seeds))


def matching_pair_reference(parent, branch):
    """Compare the TD, environment and history settings across the two ablation protocols."""
    ignored = {"id", "method", "pair_ablation_experiment", *PAIR_METHODS}
    return ({key: value for key, value in parent.items() if key not in ignored}
            == {key: value for key, value in branch.items() if key not in ignored})
