"""Exploration modes shared by the launcher and continual drivers."""

DEFAULT_EPSILON_ON_SWITCH = "per_task"


def epsilon_on_switch(protocol):
    mode = protocol.get("epsilon_on_switch", DEFAULT_EPSILON_ON_SWITCH)
    if mode not in (DEFAULT_EPSILON_ON_SWITCH, "cumulative_no_restart"):
        raise ValueError("epsilon_on_switch must be cumulative_no_restart or per_task")
    return mode
