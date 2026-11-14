"""CPU rollout block shared by the roster and history evaluators."""
import copy
import logging

import torch as th


def fixed_seed_plan(task_index, focus, *, stride=100):
    base = 91101 if focus else 91001
    blocks = [base + stride * task_index + index for index in range(4 if focus else 2)]
    return blocks, 32 if focus else 16


def evaluate_block(args, spec, controller_scheme, agent_state, task, seed, count, *,
                   make_controller, make_runner, make_logger, record=None, include_return=False):
    ea = copy.deepcopy(args)
    ea.device = 'cpu'
    ea.use_cuda = False
    ea.env_args['seed'] = seed
    _, groups, _ = spec
    mac = make_controller(controller_scheme, groups, ea)
    mac.agent.load_state_dict(agent_state)
    mac.agent.eval()
    runner = make_runner(ea, make_logger(logging.getLogger('eval')))
    rows = []
    try:
        runner.env.set_task(task)
        runner.setup(*spec, mac)
        with th.no_grad():
            for index in range(count):
                runner.run(test_mode=True)
                row = dict(env_seed=seed, episode=index,
                           won=bool(runner.env.last_info.get('battle_won', False)), length=runner.t)
                if include_return:
                    row['return_value'] = float(runner.batch['reward'].sum())
                if record is not None:
                    record(row)
                rows.append(row)
    finally:
        runner.close_env()
    return rows
