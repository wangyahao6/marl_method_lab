"""Public handle ownership for temporary and learner activation observers."""


class HookHandles:
    def __init__(self):
        self.handles = []

    def add(self, handle):
        self.handles.append(handle)
        return handle

    def remove(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def install_activation_observers(learner, registrations):
    """Retain the original online/target instrumentation, with owned handles.

    The target MAC has already been copied, so its observers get their own
    removable handles; third-party hooks on either model are untouched.
    """
    handles = HookHandles()
    targets = dict(learner.target_mac.agent.named_modules())
    agent_names = {id(module): name for name, module in learner.mac.agent.named_modules()}
    try:
        for module, callback in registrations:
            handles.add(module.register_forward_hook(callback))
            if id(module) in agent_names:
                handles.add(targets[agent_names[id(module)]].register_forward_hook(callback))
        return handles
    except BaseException:
        handles.remove()
        raise
