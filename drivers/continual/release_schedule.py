"""Latest-selection protection with finite, continuous release in environment steps."""


class ReleaseSchedule:
    def __init__(self, widths, protected_scale=0.1, recovery_steps=100000):
        assert 0 < protected_scale < 1 and recovery_steps > 0
        self.widths = dict(widths)
        self.scale = protected_scale
        self.duration = recovery_steps
        self.protected = {label: set() for label in widths}
        self.recovering = {label: {} for label in widths}

    def values(self, t_env):
        result = {}
        for label, width in self.widths.items():
            values = [1.0]*width
            for j, event in list(self.recovering[label].items()):
                elapsed = max(0, t_env-event['start'])
                fraction = min(1.0, elapsed/self.duration)
                values[j] = event['scale']+(1-event['scale'])*fraction
                if fraction == 1:
                    del self.recovering[label][j]
            for j in self.protected[label]:
                values[j] = self.scale
            result[label] = values
        return result

    def replace(self, selection, t_env):
        old_values = self.values(t_env)
        for label, width in self.widths.items():
            chosen = set(selection[label])
            assert all(0 <= j < width for j in chosen)
            # Only newly exiting units start a ramp; existing ramps keep their clock.
            for j in self.protected[label]-chosen:
                self.recovering[label][j] = dict(start=t_env, scale=old_values[label][j])
            for j in chosen:
                self.recovering[label].pop(j, None)
            self.protected[label] = chosen
        return self.values(t_env)

    def state(self, t_env):
        values = self.values(t_env)
        return dict(t_env=t_env, protected={k: sorted(v) for k,v in self.protected.items()},
                    recovering={k: {str(j): dict(event) for j,event in v.items()}
                                for k,v in self.recovering.items()},
                    unit_scales=values, recovery_steps=self.duration)
