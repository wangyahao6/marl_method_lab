"""Small SMACv2 adapter: homogeneous mirror teams and exact phase budgets."""
import numpy as np
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper


def seed_distributions(obj, sequence):
    # Official SMACv2 uses unseeded default_rng() instances, including nested
    # position generators. Seed each instance without changing its distribution.
    if hasattr(obj, 'rng'):
        obj.rng = np.random.default_rng(sequence.spawn(1)[0])
    for value in vars(obj).values():
        if value is not obj and value.__class__.__module__.endswith('distributions'):
            seed_distributions(value, sequence)


class SequentialProtoss:
    def __init__(self, **kwargs):
        self.wrapped = StarCraftCapabilityEnvWrapper(**kwargs)
        seq = np.random.SeedSequence(kwargs['seed'])
        for distribution in self.wrapped.env_key_to_distribution_map.values():
            seed_distributions(distribution, seq)
        self.phase = 0
        self.remaining = None
        self.last_info = {}
        self.audit = None
        self.set_task('stalker')

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def set_task(self, unit):
        dist = self.wrapped.env_key_to_distribution_map['team_gen']
        if unit.startswith('p'):
            index=int(unit[1:])
            weights={'stalker':(10-index)/10., 'zealot':index/10., 'colossus':0.}
        else:
            weights={u:float(u==unit) for u in dist.units}
        dist.weights = np.array([weights[u] for u in dist.units])
        self.unit = unit

    def reset(self):
        result = self.wrapped.reset()
        raw = self.wrapped.env
        teams=raw.episode_config['team_gen']
        self.last_info = {}
        if self.audit:
            self.audit(dict(unit=self.unit, weights=self.wrapped.env_key_to_distribution_map['team_gen'].weights.tolist(), sampled_team=teams['ally_team'], ally_types=[u.unit_type for u in raw.agents.values()],
                            enemy_types=[u.unit_type for u in raw.enemies.values()],
                            obs_shape=self.get_obs_size(), state_shape=self.get_state_size(),
                            max_reward=float(raw.max_reward)))
        return result

    def step(self, actions):
        reward, terminated, info = self.wrapped.step(actions)
        if self.remaining is not None:
            self.remaining -= 1
            if self.remaining == 0 and not terminated:
                terminated = True
                info = dict(info, episode_limit=True, phase_budget_truncated=True)
                # reset() must restart this episode, including a truncated first
                # episode, rather than launch an additional SC2 process.
                self.wrapped.env._episode_count += 1
        self.last_info = info
        return reward, terminated, info
