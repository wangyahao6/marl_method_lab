"""Exact independent allied/enemy compositions, whole episodes, stable encoding."""
import copy
import numpy as np
from environment import seed_distributions
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

TEAMS={'A':(2,3),'R':(4,1),'M':(1,4)}
class ExactTeams:
    def __init__(self, rosters=None):
        self.rosters = rosters
        self.task = "A"

    def generate(self):
        if getattr(self, 'rosters', None) is not None:
            units = ('stalker', 'zealot', 'colossus')
            roster = self.rosters[self.task]
            teams = {
                f'{side}_team': [unit for unit, count in zip(units, roster[side]) for _ in range(count)]
                for side in ('ally', 'enemy')
            }
            return {'team_gen': dict(teams, id=0)}
        s,z=TEAMS[self.task]
        return {'team_gen':dict(ally_team=['stalker']*2+['zealot']*3,enemy_team=['stalker']*s+['zealot']*z,id=0)}
class ExactProtoss:
    def __init__(self,rosters=None,**kwargs):
        self.wrapped=StarCraftCapabilityEnvWrapper(**kwargs)
        sequence=np.random.SeedSequence(kwargs['seed'])
        for d in self.wrapped.env_key_to_distribution_map.values():seed_distributions(d,sequence)
        self.wrapped.env_key_to_distribution_map['team_gen']=ExactTeams(rosters)
        self.last_info={};self.audit=None;self.set_task('A')
    def __getattr__(self,name):return getattr(self.wrapped,name)
    def set_task(self,task):self.task=task;self.wrapped.env_key_to_distribution_map['team_gen'].task=task
    def distribution_state(self):return copy.deepcopy(self.wrapped.env_key_to_distribution_map)
    def restore_distribution(self,state):
        self.wrapped.env_key_to_distribution_map=copy.deepcopy(state);self.set_task(self.task)
    def reset(self):
        result=self.wrapped.reset();raw=self.wrapped.env
        # SMACv2 otherwise caches first-episode health/shield normalization.
        raw.max_reward=raw.n_enemies*raw.reward_death_value+raw.reward_win+sum(u.health_max+u.shield_max for u in raw.enemies.values())
        self.last_info={}
        if self.audit:self.audit(dict(task=self.task,requested=self.wrapped.env_key_to_distribution_map['team_gen'].generate()['team_gen'],ally_types=[int(u.unit_type) for u in raw.agents.values()],enemy_types=[int(u.unit_type) for u in raw.enemies.values()],max_reward=float(raw.max_reward),obs_shape=self.get_obs_size(),state_shape=self.get_state_size()))
        return result
    def step(self,actions):
        reward,done,info=self.wrapped.step(actions);self.last_info=info;return reward,done,info
