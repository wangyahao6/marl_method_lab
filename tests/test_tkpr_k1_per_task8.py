"""Budget-only checks: no environment and no optimizer updates."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch as th

from action_set_protocol import validate,validate_fork,matching_fixed_reference_protocol,TOPK_WEIGHTED_DEFAULTS
from learners.ranking_preference import RankingPreference,topk_preference

ROOT=Path(__file__).resolve().parents[1]


class PerTask8Test(unittest.TestCase):
    def preference(self,method):
        settings=dict(TOPK_WEIGHTED_DEFAULTS,batch_episodes=8)
        if method=='tkpr_k1_per_task8':settings['batch_episodes_per_task']=8
        return RankingPreference(method,settings,np.random.default_rng(3535),None,None,None)

    def test_each_history_gets_eight_and_original_budget_stays_eight(self):
        for n in range(1,5):
            for method in ('topk_preference','tkpr_k1_per_task8'):
                p=self.preference(method);p.task='current'
                for i in range(n):p.history[i]=dict(phase=i,task=str(i),episodes=[{} for _ in range(100)])
                p.history[n]=dict(phase=n,task='current',episodes=[{}]*100)
                p.history[n+1]=dict(phase=n+1,task='empty',episodes=[])
                before=copy.deepcopy(p.history)
                for _ in range(10):
                    samples=p.sample()
                    self.assertEqual(sum(len(ids) for _,ids in samples),8*n if method=='tkpr_k1_per_task8' else 8)
                    if method=='tkpr_k1_per_task8':self.assertTrue(all(len(ids)==8 for _,ids in samples))
                    self.assertTrue(all(len(set(ids))==len(ids) for _,ids in samples))
                self.assertEqual(before,p.history)
                self.assertIs(p.loss,topk_preference)

    def test_small_pool_uses_replacement_and_task_reduction_stays_equal(self):
        p=self.preference('tkpr_k1_per_task8');p.task='B'
        p.history[0]=dict(phase=0,task='A',episodes=[{}]*3)
        samples=p.sample();self.assertEqual(len(samples[0][1]),8)
        self.assertLessEqual(len(set(samples[0][1])),3)
        loss=p.reduce_task_losses([th.tensor(1.),th.tensor(3.)],th.tensor([1,100]))
        self.assertEqual(loss.item(),2.)

    def test_native_configs_change_only_method_and_budget(self):
        for seed in (2222,3535):
            source=json.loads((ROOT/f'configs/continual/action_preference/ts5/native_k1_gap/gap05_seed{seed}.json').read_text())
            target=json.loads((ROOT/f'configs/continual/action_preference/ts5/tkpr_k1_per_task8/seed{seed}.json').read_text())
            effective=validate(target,'vdn_1ff','tkpr_k1_per_task8')
            validate_fork(effective,'shared-branch','reference')
            parent=json.loads((ROOT/'configs/continual/action_preference/tc_coverage/ts5_vdn64_tc_1m.json').read_text())
            parent=validate(parent,'vdn_1ff','full_rank_preference')
            self.assertTrue(matching_fixed_reference_protocol(parent,effective))
            self.assertNotIn('refresh_success_episodes',effective['action_set'])
            source.pop('id');target.pop('id')
            self.assertEqual(source.pop('topk_preference'),target.pop('tkpr_k1_per_task8'))
            self.assertEqual(target['action_set'].pop('batch_episodes_per_task'),8)
            self.assertEqual(source.pop("gradient_monitor_interval"),0)
            self.assertEqual(target.pop("gradient_monitor_interval"),100)
            self.assertEqual(source,target)
            invalid=copy.deepcopy(effective);invalid['action_set']['batch_episodes_per_task']=4
            with self.assertRaisesRegex(ValueError,'batch_episodes_per_task=8'):
                validate(invalid,'vdn_1ff','tkpr_k1_per_task8')


if __name__=='__main__':unittest.main()
