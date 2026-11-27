"""Small deterministic tests for own-policy histories and performance-gated replacement."""
import copy
from pathlib import Path
import json
import unittest
from unittest.mock import Mock

import torch as th

from drivers.continual.memory.win_refresh import initialize_own_pool,update_history
from action_set_protocol import validate,validate_fork,matching_fixed_reference_protocol
from learners.ranking_preference import RankingPreference,topk_preference

ROOT=Path(__file__).resolve().parents[1]


def pool(task='A',phase=0,q=1.,count=100):
    return dict(task=task,phase=phase,reference_checkpoint=f'models/phase{phase}/agent.th',
                episodes=[dict(length=1,data=dict(old_q_values=th.full((2,5,11),q))) for _ in range(count)])


class WinRefreshTest(unittest.TestCase):
    def test_tie_and_lower_win_rate_keep_pool_and_never_collect(self):
        old=initialize_own_pool(pool(),0,.5);h={0:old};collect=Mock(side_effect=AssertionError('must not collect'))
        for rate in (.5,.4):
            rows=update_history(h,{'A':rate},1,collect)
            self.assertFalse(rows[0]['gate_passed']);self.assertIs(h[0],old)
        collect.assert_not_called()

    def test_improvement_replaces_every_old_episode_and_reference_q(self):
        old=initialize_own_pool(pool(),0,.5);h={0:old};new=pool(phase=2,q=9.,count=7)
        rows=update_history(h,{'A':.6},2,lambda _:new)
        self.assertEqual(len(h),1);self.assertEqual(rows[0]['replaced_count'],100)
        self.assertEqual(len(h[0]['episodes']),7) # All seven new; none of the old100 survive.
        self.assertEqual(h[0]['phase'],0);self.assertEqual(h[0]['collection_phase'],2)
        self.assertEqual(h[0]['reference_win_rate'],.6)
        self.assertTrue(all(ep['reference_phase']==2 for ep in h[0]['episodes']))
        self.assertTrue(all((ep['data']['old_q_values']==9).all() for ep in h[0]['episodes']))
        self.assertTrue(all((ep['data']['old_q_values']==1).all() for ep in old['episodes']))
        collect=Mock(side_effect=AssertionError('compare against newly accepted .6'))
        self.assertFalse(update_history(h,{'A':.6},3,collect)[0]['gate_passed'])

    def test_no_success_keeps_old_reference_and_new_task_initializes_own_pool(self):
        old=initialize_own_pool(pool(),0,.5);h={0:old}
        result=update_history(h,{'A':.8},1,lambda _:pool(phase=1,count=0))[0]
        self.assertTrue(result['gate_passed']);self.assertFalse(result['pool_changed']);self.assertIs(h[0],old)
        self.assertEqual(h[0]['reference_win_rate'],.5)
        result=update_history(h,{'B':.4},1,lambda task:pool(task=task,phase=1,q=4.,count=101))[0]
        self.assertEqual(len(h[1]['episodes']),100);self.assertEqual(h[1]['reference_win_rate'],.4)
        self.assertEqual(len(h),2)

    def test_own_policy_configs_restore_only_prefix_and_preserve_native_loss(self):
        source=json.loads((ROOT/'configs/continual/action_preference/ts5/native_k1_gap/gap05_seed2222.json').read_text())
        parent=validate(json.loads((ROOT/'configs/continual/action_preference/tc_coverage/ts5_vdn64_tc_1m.json').read_text()),'vdn_1ff','full_rank_preference')
        for method in ('tkpr_k1_per_task8','tkpr_win_refresh'):
            raw=json.loads((ROOT/f'configs/continual/action_preference/ts5/own_policy/{method}_seed2222.json').read_text())
            checked=validate(raw,'vdn_1ff',method);validate_fork(checked,'shared-branch','parent')
            self.assertTrue(checked['own_policy_history']);self.assertFalse(checked.get('fixed_reference_history',False))
            self.assertEqual(checked['action_set']['batch_episodes_per_task'],8)
            self.assertTrue(matching_fixed_reference_protocol(parent,checked))
            pref=RankingPreference(method,dict(checked[method],batch_episodes=8,batch_episodes_per_task=8),None,None,None,None)
            self.assertIs(pref.loss,topk_preference)
            self.assertFalse(pref.use_reference_valid_mask)


if __name__=='__main__':unittest.main()
