"""Native K1 zero-pair semantics and complete-history attachment, without SC2 training."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch as th

from action_set_protocol import validate, validate_fork, matching_fixed_reference_protocol
from learners.ranking_preference import topk_preference
from drivers.continual.memory import History, PreferenceHistory
from drivers.continual.memory.fixed_reference import attach_reference_pool

ROOT=Path(__file__).resolve().parents[1]


class NativeK1FixedReferenceTests(unittest.TestCase):
    def test_native_loss_keeps_two_and_three_action_states_and_zero_pairs(self):
        q=th.tensor([[1.,0.,9.],[3.,2.,1.],[0.,0.,0.],[.12,.1,9.]])
        legal=th.tensor([[1,1,0],[1,1,1],[1,1,1],[1,1,0]],dtype=th.bool)
        decisions=th.ones(4,dtype=th.bool)
        settings=dict(k=1,weight=1.,temperature=1.,q_scale_floor=.1,min_gap=0)
        current=q.clone().requires_grad_()
        loss,count,stats=topk_preference(current,q,legal,decisions,settings)
        self.assertEqual(int(count),3)
        self.assertEqual(int(stats[2]),4) # 1 + 2 + 0 tied + 1 below .5 normalized gap.
        loss.backward();self.assertTrue(th.isfinite(current.grad).all())
        filtered,valid,filtered_stats=topk_preference(q,q,legal,decisions,dict(settings,min_gap=.5))
        self.assertEqual(int(valid),2);self.assertEqual(int(filtered_stats[2]),3)
        self.assertEqual(q.shape[0],4) # No state tensor removal.
        self.assertTrue(th.isfinite(filtered))

    def test_exact_pool_copy_activates_only_the_requested_past_phase(self):
        with TemporaryDirectory() as directory:
            parent=Path(directory)/'reference';out=Path(directory)/'branch'
            pools={i:dict(phase=i,task=str(i),episodes=[dict(length=3,data={'old_q_values':th.zeros(4,5,11)})]) for i in range(4)}
            for i,pool in pools.items():History.save(PreferenceHistory.phase_path(parent,i)/'success.th',pool)
            active={}
            first=attach_reference_pool(active,pools,parent,out,0)
            self.assertEqual(list(active),[0]);self.assertEqual(first['source_sha256'],first['local_sha256'])
            attach_reference_pool(active,pools,parent,out,1)
            self.assertEqual(list(active),[0,1]);self.assertEqual(len(active[0]['episodes']),1)
            self.assertTrue(th.equal(active[0]['episodes'][0]['data']['old_q_values'],th.zeros(4,5,11)))

    def test_fixed_reference_opt_in_retains_training_settings(self):
        parent=json.loads((ROOT/'configs/continual/action_preference/tc_coverage/ts5_vdn64_tc_1m.json').read_text())
        parent=validate(parent,'vdn_1ff','full_rank_preference')
        for gap in (0,.5):
            branch={k:v for k,v in parent.items() if k not in ('id','method','pair_ablation_experiment')}
            branch.update(id='native',fixed_reference_history=True,topk_preference=dict(k=1,weight=1.,temperature=1.,min_gap=gap,q_scale_floor=.1))
            branch=validate(branch,'vdn_1ff','topk_preference');validate_fork(branch,'shared-branch','reference')
            self.assertTrue(matching_fixed_reference_protocol(parent,branch))
            changed=copy.deepcopy(branch);changed['evaluation']['interval']=1
            self.assertFalse(matching_fixed_reference_protocol(parent,changed))
            with self.assertRaises(ValueError):validate_fork(branch,'shared-prefix',None)
