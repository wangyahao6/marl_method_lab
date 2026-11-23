import unittest
import numpy as np
import torch as th
from learners.pair_tc import ranked_graphs, quotas, select_tc_pairs


class TCSelectionTests(unittest.TestCase):
    def test_closure_and_controls(self):
        rng = np.random.default_rng(37)
        for n in range(4, 12):
            q = -np.cumsum(rng.uniform(.1, 3, (9, n)), axis=-1)
            a, b = np.triu_indices(n, 1)
            reach_counts = []
            for level in ('low', 'medium', 'high'):
                edges, reached = ranked_graphs(q, level)
                self.assertTrue(edges[:, a == 0].all())
                self.assertTrue((edges.sum(-1) == 2*n-3).all())
                bucket = np.minimum(b-a-1, 2)
                for k, quota in enumerate(quotas(n)):
                    self.assertTrue((edges[:, (a > 0) & (bucket == k)].sum(-1) == quota).all())
                for row in range(len(q)):
                    reach = np.zeros((n,n), dtype=bool)
                    reach[a,b] = edges[row]
                    for k in range(n):
                        reach |= reach[:, k:k+1] & reach[k:k+1, :]
                    self.assertEqual(reach.sum(), reached[row])
                reach_counts.append(reached)
            if n >= 6:
                self.assertGreater(reach_counts[-1].mean(), reach_counts[0].mean())

    def test_action_ids_and_invalid_rows(self):
        q = th.tensor([[[3., 9., 4., 1., 7., 2.], [1., 1., 2., 3., 4., 5.], [1.,2.,3.,4.,5.,6.]]])
        legal = th.ones_like(q, dtype=th.bool)
        decision = th.tensor([[True, True, False]])
        mask, valid, counts, audit = select_tc_pairs(q, legal, decision, 'high', 1)
        self.assertEqual(valid.tolist(), [[True, False, False]])
        self.assertFalse(mask[0,1:].any())
        self.assertEqual(mask.sum(), 9)
        left,right = th.triu_indices(6,6,1)
        self.assertTrue(mask[0,0,(left==1)|(right==1)].all())
        self.assertEqual(int(counts[...,2].sum()),5)


if __name__ == '__main__':
    unittest.main()

class FrozenPoolAuditTests(unittest.TestCase):
    def test_offline_auditor_and_frozen_loss(self):
        import sys
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'drivers/continual')]
        from offline_tc_coverage import audit_episode
        from common.runtime import episode_spec
        from components.episode_buffer import EpisodeBatch,ReplayBuffer
        from drivers.continual.memory.episodes import pack_episode
        from learners.pair_tc import TCPPreference
        spec,groups,pre=episode_spec(dict(state_shape=3,obs_shape=4,n_actions=11,n_agents=5))
        scheme=ReplayBuffer(spec,groups,1,2,preprocess=pre).scheme
        batch=EpisodeBatch(spec,groups,1,5,preprocess=pre)
        batch.update(dict(state=th.zeros(1,5,3),obs=th.zeros(1,5,5,4),actions=th.zeros(1,5,5,1,dtype=th.long),
                          avail_actions=th.ones(1,5,5,11,dtype=th.int),reward=th.zeros(1,5,1),terminated=th.zeros(1,5,1,dtype=th.uint8)))
        episode=pack_episode(batch,4,seed=2222,index=0,won=True)
        episode['data']['old_q_values']=th.arange(11.).expand(5,5,11).clone()
        pool=dict(task='A',phase=0,episodes=[episode])
        settings=dict(weight=1.,temperature=1.,min_gap=0,q_scale_floor=.1,pair_seed=2222,batch_episodes=8)
        for level in ('low','medium','high'):
            pref=TCPPreference(level,settings,np.random.default_rng(2222),scheme,groups,lambda x:None)
            prepared,audit=pref.prepare_pool(pool,prior_budget=0)
            rows,*_=audit_episode(episode,prepared[0],scheme,groups)
            self.assertEqual(sum(r['closure_pairs'] for r in rows),audit['closure_pairs'])
            with TemporaryDirectory() as directory:
                path=Path(directory)/level;path.mkdir()
                th.save(dict(masks=[ep['pair_mask'] for ep in prepared],counts=[ep['pair_counts'] for ep in prepared],audit=audit),path/'phase0000.th')
                pref.mask_root=Path(directory)
                loaded,loaded_audit=pref.prepare_pool(pool,prior_budget=0)
                self.assertTrue(th.equal(loaded[0]['pair_mask'],prepared[0]['pair_mask']))
                self.assertEqual(loaded_audit,audit)
