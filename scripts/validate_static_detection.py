"""Offline regression: reproduce live non-leaf state, compare detection isolation."""
import copy,json,logging,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'drivers/continual')]
import numpy as np
import torch as th
from controllers.basic_controller import BasicMAC
from learners.q_learner import QLearner
from static_detection import Detector
from protection import cpu_episode,digest,rng_state,screen

class Logger:
    console_logger=logging.getLogger('validation')
    def log_stat(self,*args):pass

def main():
 th.set_num_threads(1)
 p=th.load(ROOT/'runs/qmix_ABA_softprotect_seed1111_20261031/detections/1000000/input.th',map_location='cpu')
 a=SimpleNamespace(**p['args']);a.device='cpu';a.use_cuda=False
 batch=p['batch'];batch.to('cpu')
 mac=BasicMAC(batch.scheme,batch.groups,a);learner=QLearner(mac,batch.scheme,Logger(),a)
 mac.agent.load_state_dict(p['agent']);learner.mixer.load_state_dict(p['mixer'])
 # One offline optimization on saved data, no environment or training-run budget.
 learner.train(batch,100,0)
 assert mac.hidden_states.grad_fn is not None
 failed=False
 try:copy.deepcopy(mac)
 except RuntimeError as e:failed='deepcopy' in str(e)
 assert failed
 def fullstate():
  return digest([learner.mac.agent.state_dict(),learner.mixer.state_dict(),learner.target_mac.agent.state_dict(),learner.target_mixer.state_dict(),learner.optimiser.state_dict(),mac.hidden_states,learner.neu_data,learner.neu_avg,learner.mask,rng_state()])
 before=fullstate();old_hidden=mac.hidden_states
 with tempfile.TemporaryDirectory(prefix='qmix_detection_regression_') as dest:
  a.local_results_path=dest;d=Detector(learner,a)
  for i in range(batch.batch_size):d.fifo.append((cpu_episode(batch[i:i+1]),['stalker','stalker','zealot','zealot','zealot']))
  d.detect(100000,100000)
  assert not (d.out/'errors.jsonl').exists()
  assert before==fullstate() and mac.hidden_states is old_hidden
  r=json.loads((d.out/'100000/result.json').read_text())
  assert all(set(v)=={'overall','type_stalker','type_zealot',*[f'agent_{i}' for i in range(5)]} for v in r['layers'].values())
  for layer,groups in r['layers'].items():
   overall=np.array(groups['overall']['activity'])
   individual=np.array([groups[f'agent_{i}']['activity'] for i in range(5)])
   assert np.allclose(overall,individual.mean(0),rtol=1e-12,atol=1e-12)
   assert np.allclose(groups['type_stalker']['activity'],individual[:2].mean(0))
   assert np.allclose(groups['type_zealot']['activity'],individual[2:].mean(0))
  pack=th.load(d.out/'100000/input.th',map_location='cpu');assert pack['batch'].batch_size==32
  fresh=BasicMAC(batch.scheme,batch.groups,a);fresh.agent.load_state_dict(pack['agent'])
  from modules.mixers.qmix import QMixer
  mix=QMixer(a);mix.load_state_dict(pack['mixer'])
  ref=screen(fresh,mix,batch,dict(layers=['A'],threshold_beta=3))
  assert np.allclose(r['layers']['A1']['overall']['score'],ref['layers']['A']['score'],rtol=1e-6,atol=1e-6)
  # Second call exercises temporal stability and does not mutate live training state.
  before=fullstate();d.detect(200000,200000);assert before==fullstate()
  r2=json.loads((d.out/'200000/result.json').read_text())
  assert all(g['temporal_overlap'] is not None for layer in r2['layers'].values() for g in layer.values())
  audit=[json.loads(s) for s in (d.out/'audit.jsonl').read_text().splitlines()]
  assert all(x['training_state_unchanged'] and x['rng_restored'] for x in audit)
 result=dict(old_failure_reproduced=True,non_leaf_hidden_state=True,actual_learner_hooks=True,groups_checked=True,existing_A1_score_match=True,trajectory_saved_before_compute=True,training_and_auxiliary_state_unchanged=True,rng_restored=True,temporal_stability_checked=True,environment_interactions=0,synthetic_type_labels='group arithmetic only, not scientific evidence')
 out=ROOT/'results/repair_validation_2s3z_20261101';out.mkdir(exist_ok=True);(out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
