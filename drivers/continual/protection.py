"""Boundary-only W/V screening and RMSprop parameter-increment scaling."""
from collections import deque
import copy
import hashlib
import random
from types import SimpleNamespace
import numpy as np
import torch as th
from detection_common import sensitivity, summary
from release_schedule import ReleaseSchedule
from neuron_records import detection_record, write_detection


def digest(value):
    h = hashlib.sha256()
    def visit(v):
        if th.is_tensor(v):
            a = v.detach().cpu().contiguous().numpy()
            h.update(str((a.shape, a.dtype)).encode()); h.update(a.tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str((v.shape, v.dtype)).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (list, tuple)):
            for x in v: visit(x)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def rng_state():
    return (random.getstate(), np.random.get_state(), th.get_rng_state(),
            th.cuda.get_rng_state_all() if th.cuda.is_available() else [])


def training_fingerprint(learner):
    return digest([n.state_dict() for n in (learner.mac.agent, learner.mixer,
                                           learner.target_mac.agent, learner.target_mixer)]
                  + [learner.optimiser.state_dict(), rng_state()])


def cpu_episode(batch):
    length = int(batch.max_t_filled().item())
    data = SimpleNamespace(
        transition_data={k: v[:, :length].detach().cpu().clone()
                         for k, v in batch.data.transition_data.items()},
        episode_data={k: v.detach().cpu().clone() for k, v in batch.data.episode_data.items()})
    from components.episode_buffer import EpisodeBatch
    return EpisodeBatch(batch.scheme, batch.groups, 1, length, data=data, device='cpu')


def combine(episodes):
    from components.episode_buffer import EpisodeBatch
    length = max(x.max_seq_length for x in episodes)
    first = episodes[0]
    transition = {}
    for k, v in first.data.transition_data.items():
        out = th.zeros((len(episodes), length, *v.shape[2:]), dtype=v.dtype)
        for i, ep in enumerate(episodes):
            out[i, :ep.max_seq_length] = ep[k][0]
        transition[k] = out
    data = SimpleNamespace(transition_data=transition,
        episode_data={k: th.cat([ep[k] for ep in episodes]) for k in first.data.episode_data})
    return EpisodeBatch(first.scheme, first.groups, len(episodes), length, data=data, device='cpu')


@th.no_grad()
def screen(mac, mixer, batch, config, enumerate_check=False):
    """One frozen online model, complete histories; no target activations."""
    old_hidden = mac.hidden_states
    try:
        mac.init_hidden(batch.batch_size)
        qs = th.stack([mac.forward(batch, t=t) for t in range(batch.max_seq_length)], 1)
        mask = batch['filled'][:, :-1].float().clone()
        mask[:, 1:] *= 1-batch['terminated'][:, :-2].float()
        # Each shorter padded episode has its own bootstrap-only final state.
        # True terminations suppress it via terminated; time-limit truncations do not.
        lengths = batch['filled'].squeeze(-1).sum(1)-1
        positions = th.arange(batch.max_seq_length-1, device=mask.device)[None, :]
        good = mask.squeeze(-1).bool() & (positions < lengths[:, None])
        states = batch['state'][:, :-1][good]
        if not len(states):
            return dict(valid_states=0, layers={k: dict(selected=None, reason='no_valid_states')
                                              for k in ['W', 'V']})
        c, verification = sensitivity(mixer, qs[:, :-1][good],
            batch['actions'][:, :-1].squeeze(-1)[good],
            batch['avail_actions'][:, :-1][good], states, enumerate_check)
        mean_c = float(c.mean())
        omega = c/mean_c if mean_c > 0 else np.zeros_like(c)
        sample_count = len(c)
        total = float(omega.sum())
        result = dict(valid_states=len(c), sensitivity=summary(c), weights=summary(omega),
                      weight_sum=total, mean_sensitivity=mean_c, activity_denominator=sample_count,
                      weight_formula='c/mean(c)', verification=verification, layers={})
        for label, network in [('W', mixer.hyper_w_final), ('V', mixer.V)]:
            z = network[:3](states).abs().cpu().numpy().astype(np.float64)
            num = (z*omega[:, None]).sum(0)
            a = num/sample_count if mean_c > 0 else None
            avg = float(a.mean()) if a is not None else 0
            score = a/avg if avg > 0 else None
            ids = None
            if score is not None:
                ids = sorted(range(len(a)), key=lambda j: (-score[j], j))
                ids = [j for j in ids if score[j] > config['threshold_beta']]
            result['layers'][label] = dict(numerator=num.tolist(), denominator=sample_count,
                activity=a.tolist() if a is not None else None,
                score=score.tolist() if score is not None else None, selected=ids,
                reason=None if ids is not None else 'zero_weight_or_zero_layer')
        return result
    finally:
        mac.hidden_states = old_hidden


class SoftProtection:
    def __init__(self, learner, config, output, append, write):
        self.learner, self.config, self.output = learner, config, output
        self.append, self.write = append, write
        self.fifo = deque(maxlen=config['episodes'])
        self.keys = {}
        self.latest = {label: [] for label in config['layers']}
        self.schedule = ReleaseSchedule({label:64 for label in config['layers']},
                                        config['step_scale'], config['recovery_env_steps'])
        self.current = None
        self.phase = -1
        self.t_env = 0
        self.steps = 0
        self.audit_next = True
        self.params = []
        for label, network in [('W', learner.mixer.hyper_w_final), ('V', learner.mixer.V)]:
            for name, param, axis in [('input_weight', network[0].weight, 0),
                                      ('input_bias', network[0].bias, 0),
                                      ('output_weight', network[3].weight, 1)]:
                self.params.append((label, name, param, axis))
        self.masks = [th.zeros_like(p, dtype=th.bool) for _, _, p, _ in self.params]
        self.protected = {'W': [], 'V': []}
        self.original_step = learner.optimiser.step
        learner.optimiser.step = self.step

    def state(self):
        return dict(current_task=self.current, phase=self.phase, steps=self.steps,
                    task_sets=copy.deepcopy(self.keys), protected=copy.deepcopy(self.protected),
                    step_scale=self.config['step_scale'], threshold_beta=self.config['threshold_beta'],
                    latest_selection=copy.deepcopy(self.latest), release_schedule=self.schedule.state(self.t_env))

    def enter(self, task, phase, t_env):
        previous = self.protected
        self.current, self.phase, self.t_env = task, phase, t_env
        self.fifo.clear()
        self.schedule.replace(self.latest, t_env)
        self.protected = {label: sorted(ids) for label,ids in self.schedule.protected.items()}
        self.masks = []
        for label, name, param, axis in self.params:
            mask = th.zeros_like(param, dtype=th.bool)
            ids = self.protected[label]
            if axis == 0: mask[ids] = True
            else: mask[:, ids] = True
            self.masks.append(mask)
        self.audit_next = True
        self.append(self.output/'protection_transitions.jsonl', dict(t_env=t_env,
            previous=previous, released={l: sorted(set(previous[l])-set(self.protected[l]))
                                        for l in ['W', 'V']}, **self.state()))

    def remember(self, batch, episode, task, start, end, info):
        self.fifo.append((cpu_episode(batch), dict(episode=episode, task=task,
            start_step=start, end_step=end, budget_truncated=bool(info.get('phase_budget_truncated', False)))))

    @th.no_grad()
    def step(self, *args, **kwargs):
        audit = self.audit_next or self.steps % 1000 == 0
        # Save only six small parameter tensors; all unselected entries are left exact.
        before = [p.detach().clone() for _, _, p, _ in self.params]
        result = self.original_step(*args, **kwargs)
        records = []
        unit_scales = self.schedule.values(self.t_env)
        for (label, name, param, axis), old in zip(self.params, before):
            vector = th.tensor(unit_scales[label], dtype=param.dtype, device=param.device)
            factors = vector if param.ndim==1 else (vector[:,None] if axis==0 else vector[None,:])
            factors = factors.expand_as(param)
            mask = factors < 1
            raw = param-old
            if mask.any():
                param[mask] = old[mask]+factors[mask]*raw[mask]
            if audit:
                actual = param-old
                records.append(dict(layer=label, parameter=name, selected_elements=int(mask.sum()),
                    factor_min=float(factors.min()), factor_max=float(factors.max()),
                    recovering_units=len(self.schedule.recovering[label]),
                    protected_units=len(self.schedule.protected[label]),
                    raw_protected_norm=float(raw[mask].norm()), actual_protected_norm=float(actual[mask].norm()),
                    raw_unprotected_norm=float(raw[~mask].norm()), actual_unprotected_norm=float(actual[~mask].norm()),
                    unprotected_exact=bool(th.equal(actual[~mask], raw[~mask]))))
        if audit:
            self.append(self.output/'update_audit.jsonl', dict(t_env=self.t_env, phase=self.phase,
                task=self.current, optimizer_step=self.steps, step_scale=self.config['step_scale'], threshold_beta=self.config['threshold_beta'], parameters=records, unit_scales=unit_scales))
        self.audit_next = False
        self.steps += 1
        return result

    def detect(self, task, t_env, args, enumerate_check=False):
        if not self.fifo:
            self.write(self.output / 'detection_empty_fifo.json', dict(task=task,t_env=t_env,reason='no diagnostic episodes'))
            return
        before = training_fingerprint(self.learner)
        batch = combine([b for b, _ in self.fifo])
        metadata = [m for _, m in self.fifo]
        dest = self.output/'detections'/str(t_env)
        dest.mkdir(parents=True, exist_ok=False)
        # Save CPU data before moving it for analysis; this is a reproducible small package.
        th.save(dict(batch=batch, args=vars(args), config=self.config,
                     agent={k: v.detach().cpu().clone() for k,v in self.learner.mac.agent.state_dict().items()},
                     mixer={k: v.detach().cpu().clone() for k,v in self.learner.mixer.state_dict().items()}), dest/'input.th')
        batch.to(args.device)
        result = screen(self.learner.mac, self.learner.mixer, batch, self.config, enumerate_check)
        expected_states = sum(m['end_step']-m['start_step'] for m in metadata)
        after = training_fingerprint(self.learner)
        previous_selection = copy.deepcopy(self.latest)
        before_scales = self.schedule.values(t_env)
        record = self.keys.setdefault(task, {})
        for label, layer in result['layers'].items():
            if layer['selected'] is not None:
                record[label] = layer['selected']
                self.latest[label] = list(layer['selected'])
        overlaps = {}
        for other, keys in self.keys.items():
            if other == task: continue
            overlaps[other] = {}
            for label in ['W', 'V']:
                a, b = set(record.get(label, [])), set(keys.get(label, []))
                overlaps[other][label] = dict(intersection=len(a & b), union=len(a | b),
                                             jaccard=len(a & b)/len(a | b) if a | b else None)
        result = dict(task=task, phase=self.phase, t_env=t_env,
            episodes=metadata, fingerprint_before=before, fingerprint_after=after,
            detection_unchanged=(before == after), mask_matches_decisions=(result['valid_states'] == expected_states),
            fifo_task_matches=all(m['task']==task for m in metadata), overlaps=overlaps, **result)
        event, rows = detection_record(result, previous_selection, self.latest, before_scales,
            args, self.output, self.config, dest)
        result['selection_event'] = event
        self.write(dest/'result.json', result)
        write_detection(self.output, event, rows)
        self.write(dest/'task_sets.json', self.state())


class NullProtection:
    def __init__(self): self.t_env=0
    def state(self): return dict(enabled=False)
    def enter(self,*args): pass
    def remember(self,*args): pass
    def detect(self,*args,**kwargs): pass
