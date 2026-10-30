"""Same-batch state-weighted activity, without modifying training or RNG."""
import hashlib
import time
import numpy as np
import torch as th

LAYER_PATHS = dict(A='agent.net.2', A2='agent.net.4',
                   W='mixer.hyper_w_final.2', V='mixer.V.2')


def parameter_hash(mac, mixer):
    h = hashlib.sha256()
    for prefix, network in [('agent', mac.agent), ('mixer', mixer)]:
        for key, value in network.state_dict().items():
            h.update((prefix+'.'+key).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def batch_hash(batch):
    h = hashlib.sha256()
    for group in [batch.data.transition_data, batch.data.episode_data]:
        for key in sorted(group):
            value = group[key].detach().cpu().contiguous().numpy()
            h.update(key.encode())
            h.update(str(value.shape).encode())
            h.update(value.tobytes())
    return h.hexdigest()


class Capture:
    def __init__(self, mac, mixer):
        self.active = False
        self.handles = []
        modules = dict(A=mac.agent.net[2], A2=mac.agent.net[4],
                       W=mixer.hyper_w_final[2], V=mixer.V[2])
        for name, module in modules.items():
            def observe(module, inputs, output, name=name):
                if self.active:
                    self.values[name].append(output.detach())
            self.handles.append(module.register_forward_hook(observe))
        def agent_output(module, inputs, output):
            if self.active:
                self.values['q'].append(output[0].detach())
        self.handles.append(mac.agent.register_forward_hook(agent_output))

    def begin(self):
        self.values = {key: [] for key in ['A', 'A2', 'W', 'V', 'q']}
        self.active = True

    def tensors(self, batch, n_agents):
        self.active = False
        b, t = batch.batch_size, batch.max_seq_length
        result = {}
        for name in ['A', 'A2', 'q']:
            result[name] = th.stack([x.reshape(b, n_agents, -1)
                                    for x in self.values[name]], dim=1)
        for name in ['W', 'V']:
            if len(self.values[name]) != 1:
                raise ValueError('QMIX online mixer must be captured once per check')
            result[name] = self.values[name][0].reshape(b, t-1, -1)
        self.values = {}
        return result


@th.no_grad()
def sensitivity(mixer, q, actions, avail, states, enumerate_check=False):
    """Exact monotone endpoints, caching state-dependent mixer parameters."""
    results, verification = [], []
    for start in range(0, len(q), 2048):
        qq, aa, vv, ss = [x[start:start+2048] for x in (q, actions, avail, states)]
        n, agents, _ = qq.shape
        chosen = qq.gather(2, aa.unsqueeze(-1)).squeeze(-1)
        w1 = mixer.hyper_w_1(ss).abs().reshape(n, agents, mixer.embed_dim)
        b1 = mixer.hyper_b_1(ss)
        wf = mixer.hyper_w_final(ss).abs()
        value = mixer.V(ss).squeeze(-1)
        pre = th.bmm(chosen.unsqueeze(1), w1).squeeze(1) + b1
        base = (th.nn.functional.elu(pre)*wf).sum(-1)+value
        allowed = vv.bool().clone()
        allowed.scatter_(2, aa.unsqueeze(-1), False)
        lows = qq.masked_fill(~allowed, float('inf')).min(-1).values
        highs = qq.masked_fill(~allowed, -float('inf')).max(-1).values
        possible = allowed.any(-1)
        lows = th.where(possible, lows, chosen)
        highs = th.where(possible, highs, chosen)
        differences = []
        for agent in range(agents):
            ends = []
            for end in [lows, highs]:
                z = pre+(end[:,agent]-chosen[:,agent]).unsqueeze(-1)*w1[:,agent]
                qt = (th.nn.functional.elu(z)*wf).sum(-1)+value
                ends.append((qt-base).abs())
            differences.append(th.stack(ends).max(0).values * possible[:,agent])
        c = th.stack(differences, dim=1).mean(1)
        if enumerate_check and start == 0:
            k = min(32, n)
            full = []
            baseline = mixer(chosen[:k].unsqueeze(1), ss[:k].unsqueeze(1)).reshape(-1)
            for agent in range(agents):
                diffs = []
                for action in range(qq.shape[-1]):
                    alternative = chosen[:k].clone()
                    alternative[:,agent] = qq[:k,agent,action]
                    y = mixer(alternative.unsqueeze(1), ss[:k].unsqueeze(1)).reshape(-1)
                    diffs.append((y-baseline).abs()*allowed[:k,agent,action])
                full.append(th.stack(diffs).max(0).values)
            exact = th.stack(full,dim=1).mean(1)
            error = float((c[:k]-exact).abs().max().item())
            match = bool(th.allclose(c[:k], exact, rtol=1e-4, atol=1e-5))
            verification.append(dict(states=k, endpoint_vs_enumeration=match,
                                     max_abs_error=error))
            if not match:
                raise RuntimeError('Cached monotone endpoints disagree with full action enumeration')
        results.append(c)
    return th.cat(results).cpu().numpy().astype(np.float64), verification


def ratio(a, b):
    return a/b if b else None


def overlap(d, c, units):
    shared, union = d & c, d | c
    return dict(dormant=len(d), weighted=len(c), intersection=len(shared),
        union=len(union), neurons=units, jaccard=ratio(len(shared),len(union)),
        retained_dormant=ratio(len(shared),len(d)), dormant_in_weighted=ratio(len(shared),len(c)),
        retained_ids=sorted(shared), exited_ids=sorted(d-c), entered_ids=sorted(c-d))


def summary(values):
    return dict(n=len(values), quantiles=np.quantile(values,[0,.05,.25,.5,.75,.95,1]).tolist()) if len(values) else dict(n=0,quantiles=None)

