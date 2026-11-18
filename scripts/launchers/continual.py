"""Launch plan for the continual suite."""
import sys
from .common import ROOT, LaunchSpec, load_protocol, prepare_smacv2


def build(a, p, out, env):
    if a.algorithm!='qmix' or a.method not in ['baseline','softprotect']:p.error('continual driver currently supports QMIX baseline/softprotect only')
    if not a.scenario:p.error('continual requires --scenario')
    if a.map or a.steps is not None or a.set:p.error('continual settings belong in a named --protocol JSON')
    protocol = load_protocol(a.protocol)
    prepare_smacv2(env, out)
    command=[sys.executable,str(ROOT/'drivers/continual/train.py'),'--algorithm',a.algorithm,'--condition',a.method,'--scenario',a.scenario,'--seed',str(a.seed)]
    details={'protocol':protocol,'scenario':a.scenario,'steps':protocol['scenarios'][a.scenario][-1][0]}
    return LaunchSpec(command, env, details)
