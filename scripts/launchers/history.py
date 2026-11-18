"""Launch plan for the history suite."""
import sys
from .common import ROOT, LaunchSpec, load_protocol, prepare_smacv2


def build(a, p, out, env):
    if a.algorithm!='vdn' or a.method!='baseline':p.error('history requires VDN baseline')
    protocol=load_protocol(a.protocol)
    prepare_smacv2(env, out)
    command=[sys.executable,str(ROOT/'drivers/continual/history_train.py'),'--seed',str(a.seed)]
    if a.parent:command+=['--parent',a.parent]
    details={'protocol':protocol,'parent':a.parent,'steps':2000000}
    return LaunchSpec(command, env, details)
