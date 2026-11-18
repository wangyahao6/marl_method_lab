"""Launch plan for the hetero suite."""
import sys
from .common import ROOT, LaunchSpec, load_protocol, prepare_smacv2


def build(a, p, out, env):
    if a.algorithm!='qmix' or not a.hetero_role:p.error('hetero requires qmix and --hetero-role')
    if a.hetero_role=='branch' and not a.parent:p.error('branch requires --parent')
    protocol = load_protocol(a.protocol)
    if a.suite=='ts3' and a.algorithm!=protocol.get('algorithm','qmix'):p.error('algorithm must match named ts3 protocol')
    prepare_smacv2(env, out)
    command=[sys.executable,str(ROOT/'drivers/continual/hetero_train.py'),'--role',a.hetero_role,'--seed',str(a.seed),'--task',a.branch_task,'--group',a.group]
    if a.parent:command+=['--parent',a.parent]
    details={'protocol':protocol,'parent':a.parent,'branch_task':a.branch_task,'group':a.group,'steps':1000000 if a.hetero_role=='prefix' else 1500000}
    return LaunchSpec(command, env, details)
