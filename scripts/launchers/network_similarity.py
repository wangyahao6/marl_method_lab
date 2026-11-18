"""Launch plan for the network_similarity suite."""
import sys
from .common import ROOT, LaunchSpec, load_protocol


def build(a, p, out, env):
    if a.algorithm!='vdn' or a.method!='baseline':p.error('network similarity requires frozen VDN baseline models')
    protocol=load_protocol(a.protocol)
    env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    command=[sys.executable,str(ROOT/'scripts/network_similarity.py'),'--protocol',str(out/'protocol.json'),'--output',str(out)]
    details={'protocol':protocol,'steps':0,'mode':'read-only offline analysis of saved models'}
    return LaunchSpec(command, env, details)
