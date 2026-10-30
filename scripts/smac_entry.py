"""Use the original training loop with an explicit complete config."""
import json, os, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from run import run
from utils.logging import get_logger
cfg=json.loads(Path(sys.argv[1]).read_text())
SETTINGS['CAPTURE_MODE']='sys'
ex=Experiment('marl_method_lab'); ex.logger=get_logger()
ex.captured_out_filter=apply_backspaces_and_linefeeds
ex.add_config(cfg)
ex.observers.append(FileStorageObserver.create(str(Path(cfg['local_results_path'])/'sacred')))
@ex.main
def train(_run,_config,_log):
    config=dict(_config)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(config['seed'])
    torch.set_num_threads(1)
    run(_run,config,_log)
ex.run()
