"""Attach complete, immutable reference histories without any decision/pair filtering."""
import hashlib
from pathlib import Path
import shutil

from drivers.continual.memory import PreferenceHistory


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
    return digest.hexdigest()


def attach_reference_pool(history, reference_history, parent, out, phase):
    """Activate one past-task pool and retain every episode and all its original states."""
    source=PreferenceHistory.phase_path(parent,phase)/'success.th'
    target=PreferenceHistory.phase_path(out,phase)/'success.th'
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(str(source),str(target))
    history[phase]=reference_history[phase]
    pool=history[phase]
    return dict(source_phase=phase,task=pool['task'],episodes=len(pool['episodes']),
                source_pool=str(source),local_pool=str(target),source_sha256=file_sha256(source),
                local_sha256=file_sha256(target),state_filter=False,pair_selection='native topk_preference k=1')
