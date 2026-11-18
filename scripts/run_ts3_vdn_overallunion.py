"""Run the single overall-screening comparison from the completed shared prefix."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/ts3_vdn_A2_overallunion_3535_20261103'
RUN = 'ts3_vdn_overallunion_A2_union_seed3535_20261103_attempt001'
PARENT = ROOT / 'runs/ts3_vdn_typeunion_Aprefix_seed3535_20261102_attempt001'
COMMAND = [
    sys.executable, 'scripts/run.py', '--suite', 'ts3', '--algorithm', 'vdn',
    '--method', 'softprotect', '--seed', '3535', '--run-id', RUN,
    '--protocol', 'configs/continual/neuron_protection/softprotect/ts3_vdn_A2_overallunion_3535.json',
    '--ts3-role', 'branch', '--ts3-group', 'A2_union', '--parent', str(PARENT),
]

def main():
    if '--dry-run' in sys.argv:
        return subprocess.call(COMMAND + ['--dry-run'], cwd=ROOT)
    status = OUT / 'queue_status.json'
    status.write_text(json.dumps(dict(run=RUN, state='running', parent=str(PARENT))))
    code = subprocess.call(COMMAND, cwd=ROOT)
    summary = subprocess.call([
        sys.executable, 'scripts/summarize_ts3_vdn_typeunion.py', '--overall-run', RUN,
    ], cwd=ROOT)
    status.write_text(json.dumps(dict(run=RUN, state='finished_pending_review' if code==summary==0 else 'failed', exit_code=code, summary_exit_code=summary), indent=2))
    return code or summary

if __name__ == '__main__':
    sys.exit(main())
