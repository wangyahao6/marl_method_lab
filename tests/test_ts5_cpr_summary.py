"""Summary parsing and regular-evaluation AUC checks; no training or environments."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH=Path(__file__).resolve().parents[1]/'scripts/summarize_ts5_cpr_1m.py'
spec=importlib.util.spec_from_file_location('cpr_summary',PATH)
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)


class CPRSummaryTest(unittest.TestCase):
    def test_auc_uses_regular_block_and_explicit_last_value_extension(self):
        rows=[dict(phase=0,task='A',t_env=0,label='phase_start',wins=0,episodes=32),
              dict(phase=0,task='A',t_env=50,label='periodic',wins=16,episodes=32),
              dict(phase=0,task='A',t_env=100,label='stage_end',wins=128,episodes=128)]
        value,n=summary.auc(rows,dict(phase=0,task='A',start=0,t_env=100))
        self.assertEqual(value,37.5)
        self.assertEqual(n,2)

    def test_pending_export_has_required_headers_and_no_invented_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=root/'plan.json'
            plan.write_text(json.dumps(dict(tasks=[])))
            out=root/'output';summary.export(plan,out)
            required=('runs','evaluations','stage_metrics','learning_curves','paired_final',
                      'preference_diagnostics','cpr_reference_updates','gradient_diagnostics',
                      'memory_collection','sampling_cost','independent_final')
            for name in required:
                self.assertEqual(len((out/(name+'.csv')).read_text().splitlines()),1)
            self.assertFalse(json.loads((out/'execution_summary.json').read_text())['all_complete'])
            self.assertIn('等待或部分完成',(out/'README.md').read_text())


if __name__=='__main__':unittest.main()
