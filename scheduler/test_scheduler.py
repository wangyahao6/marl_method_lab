"""Regression checks use temporary databases and harmless local commands."""
import json
import sqlite3
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import core
import service


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'state.sqlite3'
        self.db = core.connect(self.path)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def task(self, name, **options):
        return dict(id=name, argv=['/usr/bin/true'], cwd=self.temp.name,
                    resources=dict(gpu_mib=0, ram_mib=0), **options)

    def test_dedup_and_cycle_atomicity(self):
        plan = dict(tasks=[self.task('a')])
        core.submit(self.db, plan)
        core.submit(self.db, plan)
        self.assertEqual(len(core.read_tasks(self.db)), 1)
        with self.assertRaises(ValueError):
            core.submit(self.db, dict(tasks=[self.task('a', priority=1)]))
        with self.assertRaises(ValueError):
            core.submit(self.db, dict(tasks=[self.task('b', dependencies=['c']),
                                             self.task('c', dependencies=['b'])]))
        self.assertEqual(len(core.read_tasks(self.db)), 1)

    def test_dependency_failure_isolated(self):
        core.submit(self.db, dict(tasks=[self.task('bad'), self.task('child', dependencies=['bad']),
                                         self.task('independent')]), {'bad':dict(state='failed')})
        with patch.object(service, 'resources', return_value=dict(ram_mib=90000, gpu_mib=9000, gpu_processes={})), patch.object(service, 'process_tree', return_value={}):
            service.tick(self.db, self.path, launch=False)
        self.assertEqual([r['state'] for r in core.read_tasks(self.db)], ['failed','blocked','pending'])

    def test_reservation_and_mutex(self):
        spec = self.task('pending')
        spec.update(queue='q',queue_limit=6,mutex='shared')
        active = [dict(id='a',spec=dict(spec,resources=dict(gpu_mib=1300,ram_mib=2000)))]
        measured = dict(gpu_mib=2000,ram_mib=90000,gpu_processes={})
        self.assertEqual(core.availability(spec, active, {}, measured, {}), 'mutex')
        spec = dict(spec, mutex=None, resources=dict(gpu_mib=1300,ram_mib=2000))
        self.assertEqual(core.availability(spec, active, {}, measured, {}), 'gpu_memory')

    def test_durable_worker_and_limit(self):
        core.submit(self.db, dict(tasks=[self.task('a'),self.task('b')]))
        with self.db:
            self.db.execute("UPDATE settings SET value='1' WHERE key='max_parallel'")
        measured = dict(gpu_mib=10000,ram_mib=90000,gpu_processes={})
        with patch.object(service, 'resources', return_value=measured), patch.object(service, 'process_tree', return_value={}):
            summary = service.tick(self.db, self.path)
            self.assertEqual(summary['active'],1)
        for _ in range(100):
            if core.read_tasks(self.db)[0]['state'] == 'completed':
                break
            time.sleep(.05)
        self.assertEqual(core.read_tasks(self.db)[0]['state'],'completed')
        self.assertEqual(core.read_tasks(self.db)[1]['state'],'pending')
        self.db.close()
        self.db = core.connect(self.path)
        core.reconcile(self.db)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0],1)

    def test_worker_authorized_launcher_and_recovery(self):
        run_dir = str(Path(self.temp.name)/'run')
        program = (
            "import sys,time;sys.path.insert(0,"+repr(str(Path(service.__file__).parent))+");"
            "from service import forward_run;"
            "from pathlib import Path;"
            "assert forward_run('gate',[],Path("+repr(self.temp.name)+"),Path("+repr(run_dir)+"),'evaluation') is False;"
            "time.sleep(.3)"
        )
        task = self.task('gate', run_dir=run_dir, type='training')
        task['argv'] = ['/usr/bin/python3','-c',program]
        core.submit(self.db,dict(tasks=[task]))
        with self.db:
            self.db.execute("UPDATE tasks SET state='starting'")
            self.db.execute("INSERT INTO attempts(id,task,created,log) VALUES ('auth','gate',?,?)",
                            (time.time(),str(Path(self.temp.name)/'worker.log')))
        worker = subprocess.Popen(['/usr/bin/python3', service.__file__, '--db',str(self.path),'worker','auth'])
        try:
            for _ in range(100):
                if core.read_tasks(self.db)[0]['state']=='running':
                    break
                time.sleep(.01)
            self.db.close()
            self.db = core.connect(self.path)
            core.reconcile(self.db)
            self.assertEqual(core.read_tasks(self.db)[0]['state'],'running')
            self.assertEqual(worker.wait(timeout=5),0)
            self.assertEqual(core.read_tasks(self.db)[0]['state'],'completed')
            self.assertEqual(self.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0],1)
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()

    def test_lost_start_is_not_relaunched(self):
        core.submit(self.db,dict(tasks=[self.task('a')]))
        with self.db:
            self.db.execute("UPDATE tasks SET state='starting'")
            self.db.execute("INSERT INTO attempts(id,task,created,log) VALUES ('lost','a',0,'/tmp/unused')")
        core.reconcile(self.db)
        self.assertEqual(core.read_tasks(self.db)[0]['state'],'unknown')


if __name__ == '__main__':
    unittest.main()
