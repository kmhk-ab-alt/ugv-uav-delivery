import json
from pathlib import Path
import tempfile
import unittest
from pai_ros.fleet import Fleet
from pai_ros.motion import KinematicMotion
from pai_ros.recording import EpisodeRecorder

class ExtensionTest(unittest.TestCase):
    def test_seed_reproducible_and_observation_excludes_true_label(self):
        a=Fleet(package_count=9,seed=4,randomize=True)
        b=Fleet(package_count=9,seed=4,randomize=True)
        self.assertEqual([j.color for j in a.jobs.values()],[j.color for j in b.jobs.values()])
        for j in a.observation()['jobs'].values():
            for key in ('color','address','recipient','contents','package_type'):
                self.assertNotIn(key,j)

    def test_motion_injected(self):
        class Backend(KinematicMotion):
            name='test_backend'
            calls=0
            def move(self,*args):
                self.calls+=1
                return super().move(*args)
        backend=Backend()
        f=Fleet(motion=backend)
        for _ in range(80): f.tick(.02)
        self.assertGreater(backend.calls,0)
        self.assertEqual(f.snapshot()['control_backend'],'test_backend')

    def test_recording_aligns_before_action_and_after(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fleet()
            rec=EpisodeRecorder(tmp,dict(seed=0))
            for _ in range(80):
                before=f.observation();f.tick(.02);rec.step(f,before)
            rec.close(f)
            rows=[json.loads(s) for s in (rec.path/'steps.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),80)
            self.assertEqual(rows[0]['timestamp'],rows[0]['observation']['time'])
            self.assertAlmostEqual(rows[0]['next_timestamp']-rows[0]['timestamp'],.02)
            self.assertTrue(any(r['action'] for r in rows))
            self.assertEqual(json.loads((rec.path/'manifest.json').read_text())['dataset_format'],'pai_episode_v1 (not LeRobot)')
