import json
import math
import unittest
from pai_ros.fleet import Fleet, MODES
from pai_ros.cargo_vlm import parse_decision, offline_decide

class FleetTest(unittest.TestCase):
    def run_mission(self, mode, ground, uav):
        fleet = Fleet(ground, uav, 7, mode)
        for step in range(30000):
            fleet.tick(.04)
            busy = [a.task for a in fleet.agents.values() if a.task]
            for name in set(busy):
                self.assertEqual(busy.count(name), 2)
            for job in fleet.jobs.values():
                if job.phase == 'observe':
                    # Deterministic fixture replaces network inference in this test only.
                    self.assertTrue(fleet.decide(job.name, job.color, 'test fixture'))
                    self.assertFalse(fleet.decide(job.name, 'red', 'duplicate'))
                self.assertTrue(all(math.isfinite(x) for x in job.pos))
            if fleet.done:
                break
        self.assertTrue(fleet.done, fleet.snapshot())
        self.assertTrue(all(a.uses > 0 for a in fleet.agents.values()))
        for job in fleet.jobs.values():
            self.assertEqual(job.pos, fleet.goal(job))
            events = [e['event'] for e in fleet.events if e.get('package') == job.name]
            self.assertEqual(events.count('delivered'), 1)
            if mode == 'relay':
                self.assertIn('handoff_complete', events)
                self.assertLess(events.index('handoff_complete'), events.index('delivered'))
            elif mode == 'scout':
                self.assertIn('ground_pickup', events)
                self.assertNotIn('handoff_complete', events)
            else:
                self.assertIn('joint_delivery', events)
        json.dumps(fleet.snapshot())

    def test_modes_and_unbalanced_fleets(self):
        for mode in MODES:
            for counts in ((1,1), (3,2), (1,4), (4,1)):
                with self.subTest(mode=mode, counts=counts):
                    self.run_mission(mode, *counts)

    def test_no_decision_means_no_delivery(self):
        fleet = Fleet()
        for _ in range(2000):
            fleet.tick(.02)
        self.assertEqual(fleet.jobs['package_0'].phase, 'observe')
        self.assertEqual(fleet.jobs['package_0'].pos, fleet.jobs['package_0'].source)
        self.assertFalse(fleet.decide('package_0', 'unknown'))
        self.assertFalse(fleet.decide('package_1', 'red'))
        self.assertFalse(fleet.decide('missing', 'red'))
        self.assertFalse(fleet.done)

    def test_configuration_validation(self):
        for args in ((0,1,1,'relay'), (1,-1,1,'relay'), (1,1,0,'relay'), (1,1,1,'bad')):
            with self.assertRaises(ValueError): Fleet(*args)

    def test_vlm_response_validation(self):
        self.assertEqual(parse_decision('```json\n{"destination":"blue","reason":"blue parcel"}\n```')[0], 'blue')
        for raw in ('red', '{"destination":"unknown"}', '{"destination":"smash"}'):
            with self.assertRaises(ValueError): parse_decision(raw)

if __name__ == '__main__': unittest.main()
