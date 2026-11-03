import unittest
from pai_ros.fleet import Fleet, MODES
from pai_ros.fleet_world import FleetWorld

class DeliveryScenarioTest(unittest.TestCase):
    def test_receipt_and_return_precede_reassignment(self):
        for mode in MODES:
            f=Fleet(1,1,3,mode)
            for _ in range(30000):
                f.tick(.04)
                for j in f.jobs.values():
                    if j.phase=='observe': f.decide(j.name,j.color)
                    if j.phase in ('receiving','return_clear','return_home','dock'):
                        self.assertIsNotNone(f.agents[j.ground].task)
                if f.done: break
            self.assertTrue(f.done)
            self.assertTrue(all(j.receipt_confirmed and j.received_at for j in f.jobs.values()))
            for a in f.agents.values(): self.assertEqual(a.pos,f.home(a))
            for j in f.jobs.values():
                names=[e['event'] for e in f.events if e.get('package')==j.name]
                self.assertLess(names.index('receipt_confirmed'),names.index('delivered'))

    def test_rich_scene_and_manifest(self):
        f=Fleet(2,2,3)
        self.assertEqual({j.package_type for j in f.jobs.values()}, {'medical_case','grocery_crate','carton'})
        self.assertTrue(all(j.address and j.weight_kg>0 for j in f.jobs.values()))
        w=FleetWorld(f)
        try:
            self.assertGreater(w.model.ngeom,200)
            for a in f.agents.values(): self.assertIsNotNone(w.model.camera('camera_'+a.name))
            for j in f.jobs.values():
                body=w.model.body(j.name)
                self.assertGreater(body.geomnum[0],5)
        finally: w.close()
