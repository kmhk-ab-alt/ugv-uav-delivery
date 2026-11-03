import math
import unittest
from pai_ros.fleet import Fleet
from pai_ros.airspace import plan_outbound, areas_for_jobs, SCENARIO_ID

class AirspaceTest(unittest.TestCase):
    def test_outbound_detours_and_no_segment_crosses_restriction(self):
        f=Fleet(1,1,3)
        for j in f.jobs.values():
            f.decide(j.name,'red') # queued commands cannot bypass normal observation
            j.destination=j.color
            start=[0,j.source[1],3.2]
            path=plan_outbound(start,f.goal(j),f.restricted_areas,j.source[1])
            self.assertTrue(any(abs(p[1]-j.source[1])>.6 for p in path))
            for a,b in zip(path,path[1:]):
                self.assertFalse(any(area.intersects(a,b,.22) for area in f.restricted_areas))
            self.assertAlmostEqual(math.dist(path[-1][:2],f.goal(j)[:2]),0.)

    def test_violation_and_clear_are_recorded(self):
        f=Fleet()
        u=f.agents['uav_0']
        u.pos[:]=[-1.5,0,3.2]
        f.monitor_airspace()
        self.assertEqual(f.warning_count,1)
        f.monitor_airspace()
        self.assertEqual(f.warning_count,1)
        u.pos[:]=[3,0,3.2]
        f.monitor_airspace()
        self.assertTrue(any(e['event']=='airspace_clear' for e in f.events))

    def test_proximity_warning(self):
        f=Fleet(1,2,1)
        f.agents['uav_1'].pos[:]=f.agents['uav_0'].pos
        f.monitor_airspace()
        self.assertTrue(any(e.get('kind')=='proximity' for e in f.events))

    def test_preflight_precedes_movement_and_relay_avoids_zone(self):
        f=Fleet(2,2,3)
        initial={a.name:a.pos.copy() for a in f.agents.values()}
        f.tick(.02)
        for a in f.agents.values(): self.assertEqual(a.pos,initial[a.name])
        for _ in range(10000):
            f.tick(.04)
            for j in f.jobs.values():
                if j.phase=='observe': f.decide(j.name,j.color)
            if f.done: break
        self.assertTrue(f.done)
        self.assertEqual(f.warning_count,0)
        self.assertEqual(f.snapshot()['scenario'],SCENARIO_ID)
        for j in f.jobs.values():
            events=[e['event'] for e in f.events if e.get('package')==j.name]
            self.assertLess(events.index('flight_plan'),events.index('simulated_flight_authorized'))
            self.assertIn('lower_package',events)
            self.assertEqual(j.phase,'delivered')

if __name__=='__main__': unittest.main()
