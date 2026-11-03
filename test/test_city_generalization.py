import unittest
import xml.etree.ElementTree as ET
import numpy as np
from pai_ros.city import generate_city,LocalObstacleMap
from pai_ros.fleet import Fleet
from pai_ros.fleet_dispatch import FleetDispatch
from pai_ros.generalization import run,digest
from pai_ros.dispatch_model import ScoringPolicy
from pai_ros.gazebo_scene import build_world

class CityTests(unittest.TestCase):
    def test_geometry_is_reproducible_and_disjoint(self):
        a=generate_city(100).manifest();b=generate_city(100).manifest();self.assertEqual(a,b)
        hashes={generate_city(s).manifest()['geometry_sha256'] for s in [100,101,10000,20000,30000,40000]}
        self.assertEqual(len(hashes),6)
        self.assertNotEqual(generate_city(100,family='shift').manifest()['geometry_sha256'],a['geometry_sha256'])
    def test_map_routes_do_not_depend_on_true_parcel_labels(self):
        f=Fleet(2,2,3);generate_city(101,3).apply(f);d=FleetDispatch('priority',adaptive_modes=True)
        obs,_=d.observation(f)
        for j in f.jobs.values():j.color='blue';j.address='hidden replacement';j.recipient='secret'
        again,_=d.observation(f)
        np.testing.assert_array_equal(obs['candidates'],again['candidates'])
    def test_local_dispatch_map_does_not_reveal_distant_obstacle(self):
        f=Fleet(2,2,3);generate_city(101,3).apply(f);m=LocalObstacleMap(f,radius=.5);m.tick(f)
        self.assertLess(len(f.observed_obstacles),len(m.truth));self.assertEqual(f.motion.obstacles,m.truth)
        f.agents['ground_0'].pos[:]=list(m.truth[0].low);m.tick(f)
        self.assertIn(m.truth[0],f.observed_obstacles)
    def test_custom_goal_transfer_and_home_are_used(self):
        f=Fleet(1,1,1,'relay');c=generate_city(123,1);c.apply(f)
        for _ in range(2000):
            f.tick(.2)
            for j in f.jobs.values():
                if j.phase=='observe' and j.destination is None:f.decide(j.name,j.color)
            if f.done:break
        self.assertTrue(f.done)
        j=f.jobs['package_0'];self.assertEqual(j.pos,j.delivery_goals[j.destination])
        self.assertAlmostEqual(f.agents['ground_0'].pos[0],c.home_x)
        self.assertNotEqual(j.transfer[0],0.)
    def test_frozen_evaluation_does_not_update_weights(self):
        m=ScoringPolicy(4);before=digest(m);r,_=run(20000,120000,model=m,max_time=5)
        self.assertEqual(digest(m),before);self.assertTrue(r['weights_unchanged']);self.assertFalse(r['complete']);self.assertEqual(r['delivered'],0)
    def test_route_endpoint_rounding_does_not_create_a_false_blockage(self):
        from pai_ros.airspace import plan_outbound,RestrictedArea
        # Target is outside the .42 margin, but its nearest grid cell is inside.
        a=RestrictedArea('closure',-2.8,-2.4,-1.4,-.6)
        target=[-3.,-.12,.08]
        route=plan_outbound([0.,0.,3.2],target,[a],0.)
        self.assertEqual(route[-1],[-3.,-.12,3.2])
        self.assertFalse(any(a.intersects(x,y,.42) for x,y in zip(route,route[1:])))
    def test_physical_joint_route_clears_buildings_and_airspace(self):
        from pai_ros.physical_motion import PhysicalMotion
        from pai_ros.live_scenario import LiveScenario
        from pai_ros.safety import Obstacle
        f=Fleet(2,2,3,motion=PhysicalMotion());generate_city(21000,3).apply(f)
        f.motion.set_obstacles([Obstacle(o.name,tuple(v-.2 for v in o.low),tuple(v+.2 for v in o.high)) for o in f.motion.obstacles])
        sc=LiveScenario(f);f.time=30;sc.tick(f)
        j=f.jobs['package_0'];j.destination='red';start=[j.source[0]+.45,j.source[1],.25]
        route=f.joint_route(start,f.goal(j),j.source[1])
        for a,b in zip(route,route[1:]):
            self.assertTrue(f.motion.static_clear([a[0],a[1],.25],[b[0],b[1],.25],'ground'))
            self.assertFalse(any(r.intersects(a,b,.42) for r in f.restricted_areas))
    def test_sdf_collisions_match_generated_map(self):
        f=Fleet(1,1,2);c=generate_city(456,2);c.apply(f);root=ET.fromstring(build_world(f,render=False,physical=True))
        models={m.attrib['name']:m for m in root.findall('world/model')}
        for o in c.obstacles:
            m=models[o.name];self.assertIsNotNone(m.find('link/collision'))
            pos=list(map(float,m.findtext('pose').split()[:3]));self.assertEqual(pos,[(a+b)/2 for a,b in zip(o.low,o.high)])
        self.assertNotIn('trolley_collision',models)
if __name__=='__main__':unittest.main()
