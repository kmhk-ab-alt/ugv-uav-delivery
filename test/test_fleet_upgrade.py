import unittest
import math
import xml.etree.ElementTree as ET
import numpy as np
from pai_ros.fleet import Fleet
from pai_ros.fleet_dispatch import FleetDispatch
from pai_ros.physical_motion import PhysicalMotion,track
from pai_ros.physical_cargo import PhysicalCargo
from pai_ros.live_scenario import LiveScenario
from pai_ros.gazebo_scene import build_world

class UpgradeTests(unittest.TestCase):
    def test_physical_waypoint_command_never_integrates_pose(self):
        motion=PhysicalMotion();f=Fleet(motion=motion);a=f.agents['ground_0'];before=a.pos.copy()
        motion.begin_step(.02,f.agents);self.assertFalse(motion.move(a,[3,0,.25],.9,.02));self.assertEqual(a.pos,before)
        self.assertEqual(motion.velocity_commands[a.name][0],0) # Turn first, rather than slide sideways.
    def test_uav_world_to_body_velocity(self):
        v=track('uav',[0,0,0],math.pi/2,[1,0,0],1)
        self.assertAlmostEqual(v[0],0);self.assertAlmostEqual(v[1],-1)
    def test_dispatch_cannot_read_true_color(self):
        d=FleetDispatch();f=Fleet(dispatcher=d);a,_=d.observation(f)
        for j in f.jobs.values():j.color='blue';j.contents='hidden';j.address='hidden'
        b,_=d.observation(f);np.testing.assert_array_equal(a['candidates'],b['candidates'])
    def test_unreleased_and_busy_are_masked(self):
        d=FleetDispatch(adaptive_modes=True);f=Fleet(1,1,2,dispatcher=d)
        list(f.jobs.values())[1].release=20
        _,actions=d.observation(f);self.assertTrue(all(a[0]=='package_0' for a in actions.values()))
        f.agents['ground_0'].task='busy';self.assertIsNone(d.select(f))
    def test_dynamic_disruption_is_removed_and_late_job_is_urgent(self):
        f=Fleet();scenario=LiveScenario(f);f.time=13;scenario.tick(f)
        self.assertTrue(f.disruption_active);self.assertEqual(list(f.jobs.values())[-1].release,22)
        f.time=91;scenario.tick(f);self.assertFalse(f.disruption_active)
        self.assertNotIn(scenario.area,f.restricted_areas)
    def test_physical_models_have_rotors_wheels_mass_and_collision(self):
        r=ET.fromstring(build_world(Fleet(2,2,3),False,True))
        for m in r.findall('./world/model'):
            if m.attrib['name'].startswith(('ground_','uav_','package_')):self.assertNotEqual(m.findtext('static'),'true')
        motors=[x for x in r.findall('.//plugin') if x.attrib.get('name')=='gz::sim::systems::MulticopterMotorModel']
        self.assertEqual(len(motors),8);self.assertIsNotNone(r.find('.//collision'))
    def test_physical_cargo_state_is_only_feedback(self):
        f=Fleet();j=f.jobs['package_0'];p=j.pos.copy();c=PhysicalCargo();c.follow(j,f.agents['ground_0'])
        self.assertEqual(j.pos,p);self.assertEqual(c.commands[-1]['command_type'],'physical_grip')
        c.place(j,[-3,0,.08]);self.assertEqual(j.pos,p)

if __name__=='__main__':unittest.main()

class JointDeliveryRegression(unittest.TestCase):
    def test_final_offset_stays_while_ground_turns_before_release(self):
        from pai_ros.motion import KinematicMotion
        class FakePhysical(KinematicMotion):
            name='gazebo_diffdrive_rotor_feedback'
            def __init__(self):super().__init__();self.turns=0
            def move(self,a,target,speed,dt):a.pos[:]=target;return True
            def face(self,a,yaw):self.turns+=1;return self.turns>=3
        f=Fleet(1,1,1,'joint',motion=FakePhysical());j=f.jobs['package_0'];j.ground='ground_0';j.uav='uav_0';j.mode='joint';j.destination='red';j.phase='joint_delivery';j.route_phase=j.phase;j.uav_route=[[-3,0,3.2]]
        f.tick(.02);x=f.agents['ground_0'].pos[0];f.tick(.02)
        self.assertAlmostEqual(x,-2.55);self.assertAlmostEqual(f.agents['ground_0'].pos[0],x)

class TeacherDataRegression(unittest.TestCase):
    def test_native_bc_teaches_against_competing_transport_modes(self):
        from pai_ros.fleet_experiment import run
        _,samples=run(1,'priority',collect=True,mode='relay')
        self.assertTrue(samples)
        obs,action=samples[0];self.assertTrue(obs['action_mask'][action])
        legal=obs['candidates'][obs['action_mask'].astype(bool)]
        self.assertTrue(any(row[10]>0 for row in legal))
        self.assertTrue(any(row[11]>0 for row in legal))
