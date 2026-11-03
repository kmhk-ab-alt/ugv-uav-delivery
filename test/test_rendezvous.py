import tempfile,unittest
from pathlib import Path
from pai_ros.fleet import Fleet
from pai_ros.city import generate_city
from pai_ros.rendezvous import configure,MeetingNetwork
from pai_ros.airspace import RestrictedArea
from pai_ros.dispatch_model import ScoringPolicy

class MeetingTests(unittest.TestCase):
    def scene(self,policy='eta'):
        f=Fleet(1,1,1);generate_city(22000,1).apply(f);configure(f,policy)
        j=f.jobs['package_0'];j.ground='ground_0';j.uav='uav_0';j.destination='red'
        f.agents[j.ground].pos[:]=[j.source[0],j.source[1],.25]
        return f,j
    def test_uav_waits_home_during_ground_pickup(self):
        f=Fleet(1,1,1);configure(f);home=f.agents['uav_0'].pos.copy()
        for _ in range(100):
            f.tick(.02)
            self.assertEqual(f.agents['uav_0'].pos,home)
        self.assertEqual(f.jobs['package_0'].phase,'observe')
    def test_inspection_required_and_true_color_is_not_used(self):
        f,j=self.scene();obs,records=f.meeting.observation(f,j)
        j.color='blue';again,_=f.meeting.observation(f,j)
        import numpy as np
        np.testing.assert_array_equal(obs['candidates'],again['candidates'])
        j.destination=None
        with self.assertRaises(ValueError):f.meeting.observation(f,j)
    def test_eta_choice_minimizes_declared_cost(self):
        f,j=self.scene();_,records=f.meeting.observation(f,j)
        self.assertGreater(len(records),1);self.assertTrue(f.meeting.plan(f,j))
        self.assertAlmostEqual(j.meeting_estimates['cost'],min(r['cost'] for r in records))
        self.assertTrue(all(p[0]>-2.25 for _,p in f.meeting.candidates(f,j) if p==j.transfer))
    def test_new_restriction_changes_meeting_and_invalidates_route(self):
        f,j=self.scene();f.meeting.plan(f,j);old=j.transfer.copy();j.route_phase='to_transfer'
        f.restricted_areas.append(RestrictedArea('new_closure',old[0]-.1,old[0]+.1,old[1]-.1,old[1]+.1))
        self.assertTrue(f.meeting.plan(f,j,'map_changed'))
        self.assertNotEqual(j.transfer,old);self.assertEqual(j.meeting_revision,2);self.assertEqual(j.route_phase,'')
    def test_fixed_policy_keeps_reference_site(self):
        f,j=self.scene('fixed');self.assertTrue(f.meeting.plan(f,j));self.assertEqual(j.transfer,f.fixed_meetings[j.name])
    def test_dispatch_checkpoint_cannot_be_misused_as_meeting_model(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'wrong.npz';ScoringPolicy().save(p)
            with self.assertRaisesRegex(ValueError,'rendezvous checkpoint'):MeetingNetwork.load(p)
            p=Path(d)/'right.npz';MeetingNetwork().save(p);MeetingNetwork.load(p)
    def test_loaded_uav_keeps_altitude_compensation_while_safety_held(self):
        from pai_ros.physical_motion import PhysicalMotion
        from pai_ros.fleet import Agent
        from pai_ros.safety import Obstacle
        m=PhysicalMotion();a=Agent('uav_0','uav',[0.,0.,2.]);m.payloads={'uav_0'};m.integral_z[a.name]=.5
        m.set_obstacles([Obstacle('blocked',(-.1,-100,0),(100,100,4))]);m.begin_step(.02,{a.name:a})
        m.move(a,[1,0,2],.6,.02);self.assertGreater(m.velocity_commands[a.name][2],0)
        a.pos[2]=1.9;m.begin_step(.02,{a.name:a});m.move(a,[1,0,2],.6,.02)
        self.assertAlmostEqual(m.hold_targets[a.name][2],2.);self.assertGreater(m.velocity_commands[a.name][2],.5)
    def test_relay_end_to_end_without_road_vehicle(self):
        from pai_ros.rendezvous_experiment import run
        r,_=run(72100,packages=3)
        self.assertTrue(r['complete']);self.assertEqual(r['delivered'],3);self.assertEqual(r['handoffs'],3);self.assertFalse(r['errors'])
if __name__=='__main__':unittest.main()
