"""ROS-only adapter tests; run in the Gazebo environment."""
import unittest
from concurrent.futures import Future
from pai_ros.fleet import Fleet
try:
    from pai_ros.gazebo_world import GazeboWorld
except ModuleNotFoundError:
    raise unittest.SkipTest("Gazebo bridge tests require the ROS/Gazebo environment")
from rosgraph_msgs.msg import Clock

class Client:
    def __init__(self):self.requests=[];self.futures=[]
    def wait_for_service(self,timeout_sec):return True
    def call_async(self,req):
        self.requests.append(req);f=Future();self.futures.append(f);return f
class Node:
    def __init__(self):self.client=Client()
    def create_client(self,*args):return self.client
    def create_subscription(self,*args):pass
class AdapterTests(unittest.TestCase):
    def setUp(self):self.node=Node();self.world=GazeboWorld(Fleet(),self.node,False)
    def test_waits_for_clock_and_pose_acknowledgment(self):
        self.assertEqual(self.world.step_time(),0)
        msg=Clock();msg.clock.sec=1;self.world.clock(msg);self.assertEqual(self.world.step_time(),0)
        msg.clock.nanosec=20000000;self.world.clock(msg);self.assertAlmostEqual(self.world.step_time(),.02)
        self.world.sync();self.assertEqual(len(self.node.client.requests),5)
        self.assertEqual(self.world.step_time(),0)
        for f in self.node.client.futures:
            f.set_result(type('Response',(),{'success':True})())
        msg.clock.nanosec=40000000;self.world.clock(msg);self.assertAlmostEqual(self.world.step_time(),.02)
    def test_rejected_pose_is_failure(self):
        self.world.sync()
        for f in self.node.client.futures:f.set_result(type('Response',(),{'success':False})())
        with self.assertRaisesRegex(RuntimeError,'rejected'):self.world.step_time()
    def test_absent_camera_is_not_synthetic(self):self.assertIsNone(self.world.image('package_0'))
if __name__=='__main__':unittest.main()

class Publisher:
    def __init__(self):self.messages=[]
    def publish(self,message):self.messages.append(message)
class PhysicalNode(Node):
    def __init__(self):super().__init__();self.publishers={}
    def create_publisher(self,msg_type,topic,qos):
        pub=Publisher();self.publishers[topic]=pub;return pub
class PhysicalBridgeTests(unittest.TestCase):
    def test_physical_vehicles_and_parcels_never_use_set_pose(self):
        from pai_ros.physical_motion import PhysicalMotion
        from pai_ros.physical_cargo import PhysicalCargo
        f=Fleet(motion=PhysicalMotion(),cargo=PhysicalCargo());n=PhysicalNode();w=GazeboWorld(f,n,False,True)
        f.motion.begin_step(.02,f.agents);w.sync()
        self.assertEqual(n.client.requests,[])
        self.assertFalse(n.publishers['/uav_0/enable'].messages[-1].data)
        self.assertTrue(n.publishers['/model/ground_0/cmd_vel'].messages)

class MeetingPhysicalBridgeTests(unittest.TestCase):
    def test_parked_uav_stays_disarmed_until_meeting_departure(self):
        from pai_ros.physical_motion import PhysicalMotion
        from pai_ros.rendezvous import configure
        f=Fleet(motion=PhysicalMotion());configure(f);n=PhysicalNode();w=GazeboWorld(f,n,False,True)
        j=f.jobs['package_0'];j.uav='uav_0';f.agents['uav_0'].task=j.name
        for phase,enabled in [('approach',False),('observe',False),('ground_pickup',False),('to_transfer',True)]:
            j.phase=phase;f.motion.begin_step(.02,f.agents);w.sync()
            self.assertEqual(n.publishers['/uav_0/enable'].messages[-1].data,enabled)
            for future in n.client.futures:
                if not future.done():future.set_result(type('Response',(),{'success':True})())
            w.pending=[]

    def test_large_physical_queue_only_sends_changed_meeting_markers(self):
        from pai_ros.physical_motion import PhysicalMotion
        from pai_ros.rendezvous import configure
        f=Fleet(2,2,12,motion=PhysicalMotion());configure(f);n=PhysicalNode();w=GazeboWorld(f,n,False,True)
        f.motion.begin_step(.02,f.agents);w.sync()
        self.assertEqual(len(n.client.requests),0)
        f.jobs['package_4'].meeting_revision=1
        w.sync();self.assertEqual(len(n.client.requests),1)
        w.sync();self.assertEqual(len(n.client.requests),1)
        f.jobs['package_4'].receipt_confirmed=True
        w.sync();self.assertEqual(len(n.client.requests),2)
        self.assertLess(n.client.requests[-1].pose.position.z,0)

class LockstepBridgeTests(unittest.TestCase):
    def make_world(self):
        import threading,time
        from pai_ros.physical_motion import PhysicalMotion
        w=GazeboWorld.__new__(GazeboWorld);w.fleet=Fleet(2,2,12,motion=PhysicalMotion());w.lock=threading.Lock();w.lockstep=True;w.physical=True
        w.now=.1;w.last=.08;w.step_target=.1;w.step_sent=time.monotonic();w.pending=[];w.active_since={}
        w.odom={n:(a.pos.copy(),0.,.1) for n,a in w.fleet.agents.items()};w.parcel_odom={n:.1 for n in w.fleet.jobs};w.advances=[];w.advance=lambda:w.advances.append(True)
        return w
    def test_bootstrap_waits_for_low_rate_parcel_feedback(self):
        w=self.make_world();w.parcel_odom={}
        self.assertEqual(w.step_time(),0);self.assertEqual(w.advances,[True]);self.assertEqual(w.fleet.time,0)
    def test_waits_until_the_requested_physics_batch_finishes(self):
        w=self.make_world();w.fleet.time=1.;w.now=.09
        self.assertEqual(w.step_time(),0);self.assertEqual(w.last,.08);self.assertEqual(w.advances,[])
        w.now=.1;self.assertAlmostEqual(w.step_time(),.02)
    def test_unacknowledged_cosmetic_marker_does_not_stall_control(self):
        w=self.make_world();w.fleet.time=1.;w.pending=[Future()]
        self.assertAlmostEqual(w.step_time(),.02);self.assertEqual(len(w.pending),1)
    def test_stale_vehicle_feedback_holds_the_paused_world(self):
        w=self.make_world();w.fleet.time=1.;w.odom['ground_0']=(w.fleet.agents['ground_0'].pos.copy(),0.,0.)
        self.assertEqual(w.step_time(),0);self.assertEqual(w.advances,[])
    def test_idle_startup_can_refresh_dropped_final_feedback(self):
        import time
        w=self.make_world();w.step_sent=time.monotonic()-3
        w.odom['ground_0']=(w.fleet.agents['ground_0'].pos.copy(),0.,0.)
        self.assertEqual(w.step_time(),0);self.assertEqual(w.advances,[True])
        self.assertEqual(w.fleet.time,0);self.assertEqual(w.last,.1)
    def test_startup_refresh_is_bounded_and_reports_stale_agent(self):
        import time
        w=self.make_world();w.step_sent=time.monotonic()-31;w.startup_feedback_retries=3
        w.odom['ground_0']=(w.fleet.agents['ground_0'].pos.copy(),0.,0.)
        with self.assertRaisesRegex(RuntimeError,'ground_0'):w.step_time()
        self.assertEqual(w.advances,[])
    def test_startup_does_not_advance_unfinished_physics_or_assigned_vehicle(self):
        import time
        for assigned in (False,True):
            w=self.make_world();w.step_sent=time.monotonic()-3
            if assigned:
                w.fleet.agents['ground_0'].task='package_0'
                w.odom['ground_0']=(w.fleet.agents['ground_0'].pos.copy(),0.,0.)
            else:w.now=.09
            self.assertEqual(w.step_time(),0);self.assertEqual(w.advances,[])
    def test_requested_batch_and_target_clock_match_the_physics_step(self):
        w=self.make_world();del w.advance
        requests=[]
        class Control:
            def request(self,*args):requests.append(args[1]);return True,type('Reply',(),{'data':True})()
        w.step_node=Control();w.advance()
        self.assertTrue(requests[0].pause);self.assertEqual(requests[0].multi_step,25)
        self.assertAlmostEqual(w.step_target-w.now,25*.002)
