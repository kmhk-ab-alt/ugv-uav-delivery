import math, unittest
from pai_ros.fleet import Agent
from pai_ros.physical_motion import PhysicalMotion
from pai_ros.street_network import StreetNetwork

class PhysicalCornerTests(unittest.TestCase):
 def setup_corner(self,x=1.7):
  m=PhysicalMotion();m.ground_speed_limit=2.4
  m.street_network=StreetNetwork([dict(a=[1,20],b=[19,20],width=2.4),dict(a=[1,4],b=[1,20],width=2.4)])
  a=Agent('ground_0','ground',[x,20.,.25]);a.yaw=math.pi
  goal=[1.25,12.,.25];m.paths[(a.name,tuple(goal))]=[[1.,20.,.25],goal.copy()]
  return m,a,goal
 def test_does_not_cut_corner_when_final_goal_becomes_visible(self):
  m,a,goal=self.setup_corner();m.begin_step(.05,{a.name:a})
  self.assertTrue(m.static_clear(a.pos,goal,'ground'))
  m.move(a,goal,2.4,.05)
  self.assertEqual(m.hold_targets[a.name],[1.,20.,.25])
  self.assertAlmostEqual(m.velocity_commands[a.name][3],0.)
 def test_stops_before_turning_even_inside_waypoint_tolerance(self):
  m,a,goal=self.setup_corner(1.05);m.last_positions[a.name]=[1.15,20.,.25]
  m.begin_step(.05,{a.name:a});m.move(a,goal,2.4,.05)
  self.assertEqual(m.velocity_commands[a.name],[0.,0.,0.,0.])
  self.assertEqual(m.paths[(a.name,tuple(goal))][0],[1.,20.,.25])
 def test_lagged_loaded_vehicle_stays_on_road_at_corner(self):
  m,a,goal=self.setup_corner(10.);v=2.4;omega=0.;dt=.05;reached=False
  # Conservative plant: actual slowing is limited to .45m/s², below the drive's
  # command acceleration; approximates the measured loaded-vehicle overshoot.
  for _ in range(2400):
   m.begin_step(dt,{a.name:a});reached=m.move(a,goal,2.4,dt)
   self.assertTrue(m.street_network.covers(a.pos,a.pos),a.pos)
   if reached:break
   cmd=m.velocity_commands[a.name]
   v+=max(-.45*dt,min(1.2*dt,cmd[0]-v))
   omega+=max(-3*dt,min(3*dt,cmd[3]-omega))
   a.yaw+=omega*dt;a.pos[0]+=math.cos(a.yaw)*v*dt;a.pos[1]+=math.sin(a.yaw)*v*dt
  self.assertTrue(reached)
if __name__=='__main__':unittest.main()
