import unittest,math
from pai_ros.fleet import Fleet,Agent,PRIORITY_TOKENS
from pai_ros.city import generate_city
from pai_ros.rendezvous import configure
from pai_ros.service_budget import BudgetDispatch,configure as budget
from pai_ros.physical_motion import PhysicalMotion,track

class PipelineTests(unittest.TestCase):
 def fleet(self):
  f=Fleet(1,1,4,dispatcher=BudgetDispatch('value'));generate_city(75100,4,layout='blocks').apply(f);configure(f);budget(f,300);f.configure_speeds()
  for j in f.jobs.values():j.deadline=1000;j.release=10000
  old=f.jobs['package_0'];new=f.jobs['package_2'];old.release=new.release=0
  old.ground='ground_0';old.uav='uav_0';old.mode='relay';old.phase='handoff';old.elapsed=1;old.destination=old.numbered_destination;old.carrier=old.ground
  g=f.agents[old.ground];u=f.agents[old.uav];g.task=u.task=old.name
  g.pos=[*old.transfer[:2],.25];u.pos=[*old.transfer[:2],.82];old.pos=[*old.transfer[:2],.5]
  return f,old,new,g,u
 def test_handoff_releases_ground_and_dispatches_next_pickup_with_busy_uav(self):
  f,old,new,g,u=self.fleet();f.tick(.1)
  self.assertTrue(old.ground_released);self.assertIsNone(g.task);self.assertEqual(u.task,old.name)
  f.tick(.1)
  self.assertEqual(g.task,new.name);self.assertIsNone(new.uav);self.assertEqual(u.task,old.name)
 def test_old_uav_return_never_commands_or_releases_reassigned_ground(self):
  f,old,new,g,u=self.fleet();f.tick(.1);f.tick(.1)
  old.phase='return_home';new.phase='observe';new.destination=None
  f.tick(.1)
  self.assertFalse(any(c.get('agent')==g.name and c.get('command_type')=='position_target' for c in f.motion.commands))
  old.phase='dock';u.pos=f.home(u).copy();f.tick(.1)
  self.assertEqual(g.task,new.name);self.assertIsNone(u.task)
 def test_prepicked_cargo_waits_then_binds_returned_uav(self):
  f,old,new,g,u=self.fleet();f.tick(.1);f.tick(.1)
  new.phase='ground_pickup';new.destination=new.numbered_destination;g.pos=[new.source[0],new.source[1],.25]
  f.tick(.1);self.assertEqual(new.phase,'await_uav');self.assertEqual(new.carrier,g.name)
  old.phase='dock';u.pos=f.home(u).copy();f.tick(.1);f.tick(.1)
  self.assertEqual(new.uav,u.name);self.assertEqual(u.task,new.name);self.assertEqual(g.task,new.name)
 def test_uav_does_not_start_descent_while_crossing_goal_at_speed(self):
  m=PhysicalMotion();a=Agent('uav_0','uav',[-10.8,17.,3.2]);m.last_positions[a.name]=[-10.6,17.,3.2]
  m.begin_step(.05,{a.name:a});self.assertFalse(m.move(a,a.pos.copy(),5.2,.05))
  self.assertEqual(m.commands[-1]['command_type'],'brake_for_waypoint')
 def test_loaded_uav_brakes_before_delivery_with_velocity_lag(self):
  m=PhysicalMotion();a=Agent('uav_0','uav',[-1.,17.,3.2]);target=[-10.8,17.,3.2];vx=-3.;dt=.05;arrived=False;minimum=a.pos[0]
  for _ in range(1600):
   m.begin_step(dt,{a.name:a});arrived=m.move(a,target,5.2,dt)
   if arrived:break
   desired=m.velocity_commands[a.name][0];vx+=max(-.8*dt,min(.8*dt,desired-vx));a.pos[0]+=vx*dt;minimum=min(minimum,a.pos[0])
  self.assertTrue(arrived);self.assertGreater(minimum,-11.1);self.assertLess(abs(vx),.12)
if __name__=='__main__':unittest.main()
