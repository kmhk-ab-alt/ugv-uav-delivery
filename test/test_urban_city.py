import unittest
import xml.etree.ElementTree as ET
from pai_ros.city import generate_city
from pai_ros.fleet import Fleet
from pai_ros.rendezvous import configure
from pai_ros.service_budget import BudgetDispatch,configure as budget,orders
from pai_ros.gazebo_scene import build_world

class UrbanTests(unittest.TestCase):
 def test_street_and_building_counts_do_not_grow_with_orders(self):
  a=generate_city(75100,3,layout='blocks');b=generate_city(75100,18,layout='blocks')
  self.assertEqual(a.roads,b.roads);self.assertEqual(a.buildings,b.buildings)
  self.assertEqual(len(b.sources),18);self.assertEqual(len(set(map(tuple,b.sources))),18)
  self.assertTrue(any(r['a'][0]==r['b'][0] for r in b.roads));self.assertTrue(any(r['a'][1]==r['b'][1] for r in b.roads))
 def test_two_distinct_collection_and_meeting_facilities(self):
  f=Fleet(2,2,12);generate_city(75100,12,layout='blocks').apply(f);configure(f)
  self.assertEqual(len(f.city.pickup_sites),2);self.assertEqual(len(f.city.meeting_sites),2)
  self.assertTrue(set(map(tuple,f.city.pickup_sites)).isdisjoint(set(map(tuple,f.city.meeting_sites))))
  first=f.jobs['package_0'];first.phase='approach'
  self.assertFalse(f.city.pickup_available(f,f.jobs['package_1']))
  self.assertTrue(f.city.pickup_available(f,f.jobs['package_2']))
  first.destination=first.numbered_destination
  candidates=f.meeting.candidates(f,first)
  self.assertEqual(len(candidates),2)
  first.transfer=f.city.meeting_sites[0].copy();first.meeting_revision=1;first.phase='uav_delivery'
  other=f.jobs['package_1'];other.destination=other.numbered_destination;other.ground='ground_1';other.uav='uav_1'
  _,records=f.meeting.observation(f,other)
  self.assertEqual(len(records),1)
  self.assertEqual(records[0]['position'],f.city.meeting_sites[1])
 def test_compact_city_removes_unused_western_blocks(self):
  c=generate_city(75100,12,layout='blocks')
  self.assertFalse(any(b['name'].startswith('west_') for b in c.buildings))
  self.assertEqual(len(c.buildings),19)
  self.assertTrue(all(min(r['a'][0],r['b'][0])>=1 for r in c.roads))
  self.assertEqual(c.meeting_sites,[[1.,4.,.08],[1.,12.,.08]])
  self.assertEqual(len(c.pickup_sites),2);self.assertEqual(len(c.meeting_sites),2)
  self.assertTrue(all(-14<g[0]<-3 for i,goals in enumerate(c.goals) if i%2==0 for g in goals))
 def test_extra_agents_extend_parking_without_overlapping_homes(self):
  f=Fleet(8,8,12);generate_city(75100,12,layout='blocks').apply(f);configure(f)
  self.assertEqual(len(set(tuple(f.home(a)) for a in f.agents.values())),16)
  for a in f.agents.values():
   if a.kind=='ground':self.assertTrue(f.motion.street_network.covers(a.pos,a.pos))
 def test_robot_route_turns_through_connected_streets(self):
  f=Fleet(2,2,12);generate_city(75100,12,layout='blocks').apply(f);configure(f)
  start=[21.,-4.,.25];target=[7.15,8.9,.25]
  self.assertFalse(f.motion.static_clear(start,target,'ground'))
  path=f.motion.plan(start,target,'ground');self.assertIsNotNone(path);self.assertGreaterEqual(len(path),2)
  self.assertTrue(all(f.motion.static_clear(a,b,'ground') for a,b in zip([start]+path,path)))
  detour=f.motion.plan(start,[1.,8.9,.25],'ground')
  self.assertIsNotNone(detour)
  self.assertTrue(all(p[1]<23 for p in detour))
  self.assertEqual(f.rendezvous_obstacles,[])
  self.assertTrue(all(f.motion.static_clear(a,b,'ground') for a,b in zip([start]+detour,detour)))
 def test_mountain_delivery_is_elevated_and_unreachable_by_ugv(self):
  f=Fleet(2,2,12);generate_city(75100,12,layout='blocks').apply(f);configure(f)
  for j in f.jobs.values():
   if f.delivery_mode(j)=='ground':continue
   for goal in j.delivery_goals.values():
    self.assertAlmostEqual(goal[2],1.28)
    self.assertFalse(f.motion.street_network.covers(goal,goal))
    self.assertIsNone(f.motion.plan(f.agents['ground_0'].pos,goal,'ground'))
 def test_mixed_ground_and_uav_delivery_without_reserving_unused_uav(self):
  f=Fleet(2,2,4);f.configure_speeds();generate_city(75100,4,layout='blocks').apply(f);configure(f)
  ground_seen=False
  for _ in range(6000):
   f.tick(.1)
   for j in f.jobs.values():
    if j.phase=='observe':f.decide(j.name,__import__('pai_ros.fleet',fromlist=['PRIORITY_TOKENS']).PRIORITY_TOKENS[j.priority])
    if j.mode=='ground' and j.phase!='queued':
     self.assertIsNone(j.uav)
     ground_seen=ground_seen or j.phase=='ground_delivery'
   f.events.clear()
   if f.done:break
  self.assertTrue(ground_seen);self.assertTrue(f.done)
  self.assertTrue(all(j.receipt_confirmed for j in f.jobs.values()))
 def test_ground_dispatch_does_not_require_available_uav(self):
  f=Fleet(2,2,4,dispatcher=BudgetDispatch('value'));f.configure_speeds()
  generate_city(75100,4,layout='blocks').apply(f);configure(f);budget(f,300)
  for a in f.agents.values():
   if a.kind=='uav':a.task='other_flight'
  _,records=f.dispatcher.observation(f)
  self.assertTrue(records)
  self.assertTrue(all(r['mode']=='ground' and r['uav'] is None for r in records))
 def test_delivery_pads_clear_road_and_do_not_overlap_adjacent_slots(self):
  from pai_ros.urban_city import scenery
  f=Fleet(2,2,12);generate_city(75100,12,layout='blocks').apply(f)
  palette={'.78 .80 .78 1'}
  pads=[g for g in ET.fromstring('<root>'+scenery(f)+'</root>') if g.get('rgba') in palette]
  self.assertEqual(len(pads),6)
  ground=[]
  for g in pads:
   x,y,z=map(float,g.get('pos').split());radius,h=map(float,g.get('size').split())
   self.assertAlmostEqual(radius,.30)
   if z<1:
    self.assertGreater(z+h,.03)
    ground.append((x,y,radius))
  for i,(x,y,radius) in enumerate(ground):
   for a,b,r in ground[i+1:]:self.assertGreaterEqual(__import__('math').hypot(x-a,y-b),radius+r)
 def test_reused_home_route_is_replanned_from_new_position(self):
  f=Fleet(package_count=3);generate_city(81000,3,layout='blocks').apply(f)
  a=f.agents['ground_0'];a.pos[:]=[13.,12.,.25];home=f.home(a)
  f.motion.paths[(a.name,tuple(home))]=[[19.,-2.,.25],home]
  for _ in range(1000):
   f.motion.begin_step(.1,f.agents)
   if f.motion.move(a,home,.9,.1):break
  self.assertLess(__import__('math').dist(a.pos,home),1e-6)
 def test_timed_rollout_keeps_robots_on_pavement_and_returns(self):
  f=Fleet(2,2,4,dispatcher=BudgetDispatch('value'));generate_city(81000,4,layout='blocks').apply(f);configure(f);budget(f,90);orders(f,181000)
  for _ in range(2200):
   f.tick(.25)
   for a in f.agents.values():
    if a.kind=='ground':self.assertTrue(f.motion.street_network.covers(a.pos,a.pos))
   for j in f.jobs.values():
    if j.phase=='observe':f.decide(j.name,__import__('pai_ros.fleet',fromlist=['PRIORITY_TOKENS']).PRIORITY_TOKENS[j.priority])
   f.events.clear()
   if f.done:break
  self.assertTrue(f.done);self.assertTrue(all(a.task is None for a in f.agents.values()))
 def test_sdf_uses_same_buildings_for_visual_and_collision(self):
  f=Fleet(2,2,12);generate_city(75100,12,layout='blocks').apply(f);configure(f)
  world=ET.fromstring(build_world(f,render=False,physical=True,real_time_factor=.5)).find('world')
  self.assertEqual(world.findtext('physics/real_time_factor'),'0.5')
  self.assertEqual(world.findtext("model[@name='package_0']/plugin/odom_publish_frequency"),'10')
  self.assertEqual(world.findtext("model[@name='ground_0']/plugin[@name='gz::sim::systems::OdometryPublisher']/odom_publish_frequency"),'50')
  self.assertIsNotNone(world.find("model[@name='urban_layout']"))
  for o in f.city.obstacles:self.assertIsNotNone(world.find(f"model[@name='{o.name}']/link/collision"))
if __name__=='__main__':unittest.main()
