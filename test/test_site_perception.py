import json,unittest
import numpy as np
from pai_ros.fleet import Fleet,PRIORITY_COLORS
from pai_ros.city import generate_city
from pai_ros.rendezvous import configure
from pai_ros.service_budget import BudgetDispatch,configure as budget
from pai_ros.site_perception import SiteState,fixture_image,fixture_observe,parse_sites
from pai_ros.cargo_vlm import priority_decide

class SitePerceptionTests(unittest.TestCase):
 def fleet(self):
  f=Fleet(2,2,6,dispatcher=BudgetDispatch('value'));f.configure_speeds()
  generate_city(75100,6,layout='blocks').apply(f);configure(f);budget(f,300)
  f.site_perception=SiteState(f.city,'fixture',f);return f
 def observation(self,f,blocked=()):
  return fixture_observe({k:fixture_image(k in blocked) for k in f.site_perception.positions})
 def test_unknown_sites_are_not_guessed_available(self):
  f=self.fleet();self.assertFalse(f.city.pickup_available(f,f.jobs['package_0']))
  self.assertEqual(f.dispatcher.observation(f),(None,[]))
 def test_image_observation_changes_pickup_candidates(self):
  f=self.fleet();f.site_perception.apply(self.observation(f,{'pickup_A'}))
  _,rows=f.dispatcher.observation(f)
  self.assertTrue(rows);self.assertTrue(all(f.city.pickup_index(f.jobs[r['package']].source)==1 for r in rows))
  f.site_perception.apply(self.observation(f,{'pickup_B'}));_,rows=f.dispatcher.observation(f)
  self.assertTrue(rows);self.assertTrue(all(f.city.pickup_index(f.jobs[r['package']].source)==0 for r in rows))
 def test_image_closure_replans_using_remaining_streets(self):
  f=self.fleet();f.site_perception.apply(self.observation(f));j=f.jobs['package_0']
  j.ground='ground_0';j.uav='uav_0';j.phase='ground_pickup';j.destination=j.numbered_destination
  self.assertTrue(f.meeting.plan(f,j));old=j.transfer.copy()
  key='meeting_'+'AB'[f.city.meeting_sites.index(old)]
  f.site_perception.apply(self.observation(f,{key}));self.assertTrue(f.meeting.plan(f,j,'site_image_changed'))
  self.assertEqual(j.meeting_revision,2)
  self.assertTrue(f.events[-1]['event']=='meeting_selected')
  self.assertEqual(f.events[-1]['site_changed'],old!=j.transfer)
 def test_road_closure_blocks_direct_approach_but_keeps_meeting_pad_clear(self):
  f=self.fleet();f.site_perception.apply(self.observation(f))
  start=f.agents['ground_0'].pos;pad=f.city.meeting_sites[1];target=[pad[0]+.25,pad[1],.25]
  self.assertIsNotNone(f.motion.plan(start,target,'ground'))
  f.site_perception.apply(self.observation(f,{'meeting_B'}))
  self.assertTrue(f.motion.static_clear(target,target,'ground'))
  self.assertFalse(f.motion.static_clear([7.,pad[1],.25],target,'ground'))
  detour=f.motion.plan(start,target,'ground')
  self.assertIsNotNone(detour) # an open alternate street must remain usable
  self.assertTrue(all(f.motion.static_clear(a,b,'ground') for a,b in zip([start]+detour,detour)))
  other=f.city.meeting_sites[0]
  self.assertIsNotNone(f.motion.plan(start,[other[0]+.25,other[1],.25],'ground'))
  # Reopening comes from new pixels, not the authored obstruction flags.
  f.site_perception.apply(self.observation(f))
  self.assertIsNotNone(f.motion.plan(start,target,'ground'))
 def test_destinations_follow_hut_fronts_and_neighborhoods(self):
  f=self.fleet();sites=f.city.delivery_sites
  self.assertEqual([s['position'] for s in sites[3:]], [[-10.8,1.,1.28],[-10.8,9.,1.28],[-10.8,17.,1.28]])
  for site in sites[:3]:
   x,y,_=site['position']
   self.assertIsNotNone(f.motion.plan(f.agents['ground_0'].pos,[x,y,.25],'ground'))
 def test_malformed_vlm_json_cannot_open_a_site(self):
  with self.assertRaises(ValueError):parse_sites('{"pickup_A":{"available":"true"}}',['pickup_A'])
  with self.assertRaises(ValueError):parse_sites('{}',['pickup_A'])
 def test_priority_color_does_not_choose_destination(self):
  f=self.fleet();j=f.jobs['package_0'];address=j.numbered_destination;point=list(j.delivery_goals[address])
  for priority in (1,2,3):
   j.priority=priority;j.phase='observe';j.destination=None
   image=np.full((32,32,3),np.array(PRIORITY_COLORS[priority])*255,dtype=np.uint8)
   color,_=priority_decide(image,offline=True);self.assertTrue(f.decide(j.name,color))
   self.assertEqual(j.destination,address);self.assertEqual(f.goal(j),point)
   self.assertEqual(j.recognized_priority,priority)
 def test_only_six_neutral_numbered_destinations(self):
  f=self.fleet();self.assertEqual([s['id'] for s in f.city.delivery_sites],['D01','D02','D03','D04','D05','D06'])
  self.assertTrue(all(len(j.delivery_goals)==1 for j in f.jobs.values()))
if __name__=='__main__':unittest.main()
