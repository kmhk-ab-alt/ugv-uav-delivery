import math
import unittest
from pai_ros.fleet import Fleet
from pai_ros.physical_motion import PhysicalMotion,track

class SpeedProfileTests(unittest.TestCase):
 def test_physical_ugv_cruise_command_and_close_range_braking(self):
  f=Fleet(motion=PhysicalMotion());f.configure_speeds(1.2,2.6)
  self.assertEqual(f.estimated_ground_speed,1.2)
  self.assertAlmostEqual(track('ground',[0,0,.25],0,[10,0,.25],f.estimated_ground_speed)[0],1.2)
  self.assertLess(track('ground',[0,0,.25],0,[.1,0,.25],f.estimated_ground_speed)[0],.2)
  self.assertEqual(track('ground',[0,0,.25],math.pi,[10,0,.25],1.2)[0],0)
 def test_invalid_speed_is_rejected(self):
  for speed in (0,-1,float('nan'),float('inf')):
   with self.assertRaises(ValueError):Fleet().configure_speeds(speed,2.6)
 def test_uav_cruise_command_obeys_configured_limit(self):
  v=track('uav',[0,0,3.2],0,[10,0,3.2],2.6)
  self.assertAlmostEqual(math.sqrt(sum(x*x for x in v[:3])),2.6)

if __name__=='__main__':unittest.main()
