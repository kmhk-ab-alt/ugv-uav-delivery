import unittest
import xml.etree.ElementTree as ET
from pai_ros.fleet import Fleet
from pai_ros.gazebo_scene import build_world,visuals

class GazeboSceneTests(unittest.TestCase):
    def test_configurable_counts_and_modes(self):
        for mode in ('relay','scout','joint'):
            f=Fleet(3,2,4,mode)
            root=ET.fromstring(build_world(f))
            names=[m.attrib['name'] for m in root.findall('./world/model')]
            self.assertEqual(len(names),15)  # includes the visual-only background ridge
            self.assertEqual(len(names),len(set(names)))
            self.assertEqual(len(root.findall('.//sensor')),4)
            self.assertEqual(root.find('.//render_engine').text,'ogre2')
            self.assertEqual(root.find('.//gui/plugin/engine').text,'ogre2')
    def test_stub_has_no_camera(self):
        root=ET.fromstring(build_world(Fleet(),render=False))
        self.assertEqual(root.findall('.//sensor'),[])
    def test_mujoco_half_extents_become_sdf_full_size(self):
        root=ET.fromstring('<root>'+visuals('<geom type="box" pos="1 2 3" size=".1 .2 .3" rgba="1 0 0 1"/>')+'</root>')
        self.assertEqual(root.find('.//size').text,'0.2 0.4 0.6')
        self.assertEqual(root.find('.//pose').text,'1.0 2.0 3.0 0 0 0')
    def test_inspection_camera_points_down(self):
        root=ET.fromstring(build_world(Fleet()))
        camera=root.find('.//sensor')
        self.assertAlmostEqual(float(camera.find('pose').text.split()[4]),1.57079632679)
        self.assertEqual(camera.find('camera/image/format').text,'R8G8B8')

if __name__=='__main__':unittest.main()
