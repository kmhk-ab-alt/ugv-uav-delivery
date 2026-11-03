import json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET
from pai_ros.fleet import Fleet,Agent
from pai_ros.city import generate_city
from pai_ros.rendezvous import configure
from pai_ros.gazebo_scene import build_world
from pai_ros.showcase_server import scene_document,LiveData
from pai_ros import physical_models

class ShowcaseTests(unittest.TestCase):
    def test_preview_does_not_fabricate_live_mission_state(self):
        s=LiveData(True).snapshot();self.assertTrue(s['preview']);self.assertIsNone(s['state']);self.assertIsNone(s['age_s'])
        data=LiveData();state={'time':1,'complete':False};data.update_state(state)
        self.assertEqual(data.snapshot()['state'],state)
    def test_visual_detail_does_not_change_vehicle_physics(self):
        def physics(s):
            r=ET.fromstring(s)
            for link in r.findall('link'):
                for visual in link.findall('visual'):link.remove(visual)
            return ET.tostring(r)
        g=Agent('ground_0','ground',[0,0,.25]);u=Agent('uav_0','uav',[0,0,.2])
        ground=physical_models.ground_model(g);uav=physical_models.uav_model(u)
        with patch.object(physical_models,'ground_details',return_value=''),patch.object(physical_models,'drone_details',return_value=''):
            self.assertEqual(physics(ground),physics(physical_models.ground_model(g)))
            self.assertEqual(physics(uav),physics(physical_models.uav_model(u)))
    def test_scene_geometry_is_shared_and_preserves_real_initial_positions(self):
        f=Fleet(2,2,3);generate_city(75100,3).apply(f);configure(f)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'scene.sdf';p.write_text(build_world(f,False,True,True));d=scene_document(p)
        self.assertEqual(d['mission_count'],3)
        m=next(m for m in d['models'] if m['name']=='uav_0');self.assertEqual(m['pose'][:3],f.agents['uav_0'].pos)
        mesh=next(m for m in d['models'] if m['name']=='background_ridge')['links'][0]['objects'][0]
        self.assertTrue(all(math.isfinite(x) for v in mesh['vertices'] for x in v));self.assertLessEqual(max(v[0] for v in mesh['vertices']),-8)
        self.assertFalse('physical_floor' in [m['name'] for m in d['models']]) # collision-only models need no visuals
    def test_live_event_buffer_is_bounded(self):
        d=LiveData()
        for i in range(100):d.update_event({'time':i})
        self.assertEqual(len(d.snapshot()['events']),24);self.assertEqual(d.snapshot()['events'][-1]['time'],99)
if __name__=='__main__':unittest.main()
