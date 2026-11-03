import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock
from pai_ros.fleet import Fleet
from pai_ros.city import generate_city
from pai_ros.site_perception import SiteSurvey,fixture_image,fixture_directive

class SurveyFrequencyTests(unittest.TestCase):
    def setup_survey(self,events=False):
        f=Fleet(1,1,4);generate_city(75100,4,layout='blocks').apply(f)
        s=SiteSurvey(f,'vlm',events=events)
        self.addCleanup(s.close)
        s.state.apply({k:dict(available=True,reason='open') for k in s.state.positions},fixture_directive())
        return f,s
    def test_time_alone_never_requests_another_observation(self):
        f,s=self.setup_survey();s.pool.submit=Mock()
        for t in (0,20,40,180,1000):
            f.time=t;self.assertTrue(s.poll())
        s.pool.submit.assert_not_called();self.assertFalse(s.state.pending)
    def test_road_change_requests_once_and_preserves_observation(self):
        f,s=self.setup_survey(events=True);job=next(iter(f.jobs.values()))
        job.phase='to_transfer';job.elapsed=3;job.transfer=f.city.meeting_sites[1].copy()
        job.ground='ground_0';job.uav='uav_0'
        world=SimpleNamespace(now=100.,image=lambda key:fixture_image(False),image_stamps={'site_'+k:101. for k in s.state.positions})
        future=Future();s.pool.submit=Mock(return_value=future)
        self.assertFalse(s.poll(world));self.assertTrue(s.state.pending)
        self.assertFalse(s.poll(world));self.assertEqual(s.pool.submit.call_count,1)
        for _ in range(10):self.assertFalse(s.poll(world))
        self.assertEqual(s.pool.submit.call_count,1)
        future.set_result(dict(sites={k:dict(available=k!='meeting_B',reason='observed') for k in s.state.positions},directive=fixture_directive()))
        self.assertTrue(s.poll(world));self.assertFalse(s.state.sites['meeting_B']['available'])
        f.time=1000;self.assertTrue(s.poll(world));self.assertEqual(s.pool.submit.call_count,1)
    def test_initial_observation_is_still_required(self):
        _,s=self.setup_survey();s.state.pending=True
        world=SimpleNamespace(now=0.,image=lambda key:None)
        s.pool.submit=Mock()
        self.assertFalse(s.poll(world));self.assertFalse(s.poll(world));s.pool.submit.assert_not_called()

if __name__=='__main__':unittest.main()
