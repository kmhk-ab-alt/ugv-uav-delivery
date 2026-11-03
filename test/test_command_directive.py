import unittest
from concurrent.futures import Future
from types import SimpleNamespace
import numpy as np
from pai_ros.fleet import Fleet
from pai_ros.city import generate_city
from pai_ros.site_perception import SiteSurvey,fixture_observe,fixture_image,parse_directive,command_context
from pai_ros.service_budget import BudgetDispatch

class CommandDirectiveTests(unittest.TestCase):
    def test_bad_directives_rejected(self):
        for order in ([3,3,1],[3,2],['3',2,1],[True,2,3],None):
            with self.assertRaises(ValueError):parse_directive(dict(priority_order=order,reason='test'))
        with self.assertRaises(ValueError):parse_directive(dict(priority_order=[3,2,1],reason=''))

    def test_survey_publishes_explicit_fixture_source(self):
        f=Fleet(1,1,4);generate_city(75100,4,layout='blocks').apply(f)
        survey=SiteSurvey(f,events=False)
        try:
            survey.future=Future()
            survey.future.set_result(fixture_observe({k:fixture_image(False) for k in survey.state.positions}))
            self.assertTrue(survey.poll())
            self.assertEqual(survey.state.directive['priority_order'],[3,2,1])
            self.assertEqual(f.events[-1]['event'],'command_directive')
            self.assertEqual(f.events[-1]['source'],'fixture')
            self.assertEqual(survey.state.snapshot()['directive'],survey.state.directive)
            self.assertEqual(len(command_context(f)['orders']),4)
        finally:survey.close()

    def dispatch(self,priorities,feasible,order=(3,2,1)):
        d=BudgetDispatch('value')
        records=[dict(package='package_'+str(i),ground='ground_0',uav=None,mode='ground',priority=p,feasible=ok,value_rate=i+1,receipt_eta_s=10)
                 for i,(p,ok) in enumerate(zip(priorities,feasible))]
        obs=dict(candidates=np.zeros((len(records),13)),action_mask=np.ones(len(records)),global_state=np.zeros(6))
        d.observation=lambda f:(obs,records)
        f=SimpleNamespace(time=0,operation_limit=180,site_perception=SimpleNamespace(pending=False,directive=dict(priority_order=list(order),source='vlm',reason='test',version=1)))
        return d,f

    def test_highest_feasible_priority_precedes_lower_value_score(self):
        d,f=self.dispatch([3,3,1],[True,True,True])
        self.assertEqual(d.select(f)[0],'package_1')
        self.assertEqual(d.last_decision['selected_priority'],3)
        self.assertEqual(len(d.last_decision['ranking']),2)

    def test_unreachable_or_late_high_priority_does_not_block_other_work(self):
        d,f=self.dispatch([3,2,1],[False,True,True])
        self.assertEqual(d.select(f)[0],'package_1')
        self.assertEqual(d.last_decision['selected_priority'],2)

    def test_model_receives_only_permitted_candidates(self):
        d,f=self.dispatch([1,3,3],[True,True,True])
        class Network:
            def select(self,obs):
                self.count=len(obs['candidates']);return 1
            def forward(self,obs):return np.arange(2),None,None,np.array([.2,.8])
        d.policy='learned';d.network=Network()
        self.assertEqual(d.select(f)[0],'package_2')
        self.assertEqual(d.network.count,2)

    def test_pending_perception_or_no_feasible_candidate_never_dispatches(self):
        d,f=self.dispatch([3,2],[False,False]);self.assertIsNone(d.select(f))
        d,f=self.dispatch([3],[True]);f.site_perception.pending=True;self.assertIsNone(d.select(f))

if __name__=='__main__':unittest.main()
