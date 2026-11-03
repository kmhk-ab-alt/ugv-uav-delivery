import tempfile
from pathlib import Path
import unittest
import numpy as np
from pai_ros.dispatch_env import DispatchEnv,DispatchConfig
from pai_ros.dispatch_policy import PriorityDeadlinePolicy
from pai_ros.dispatch_model import ScoringPolicy,Adam

class ModelTest(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        obs,_=DispatchEnv().reset(seed=2);model=ScoringPolicy(seed=5)
        action=PriorityDeadlinePolicy().select(obs);_,grads=model.imitation(obs,action)
        for key,index in (('w1',(0,0)),('b1',(0,)),('w2',(0,))):
            original=model.params[key][index];eps=1e-5
            model.params[key][index]=original+eps;plus=model.imitation(obs,action)[0]
            model.params[key][index]=original-eps;minus=model.imitation(obs,action)[0]
            model.params[key][index]=original
            self.assertAlmostEqual(grads[key][index],(plus-minus)/(2*eps),places=6)

    def test_bc_learns_example(self):
        obs,_=DispatchEnv().reset(seed=2);model=ScoringPolicy(seed=1)
        action=PriorityDeadlinePolicy().select(obs);initial=model.imitation(obs,action)[0]
        opt=Adam(model.params,lr=.01)
        for _ in range(100):loss,gradient=model.imitation(obs,action);opt.step(model.params,gradient)
        self.assertLess(model.imitation(obs,action)[0],initial)
        self.assertTrue(obs['action_mask'][model.select(obs)])

    def test_reinforce_gradient_increases_positive_action_probability(self):
        obs,_=DispatchEnv().reset(seed=2);m=ScoringPolicy(seed=1)
        action=PriorityDeadlinePolicy().select(obs);legal,_,_,p=m.forward(obs);index=list(legal).index(action)
        before=p[index];grads=m.reinforce(obs,action,1.)
        for k in m.params:m.params[k]-=.01*grads[k]
        self.assertGreater(m.forward(obs)[3][index],before)

    def test_checkpoint_and_variable_agent_counts(self):
        model=ScoringPolicy(seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';model.save(path);loaded=ScoringPolicy.load(path)
            for g,u in ((1,1),(2,2),(3,1)):
                obs,_=DispatchEnv(DispatchConfig(g,u)).reset(seed=0)
                self.assertEqual(model.select(obs),loaded.select(obs))
                self.assertTrue(obs['action_mask'][loaded.select(obs)])

if __name__=='__main__':unittest.main()

class TrainingIntegrationTest(unittest.TestCase):
    def test_cpu_training_writes_loadable_checkpoints_and_heldout_report(self):
        from pai_ros.train_dispatch import train
        import json
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            report=train(output,bc_episodes=2,bc_epochs=2,rl_episodes=2,eval_episodes=1)
            self.assertIn('untrained',report['test']);self.assertIn('bc',report['test']);self.assertIn('rl_last',report['test'])
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertFalse(set(manifest['bc_seeds']+manifest['rl_seeds']) & set(manifest['test_seeds']+manifest['validation_seeds']))
            obs,_=DispatchEnv().reset(seed=1234)
            loaded=ScoringPolicy.load(output/'best.npz')
            self.assertTrue(obs['action_mask'][loaded.select(obs)])
