import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from pai_ros.dispatch_env import DispatchEnv,DispatchConfig,DEPOT,HUB,edge
from pai_ros.dispatch_policy import PriorityDeadlinePolicy

class DispatchTest(unittest.TestCase):
    def test_reset_reproducible_and_fixed_action_slots(self):
        env=DispatchEnv();a,_=env.reset(seed=4);b,_=env.reset(seed=4)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        self.assertEqual(a['candidates'].shape,(97,13))
        for action in np.flatnonzero(a['action_mask']):
            if action:
                oid,g,u,mode=env.decode(action)
                self.assertEqual(env.encode(oid,int(g.split('_')[1]),int(u.split('_')[1]),mode),action)

    def test_hidden_future_orders_and_online_arrival(self):
        env=DispatchEnv();obs,_=env.reset(seed=3)
        self.assertEqual(sum(o.status=='waiting' for o in env.orders),2)
        for o in env.orders:
            if o.release>0:
                for gi in range(2):
                    for ui in range(2):
                        for m in ('relay','scout','joint'):
                            action=env.encode(o.id,gi,ui,m)
                            self.assertFalse(obs['action_mask'][action]);self.assertFalse(obs['candidates'][action].any())
        for _ in range(61):env.step(0)
        self.assertTrue(all(o.status=='waiting' for o in env.orders))

    def test_payload_and_battery_masks(self):
        env=DispatchEnv();env.reset(seed=1);o=env.orders[0];o.weight=3.3
        obs=env.observation()
        self.assertFalse(obs['action_mask'][env.encode(0,0,0,'relay')])
        self.assertFalse(obs['action_mask'][env.encode(0,0,0,'scout')])
        self.assertTrue(obs['action_mask'][env.encode(0,0,0,'joint')])
        env.vehicles['uav_0'].battery=0
        self.assertFalse(env.observation()['action_mask'][env.encode(0,0,0,'joint')])
        env.step(0)
        self.assertGreater(env.vehicles['uav_0'].battery,0)

    def test_invalid_action_does_not_advance_time(self):
        env=DispatchEnv();env.reset(seed=2)
        with self.assertRaises(ValueError):env.step(env.action_count)
        self.assertEqual(env.time,0)

    def test_closure_detour_and_reopen(self):
        env=DispatchEnv();env.reset(seed=2)
        direct=env.route('ground',DEPOT,(2,0))
        env.time=20
        detour=env.route('ground',DEPOT,(2,0))
        self.assertGreater(len(detour),len(direct))
        self.assertNotIn(edge(HUB,(2,0)),[edge(a,b) for a,b in zip(detour,detour[1:])])
        env.time=70
        self.assertEqual(env.route('ground',DEPOT,(2,0)),direct)

    def test_capacity_reservations_and_finite_episode(self):
        for mode in ('relay','scout','joint'):
            env=DispatchEnv(DispatchConfig(1,1,1,240));obs,_=env.reset(seed=0)
            env.orders[0].weight=.5
            action=env.encode(0,0,0,mode)
            obs,reward,term,trunc,info=env.step(action)
            self.assertEqual(env.vehicles['ground_0'].task,0)
            for _ in range(240):
                self.assertTrue(all(v.battery>=0 for v in env.vehicles.values()))
                for kind in ('ground','uav'):
                    links=[edge(*v.transit) for v in env.vehicles.values() if v.kind==kind and v.transit]
                    self.assertEqual(len(links),len(set(links)))
                if term or trunc:break
                obs,reward,term,trunc,info=env.step(0)
            self.assertTrue(term,(mode,info))
            self.assertEqual(info['delivered'],1)
            self.assertTrue(all(v.node==DEPOT and v.task is None for v in env.vehicles.values()))

    def test_wait_only_truncates_with_failure_penalty(self):
        env=DispatchEnv(DispatchConfig(orders=2,horizon=3));env.reset(seed=0)
        for _ in range(3):obs,reward,term,trunc,info=env.step(0)
        self.assertFalse(term);self.assertTrue(trunc);self.assertLess(reward,-1)
        with self.assertRaises(RuntimeError):env.step(0)

    def test_priorities_rewards(self):
        rewards=[]
        for priority in (1,3):
            env=DispatchEnv(DispatchConfig(1,1,1));env.reset(seed=0)
            env.orders[0].priority=priority;env.orders[0].weight=.5;env.orders[0].deadline=200
            _,r,t,x,info=env.step(env.encode(0,0,0,'relay'));total=r
            while not(t or x):_,r,t,x,info=env.step(0);total+=r
            rewards.append(total)
        self.assertAlmostEqual(rewards[1]-rewards[0],20.)

    def test_no_render_or_api_imports(self):
        code='from pai_ros.dispatch_env import DispatchEnv; import sys; assert not any(m in sys.modules for m in ("mujoco","rclpy","openai"))'
        subprocess.run([sys.executable,'-c',code],check=True)

    def test_baseline_small_and_unbalanced_counts(self):
        for g,u in ((1,1),(3,1),(1,3)):
            env=DispatchEnv(DispatchConfig(g,u,4,240));obs,_=env.reset(seed=6);policy=PriorityDeadlinePolicy()
            for _ in range(240):
                obs,r,term,trunc,info=env.step(policy.select(obs))
                if term or trunc:break
            self.assertTrue(term or trunc)
            self.assertTrue(math.isfinite(info['reward']))
            self.assertTrue(all(v.battery>=0 for v in env.vehicles.values()))

    def test_shared_edge_reservations_with_multiple_agents(self):
        env=DispatchEnv(DispatchConfig(3,3,8));obs,_=env.reset(seed=8);policy=PriorityDeadlinePolicy()
        for _ in range(240):
            obs,r,t,x,info=env.step(policy.select(obs))
            for kind in ('ground','uav'):
                links=[edge(*v.transit) for v in env.vehicles.values() if v.kind==kind and v.transit]
                self.assertEqual(len(links),len(set(links)))
            if t or x:break

    def test_joint_carriers_share_position_and_edge(self):
        env=DispatchEnv(DispatchConfig(1,1,1));env.reset(seed=1);env.orders[0].weight=3.3
        env.step(env.encode(0,0,0,'joint'))
        for _ in range(240):
            order=env.orders[0]
            if order.stage=='delivery':
                g,u=env.vehicles['ground_0'],env.vehicles['uav_0']
                self.assertEqual(g.node,u.node);self.assertEqual(g.transit,u.transit);self.assertEqual(g.remaining,u.remaining)
            _,_,t,x,_=env.step(0)
            if t or x:break

if __name__=='__main__':unittest.main()
