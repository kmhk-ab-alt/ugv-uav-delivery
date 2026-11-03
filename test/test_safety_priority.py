import unittest
from pai_ros.fleet import Fleet, Agent
from pai_ros.safety import SafeKinematicMotion, Obstacle, swept_conflict
from pai_ros.scheduling import parse_priorities

class SafetyPriorityTest(unittest.TestCase):
    def test_priority_then_deadline_and_no_preemption(self):
        f=Fleet(1,1,3,priorities=[1,3,2]);f.tick(.02)
        self.assertEqual(f.agents['ground_0'].task,'package_1')
        f.jobs['package_0'].priority=3
        f.tick(.02)
        self.assertEqual(f.agents['ground_0'].task,'package_1')
        f=Fleet(1,1,3,priorities=[2,2,2]);f.jobs['package_2'].deadline=10
        f.tick(.02)
        self.assertEqual(f.agents['ground_0'].task,'package_2')

    def test_priorities_validate(self):
        self.assertEqual(parse_priorities('1,3,2',3),[1,3,2])
        for text in ('1,3','0,1,2','1,x,3'):
            with self.assertRaises(ValueError):parse_priorities(text,3)

    def test_head_on_yields_without_swept_collision(self):
        a=Agent('ground_0','ground',[-1.,0.,.25],task='a')
        b=Agent('ground_1','ground',[1.,0.,.25],task='b')
        m=SafeKinematicMotion();agents={a.name:a,b.name:b}
        goals={a.name:[1.,0.,.25],b.name:[-1.,0.,.25]}
        for _ in range(2000):
            previous={x.name:x.pos.copy() for x in agents.values()}
            m.begin_step(.02,agents)
            for x in agents.values():m.move(x,goals[x.name],.8,.02)
            self.assertFalse(swept_conflict(previous[a.name],a.pos,previous[b.name],b.pos,a.kind,b.kind))
            if all(x.pos==goals[x.name] for x in agents.values()):break
        self.assertEqual(a.pos,goals[a.name]);self.assertEqual(b.pos,goals[b.name])
        self.assertGreater(m.hold_count,0)

    def test_static_obstacle_detour(self):
        a=Agent('ground_0','ground',[-1.,0.,.25],task='a')
        m=SafeKinematicMotion();wall=Obstacle('wall',(-.1,-.3,0),(.1,.3,1))
        m.set_obstacles([wall])
        for _ in range(2000):
            before=a.pos.copy();m.begin_step(.02,{a.name:a})
            arrived=m.move(a,[1.,0.,.25],.8,.02)
            self.assertFalse(wall.intersects(before,a.pos,(.36,.34,.27)))
            if arrived:break
        self.assertEqual(a.pos,[1.,0.,.25]);self.assertGreater(m.detour_count,0)

    def test_unknown_vertical_blockage_holds(self):
        a=Agent('uav_0','uav',[0.,0.,.2],task='a')
        m=SafeKinematicMotion();m.set_obstacles([Obstacle('ceiling',(-1,-1,1),(1,1,1.1))])
        m.begin_step(.02,{a.name:a})
        self.assertFalse(m.move(a,[0,0,2],1.,.02))
        self.assertEqual(a.pos,[0.,0.,.2])

if __name__=='__main__':unittest.main()

class RichSceneSafetyTest(unittest.TestCase):
    def test_priority_delivery_with_real_scene_obstacles(self):
        from pai_ros.fleet_world import FleetWorld
        for mode in ('relay','scout','joint'):
            with self.subTest(mode=mode):
                f=Fleet(2,2,3,mode,priorities=[1,3,2]);world=FleetWorld(f)
                try:
                    self.assertGreater(len(f.motion.obstacles),100)
                    for _ in range(5000):
                        f.tick(.1)
                        for j in f.jobs.values():
                            if j.phase=='observe':f.decide(j.name,j.color)
                        if f.done:break
                    self.assertTrue(f.done,[(j.name,j.phase) for j in f.jobs.values()])
                    first=[e['package'] for e in f.events if e['event']=='assigned'][:2]
                    self.assertEqual(first,['package_1','package_2'])
                    self.assertGreater(f.motion.detour_count,0)
                finally:world.close()
