"""Repeatable, observable road/airspace disruptions and a late urgent order."""
from .safety import Obstacle
from .airspace import RestrictedArea
import random

class LiveScenario:
    def __init__(self,fleet):
        self.active=False;self.finished=False;self.urgent_announced=False;self.base_obstacles=None;self.base_areas=list(fleet.restricted_areas)
        self.y=next(iter(fleet.jobs.values())).source[1]
        self.obstacle=Obstacle('temporary_roadwork',(.45,self.y-.35,0),(1.05,self.y+.35,.7))
        self.area=RestrictedArea('temporary_airspace',-2.8,-2.4,self.y-1.4,self.y-.6)
        self.road_position=[.75,self.y,.35]
        self.visual_position=[.75,self.y,-2.]
        self.air_position=[-2.6,self.y-1.,-2.]
        self.start_time=12.;self.end_time=90.;self.urgent_time=22.
        if hasattr(fleet,'city'):
            rng=random.Random(fleet.city.seed+7301)
            self.start_time=rng.uniform(10,28);self.end_time=self.start_time+rng.uniform(45,75);self.urgent_time=rng.uniform(18,38)
        fleet.disruption_active=False
        if len(fleet.jobs)>2:
            j=list(fleet.jobs.values())[-1];j.release=self.urgent_time;j.priority=3;j.deadline=self.urgent_time+138.
    def tick(self,fleet):
        if len(fleet.jobs)>2 and not self.urgent_announced and fleet.time>=self.urgent_time:
            j=list(fleet.jobs.values())[-1];fleet.event(j,'urgent_order_released',priority=j.priority,deadline=j.deadline);self.urgent_announced=True
        if self.base_obstacles is None:self.base_obstacles=list(fleet.motion.obstacles)
        if not self.active and not self.finished and fleet.time>=self.start_time:
            # Do not spawn a barrier or airspace closure around an occupied vehicle.
            if any(self.obstacle.intersects(a.pos,a.pos,(.45,.45,.25)) or (a.kind=='uav' and self.area.contains(a.pos,.5)) for a in fleet.agents.values()):return
            self.active=True;fleet.disruption_active=True
            fleet.motion.set_obstacles(self.base_obstacles+[self.obstacle])
            fleet.restricted_areas=self.base_areas+[self.area];fleet.motion.set_restrictions(fleet.restricted_areas)
            self.visual_position=self.road_position.copy();self.air_position=[-2.6,self.y-1.,.035]
            fleet.events.append(dict(time=round(fleet.time,3),event='disruption_started',road='temporary_roadwork',airspace=self.area.name))
            for j in fleet.jobs.values():
                if j.phase in ('uav_delivery','joint_delivery'):
                    j.route_phase='';fleet.event(j,'route_invalidated',reason='new airspace restriction')
        if self.active and fleet.time>=self.end_time:
            self.active=False;self.finished=True;fleet.disruption_active=False
            fleet.motion.set_obstacles(self.base_obstacles);fleet.restricted_areas=self.base_areas;fleet.motion.set_restrictions(self.base_areas)
            self.visual_position=[.75,self.y,-2.];self.air_position=[-2.6,self.y-1.,-2.]
            fleet.events.append(dict(time=round(fleet.time,3),event='disruption_ended'))
