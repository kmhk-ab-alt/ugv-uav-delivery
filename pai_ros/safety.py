"""Conservative kinematic avoidance, not a physical flight safety controller."""
from dataclasses import dataclass
import heapq
import math
from .motion import KinematicMotion

@dataclass(frozen=True)
class Obstacle:
    name: str
    low: tuple
    high: tuple
    def intersects(self,a,b,extent):
        near,far=0.,1.
        for x,y,lo,hi,r in zip(a,b,self.low,self.high,extent):
            lo-=r;hi+=r
            d=y-x
            if abs(d)<1e-10:
                if x<lo or x>hi:return False
            else:
                q0,q1=(lo-x)/d,(hi-x)/d
                near=max(near,min(q0,q1));far=min(far,max(q0,q1))
                if near>far:return False
        return True

EXTENTS={'ground':(.32,.30,.23),'uav':(.36,.36,.15)}

def swept_conflict(a0,a1,b0,b1,ka,kb):
    radii=[EXTENTS[ka][i]+EXTENTS[kb][i]+.05 for i in range(3)]
    p=[(a0[i]-b0[i])/radii[i] for i in range(3)]
    v=[((a1[i]-a0[i])-(b1[i]-b0[i]))/radii[i] for i in range(3)]
    vv=sum(x*x for x in v)
    t=max(0.,min(1.,-sum(x*y for x,y in zip(p,v))/vv)) if vv>1e-12 else 0.
    return sum((p[i]+t*v[i])**2 for i in range(3))<1.-1e-9

class SafeKinematicMotion(KinematicMotion):
    name='safe_kinematic_demo'
    def __init__(self):
        super().__init__();self.obstacles=[];self.waits={};self.escape={};self.paths={}
        self.route_preview={};self.hold_count=0;self.detour_count=0;self.events=[];self.restrictions=[]
    def set_restrictions(self,areas):
        self.restrictions=list(areas)
        if hasattr(self,'air_network'):
            self.air_network[:]=areas;self.air_network.cached.clear()
    def set_obstacles(self,obstacles): self.obstacles=list(obstacles);self.paths.clear()
    def begin_step(self,dt,agents):
        super().begin_step(dt,agents)
        self.agents=agents;self.start={a.name:a.pos.copy() for a in agents.values()};self.events=[]
    def static_clear(self,a,b,kind):
        if kind=='ground' and hasattr(self,'street_network') and not self.street_network.covers(a,b):return False
        extent=tuple(x+.04 for x in EXTENTS[kind])
        return not any(o.intersects(a,b,extent) for o in self.obstacles) and (kind!='uav' or not any(r.intersects(a,b,.40) for r in self.restrictions))
    def blocker(self,agent,next_pos):
        for other in self.agents.values():
            if other.name==agent.name:continue
            if swept_conflict(self.start[agent.name],next_pos,self.start[other.name],other.pos,agent.kind,other.kind):
                return other
        return None
    def plan(self,start,target,kind):
        # Plan at the current altitude; vertical segments wait rather than cut obstacles.
        if abs(start[2]-target[2])>.1:return None
        if kind=='ground' and hasattr(self,'street_network'):
            key=tuple((o.name,o.low,o.high) for o in self.obstacles)
            return self.street_network.route(start,target,lambda a,b:self.static_clear(a,b,kind),key)
        if kind=='uav' and hasattr(self,'air_network'):
            try:return self.air_network.plan(start,target,z=start[2])
            except ValueError:return None
        step=.25;z=start[2]
        origin=(round(start[0]/step),round(start[1]/step));goal=(round(target[0]/step),round(target[1]/step))
        point=lambda c:[c[0]*step,c[1]*step,z]
        if not self.static_clear(start,point(origin),kind) or not self.static_clear(point(goal),target,kind):return None
        queue=[(0,origin)];parent={origin:None};cost={origin:0}
        margin=24 if hasattr(self,'street_network') else 8
        xmin=min(origin[0],goal[0])-margin;xmax=max(origin[0],goal[0])+margin
        ymin=min(origin[1],goal[1])-margin;ymax=max(origin[1],goal[1])+margin
        while queue:
            _,c=heapq.heappop(queue)
            if c==goal:break
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                n=(c[0]+dx,c[1]+dy)
                if not(xmin<=n[0]<=xmax and ymin<=n[1]<=ymax):continue
                if not self.static_clear(point(c),point(n),kind):continue
                score=cost[c]+1
                if score<cost.get(n,math.inf):
                    parent[n]=c;cost[n]=score
                    heapq.heappush(queue,(score+abs(n[0]-goal[0])+abs(n[1]-goal[1]),n))
        if goal not in parent:return None
        path=[];c=goal
        while c is not None:path.append(point(c));c=parent[c]
        return list(reversed(path))+[list(target)]
    def move(self,agent,target,speed,dt):
        target=list(target);original=target.copy()
        if agent.kind=='ground':self.route_preview[agent.name]=[agent.pos.copy(),original.copy()]
        if not self.static_clear(agent.pos,target,agent.kind):
            key=(agent.name,tuple(target))
            if key not in self.paths or self.paths[key]==[] or (self.paths[key] and not self.static_clear(agent.pos,self.paths[key][0],agent.kind)):
                self.paths[key]=self.plan(agent.pos,target,agent.kind)
                if self.paths[key]:
                    self.detour_count+=1
                    self.events.append(dict(event='obstacle_detour',agent=agent.name))
            path=self.paths[key]
            while path and math.dist(agent.pos,path[0])<1e-6:path.pop(0)
            if path:
                target=path[0]
                if agent.kind=='ground':self.route_preview[agent.name]=[agent.pos.copy()]+[p.copy() for p in path]
            else:
                self.hold_count+=1
                self.commands.append(dict(agent=agent.name,command_type='safety_hold',reason='static obstacle/no route'))
                return False
        if agent.name in self.escape:target=self.escape[agent.name]
        distance=math.dist(agent.pos,target)
        fraction=min(1.,speed*dt/distance) if distance else 1.
        candidate=[agent.pos[i]+(target[i]-agent.pos[i])*fraction for i in range(3)]
        other=self.blocker(agent,candidate)
        if other is not None or not self.static_clear(agent.pos,candidate,agent.kind):
            self.hold_count+=1;self.waits[agent.name]=self.waits.get(agent.name,0.)+dt
            if self.waits[agent.name]<=dt+1e-9:
                self.events.append(dict(event='collision_hold',agent=agent.name,other=other.name if other else 'static'))
            self.commands.append(dict(agent=agent.name,command_type='safety_hold',reason=other.name if other else 'static'))
            # Deterministic yielding: lower-order agent keeps its course; the other sidesteps.
            # A stationary parked agent may also be bypassed, without requiring it to move.
            if other and self.waits[agent.name]>.6:
                dx,dy=original[0]-agent.pos[0],original[1]-agent.pos[1]
                norm=math.hypot(dx,dy)
                if norm<=.01: dx,dy,norm=1.,0.,1.
                if norm>0:
                    right=math.atan2(dy,dx)-math.pi/2
                    for offset in (0.,.4,-.4,.8,-.8,1.2,-1.2,1.6,-1.6,2.,-2.,2.8):
                        angle=right+offset
                        sidestep=[agent.pos[0]+math.cos(angle)*.9,agent.pos[1]+math.sin(angle)*.9,agent.pos[2]]
                        if self.static_clear(agent.pos,sidestep,agent.kind) and self.blocker(agent,sidestep) is None:
                            self.escape[agent.name]=sidestep;break
            return False
        self.waits[agent.name]=0.
        agent.pos[:]=candidate
        self.commands.append(dict(agent=agent.name,command_type='position_target',target=target,speed_limit=speed))
        if agent.name in self.escape and math.dist(candidate,target)<1e-6:self.escape.pop(agent.name)
        return math.dist(agent.pos,original)<1e-6
