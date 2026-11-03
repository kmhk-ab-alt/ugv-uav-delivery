"""CPU-only dispatch learning environment: no ROS, VLM, or rendering.

Units: seconds, metres, kg, and abstract battery units (NOT a flight energy model).
Fixed action slots + validity mask support changing availability during an episode.
"""
from dataclasses import dataclass, asdict
from collections import deque
import math
import random
import numpy as np

MODES=('relay','scout','joint')
FEATURES=('priority','weight','deadline_slack','estimated_seconds','ground_battery',
          'uav_battery','ground_capacity','uav_capacity','distance',
          'relay','scout','joint','wait')
NODES=tuple((x,y) for x in range(3) for y in (-1,0,1))
DEPOT=(0,0);HUB=(1,0)

def edge(a,b): return tuple(sorted((a,b)))

@dataclass(frozen=True)
class DispatchConfig:
    ground_count: int=2
    uav_count: int=2
    orders: int=8
    horizon: int=240
    dt: float=1.
    closures: bool=True
    initial_battery: float=100.
    def __post_init__(self):
        for k in ('ground_count','uav_count','orders','horizon'):
            if type(getattr(self,k)) is not int or getattr(self,k)<1: raise ValueError(k+' must be positive integer')
        if not math.isfinite(self.dt) or self.dt<=0: raise ValueError('dt must be positive')
        if not 0<self.initial_battery<=100: raise ValueError('initial_battery must be in (0,100]')

@dataclass
class Vehicle:
    name: str
    kind: str
    node: tuple=DEPOT
    battery: float=100.
    capacity: float=2.5
    task: int|None=None
    transit: tuple|None=None
    remaining: float=0.
    distance: float=0.
    waits: int=0

@dataclass
class Order:
    id: int
    destination: tuple
    release: float
    deadline: float
    priority: int
    weight: float
    status: str='unreleased'
    delivered_at: float|None=None
    assignment: tuple|None=None
    stage: str=''
    dwell: float=0.
    completed: bool=False

class DispatchEnv:
    """reset -> (observation, info); step -> obs, reward, terminated, truncated, info.

    Actions: 0=wait; remaining fixed slots are (order, ground, UAV, mode).
    This class has no Gymnasium dependency; gym_dispatch.py supplies an adapter.
    """
    def __init__(self,config=None):
        self.config=config or DispatchConfig()
        self.action_count=1+self.config.orders*self.config.ground_count*self.config.uav_count*3
        self.finished=False
        self.reset(seed=0)

    def reset(self,*,seed=0,options=None):
        self.seed=seed;self.rng=random.Random(seed);self.time=0.;self.finished=False
        self.total_reward=0.;self.events=[];self.deliveries=0;self.on_time=0;self.reroutes=0
        self.vehicles={}
        for kind,count in (('ground',self.config.ground_count),('uav',self.config.uav_count)):
            for i in range(count):
                name=f'{kind}_{i}'
                self.vehicles[name]=Vehicle(name,kind,battery=self.config.initial_battery,
                    capacity=(2.5+.5*(i%2)) if kind=='ground' else 1.5)
        self.orders=[]
        for i in range(self.config.orders):
            release=0. if i<2 else float(self.rng.randint(5,60))
            self.orders.append(Order(i,(2,self.rng.choice((-1,0,1))),release,
                release+self.rng.randint(35,95),self.rng.randint(1,3),self.rng.choice((.5,1.,2.,3.3))))
        self.closure_plan=[]
        if self.config.closures:
            self.closure_plan=[dict(kind='ground',edge=edge((1,0),(2,0)),start=15.,end=45.),
                               dict(kind='uav',edge=edge((1,0),(2,0)),start=25.,end=65.)]
        self._closed_before=set();self.last_routes={};self._release()
        return self.observation(),self.info()

    def _release(self):
        for o in self.orders:
            if o.status=='unreleased' and o.release<=self.time:
                o.status='waiting';self.events.append(dict(event='order_released',order=o.id,time=self.time))

    def closed(self,kind):
        return {c['edge'] for c in self.closure_plan if c['kind']==kind and c['start']<=self.time<c['end']}

    def route(self,kind,start,goal):
        if start==goal:return [start]
        blocked=(self.closed('ground')|self.closed('uav')) if kind=='joint' else self.closed(kind);queue=deque([start]);prev={start:None}
        while queue:
            n=queue.popleft()
            for dx,dy in ((1,0),(0,1),(0,-1),(-1,0)):
                m=(n[0]+dx,n[1]+dy)
                if m not in NODES or m in prev or edge(n,m) in blocked:continue
                prev[m]=n
                if m==goal:
                    path=[m]
                    while prev[path[-1]] is not None:path.append(prev[path[-1]])
                    return list(reversed(path))
                queue.append(m)
        return None

    def decode(self,action):
        a=int(action)-1
        if a<0:return None
        mode=MODES[a%3];a//=3
        u=a%self.config.uav_count;a//=self.config.uav_count
        g=a%self.config.ground_count;o=a//self.config.ground_count
        return o,f'ground_{g}',f'uav_{u}',mode

    def encode(self,o,g,u,mode):
        return 1+(((o*self.config.ground_count+g)*self.config.uav_count+u)*3+MODES.index(mode))

    def estimate(self,o,g,u,mode):
        if mode=='relay' and o.weight>min(g.capacity,u.capacity):return None
        if mode=='scout' and o.weight>g.capacity:return None
        if mode=='joint' and o.weight>g.capacity+u.capacity:return None
        # Exact per-vehicle route totals for the abstract stage sequence.
        if mode=='relay':gp=[DEPOT,HUB,DEPOT];up=[DEPOT,HUB,o.destination,DEPOT]
        elif mode=='scout':gp=[DEPOT,o.destination,DEPOT];up=[DEPOT,o.destination,DEPOT]
        else:gp=[DEPOT,o.destination,DEPOT];up=gp
        def distances(kind,points):
            lengths=[]
            for a,b in zip(points,points[1:]):
                path=self.route(kind,a,b)
                if path is None:return None
                lengths.append((len(path)-1)*10.)
            return lengths
        gd,ud=distances('ground',gp),distances('uav',up)
        if mode=='joint':
            shared=self.route('joint',DEPOT,o.destination)
            if shared is None:return None
            if gd is not None:gd[0]=(len(shared)-1)*10.
            if ud is not None:ud[0]=(len(shared)-1)*10.
        if gd is None or ud is None:return None
        # Fixed cost per metre makes battery accounting transparent and testable.
        ge,ue=sum(gd)*.20,sum(ud)*.40
        if g.battery<ge+8 or u.battery<ue+8:return None
        if mode=='relay':eta=2+max(gd[0]/2.,ud[0]/5.)+2+ud[1]/5.+1
        elif mode=='scout':eta=2+ud[0]/5.+1+gd[0]/2.+1
        else:eta=2+max(gd[0]/2.,ud[0]/2.)+1
        return eta,ge,ue,sum(gd)+sum(ud)

    def observation(self):
        features=np.zeros((self.action_count,len(FEATURES)),dtype=np.float32)
        mask=np.zeros(self.action_count,dtype=np.int8);mask[0]=1;features[0,-1]=1
        for o in self.orders:
            if o.status!='waiting':continue
            for gi in range(self.config.ground_count):
                g=self.vehicles[f'ground_{gi}']
                if g.task is not None or g.node!=DEPOT or g.transit:continue
                for ui in range(self.config.uav_count):
                    u=self.vehicles[f'uav_{ui}']
                    if u.task is not None or u.node!=DEPOT or u.transit:continue
                    for mi,mode in enumerate(MODES):
                        value=self.estimate(o,g,u,mode)
                        if value is None:continue
                        eta,ge,ue,dist=value;a=self.encode(o.id,gi,ui,mode)
                        mask[a]=1
                        features[a,:9]=[o.priority/3,o.weight/4,(o.deadline-self.time)/120,eta/120,
                                        g.battery/100,u.battery/100,g.capacity/4,u.capacity/4,dist/100]
                        features[a,9+mi]=1
        global_state=np.array([self.time/self.config.horizon,
            sum(o.status=='waiting' for o in self.orders)/self.config.orders,
            self.deliveries/self.config.orders,
            sum(v.task is None for v in self.vehicles.values())/len(self.vehicles),
            len(self.closed('ground'))/12,len(self.closed('uav'))/12],dtype=np.float32)
        return dict(candidates=features,action_mask=mask,global_state=global_state)

    def _move(self,v,goal,speed):
        if v.transit is None:
            if v.node==goal:return True
            path=self.route(v.kind,v.node,goal)
            if not path:
                v.waits+=1;return False
            key=(v.name,goal,tuple(sorted(self.closed(v.kind))))
            if (len(path)-1)>abs(goal[0]-v.node[0])+abs(goal[1]-v.node[1]) and key not in self.last_routes:
                self.reroutes+=1;self.last_routes[key]=True
                self.events.append(dict(event='route_detour',agent=v.name,time=self.time,goal=goal))
            nxt=path[1];link=edge(v.node,nxt)
            # Shared edge is reserved in both directions within each vehicle class.
            if any(w.name!=v.name and w.kind==v.kind and w.transit and edge(*w.transit)==link
                   for w in self.vehicles.values()):
                v.waits+=1;return False
            v.transit=(v.node,nxt);v.remaining=10.
        rate=.20 if v.kind=='ground' else .40
        distance=min(speed*self.config.dt,v.remaining,max(0.,v.battery)/rate)
        if distance<=0:v.waits+=1;return False
        v.battery=max(0.,v.battery-distance*rate);v.distance+=distance;v.remaining-=distance
        if v.remaining<=1e-8:
            v.node=v.transit[1];v.transit=None
        return v.node==goal and v.transit is None

    def _joint_move(self,g,u,goal):
        # One shared edge and speed while carrying one parcel together.
        if g.transit is None:
            if g.node==goal:return True
            path=self.route('joint',g.node,goal)
            if not path:g.waits+=1;u.waits+=1;return False
            nxt=path[1];link=edge(g.node,nxt)
            if any(w.name not in (g.name,u.name) and w.transit and edge(*w.transit)==link
                   for w in self.vehicles.values()):
                g.waits+=1;u.waits+=1;return False
            g.transit=u.transit=(g.node,nxt);g.remaining=u.remaining=10.
        distance=min(2.*self.config.dt,g.remaining,g.battery/.20,u.battery/.40)
        if distance<=0:g.waits+=1;u.waits+=1;return False
        for v,rate in ((g,.20),(u,.40)):
            v.battery=max(0.,v.battery-distance*rate);v.distance+=distance;v.remaining-=distance
            if v.remaining<=1e-8:v.node=v.transit[1];v.transit=None
        return g.node==goal and g.transit is None

    def _task(self,o):
        _,gn,un,mode=o.assignment;g,u=self.vehicles[gn],self.vehicles[un]
        if o.stage=='pickup':
            o.dwell+=self.config.dt
            if o.dwell>=2:o.stage={'relay':'transfer','scout':'inspect','joint':'delivery'}[mode];o.dwell=0.
        elif o.stage=='transfer':
            ga=self._move(g,HUB,2.);ua=self._move(u,HUB,5.)
            if ga and ua:o.stage='handoff';o.dwell=0.
        elif o.stage=='handoff':
            o.dwell+=self.config.dt
            if o.dwell>=2:o.stage='delivery'
        elif o.stage=='inspect':
            if self._move(u,o.destination,5.):o.stage='delivery'
        elif o.stage=='delivery':
            if mode=='relay':
                self._move(g,DEPOT,2.);arrived=self._move(u,o.destination,5.)
            elif mode=='scout':
                self._move(u,DEPOT,5.);arrived=self._move(g,o.destination,2.)
            else:
                arrived=self._joint_move(g,u,o.destination)
            if arrived:o.stage='receipt';o.dwell=0.
        elif o.stage=='receipt':
            o.dwell+=self.config.dt
            if o.dwell>=1:
                o.delivered_at=self.time;o.status='delivered';o.stage='return'
                self.deliveries+=1;self.on_time+=int(self.time<=o.deadline)
                self.events.append(dict(event='delivered',order=o.id,time=self.time,priority=o.priority))
        elif o.stage=='return':
            ga=self._move(g,DEPOT,2.);ua=self._move(u,DEPOT,5.)
            if ga and ua:
                g.task=u.task=None;o.completed=True;o.stage='done'

    def step(self,action):
        if self.finished:raise RuntimeError('episode ended; call reset')
        obs=self.observation()
        if not isinstance(action,(int,np.integer)) or not 0<=action<self.action_count or not obs['action_mask'][action]:
            raise ValueError('invalid/masked action; state has not advanced')
        self.events=[]
        if action:
            oid,gn,un,mode=self.decode(action);o=self.orders[oid]
            o.assignment=(oid,gn,un,mode);o.status='active';o.stage='pickup';o.dwell=0.
            self.vehicles[gn].task=self.vehicles[un].task=oid
            self.events.append(dict(event='assigned',order=oid,ground=gn,uav=un,mode=mode,time=self.time))
        old_distance=sum(v.distance for v in self.vehicles.values());old_delivered={o.id for o in self.orders if o.delivered_at is not None}
        self.time+=self.config.dt
        closed_now={(k,e) for k in ('ground','uav') for e in self.closed(k)}
        for k,e in sorted(closed_now-self._closed_before):
            self.events.append(dict(event='edge_closed',kind=k,edge=e,time=self.time))
        for k,e in sorted(self._closed_before-closed_now):
            self.events.append(dict(event='edge_reopened',kind=k,edge=e,time=self.time))
        self._closed_before=closed_now
        self._release()
        for o in self.orders:
            if o.assignment is not None and not o.completed:self._task(o)
        for v in self.vehicles.values():
            if v.task is None and v.node==DEPOT and v.transit is None:
                v.battery=min(100.,v.battery+1.5*self.config.dt)
        reward=-.01*self.config.dt-.005*(sum(v.distance for v in self.vehicles.values())-old_distance)
        for o in self.orders:
            if o.delivered_at is not None and o.id not in old_delivered:
                reward+=10*o.priority-.1*o.priority*max(0.,o.delivered_at-o.deadline)
        terminated=all(o.completed for o in self.orders)
        truncated=self.time>=self.config.horizon and not terminated
        if truncated:reward-=sum(10*o.priority for o in self.orders if o.delivered_at is None)
        self.finished=terminated or truncated;self.total_reward+=reward
        return self.observation(),float(reward),terminated,truncated,self.info()

    def info(self):
        lateness=[max(0.,o.delivered_at-o.deadline) for o in self.orders if o.delivered_at is not None]
        priority_metrics={str(p):dict(total=sum(o.priority==p for o in self.orders),
            delivered=sum(o.priority==p and o.delivered_at is not None for o in self.orders),
            on_time=sum(o.priority==p and o.delivered_at is not None and o.delivered_at<=o.deadline for o in self.orders))
            for p in (1,2,3)}
        return dict(priority_metrics=priority_metrics,mean_lateness_delivered=float(np.mean(lateness)) if lateness else None,
            p95_lateness_delivered=float(np.percentile(lateness,95)) if lateness else None,
            seed=self.seed,time=self.time,delivered=self.deliveries,on_time=self.on_time,
            total=self.config.orders,returned=sum(o.completed for o in self.orders),
            distance=sum(v.distance for v in self.vehicles.values()),
            energy_used=sum(v.distance*(.20 if v.kind=='ground' else .40) for v in self.vehicles.values()),
            waits=sum(v.waits for v in self.vehicles.values()),route_detours=self.reroutes,reward=self.total_reward,
            events=list(self.events),backend='abstract_network_v1',perception='structured_state_no_vlm')
