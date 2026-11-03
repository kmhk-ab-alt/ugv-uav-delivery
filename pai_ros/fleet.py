"""Deterministic coordination core; no ROS, renderer, or network dependency."""
from dataclasses import dataclass, field
import math
import random
from .motion import KinematicMotion, KinematicCargo
from .safety import SafeKinematicMotion
from .scheduling import PriorityDeadlineScheduler
from .airspace import SCENARIO_ID, SCENARIO_URL, CRUISE, RETURN_ALTITUDE, areas_for_jobs, plan_outbound

PRIORITY_COLORS={1:(.12,.38,1.),2:(1.,.8,.08),3:(.95,.12,.12)}
PRIORITY_TOKENS={1:'blue',2:'yellow',3:'red'}

COLORS = {'red': (.85, .2, .2), 'green': (.2, .7, .3), 'blue': (.2, .4, .85)}
MODES = ('relay', 'scout', 'joint')
LANE_SPACING = 4.0
RECIPIENTS = {'red': 'Community Clinic', 'green': 'Fresh Market', 'blue': 'Residential Locker'}

@dataclass
class Agent:
    name: str
    kind: str
    pos: list
    task: str | None = None
    uses: int = 0
    yaw: float = 0.0
    orientation: list = field(default_factory=list)

@dataclass
class Job:
    name: str
    color: str
    source: list
    pos: list
    phase: str = 'queued'
    ground: str | None = None
    uav: str | None = None
    destination: str | None = None
    reason: str = ''
    carrier: str | None = None
    elapsed: float = 0.0
    observed: bool = False
    package_type: str = 'carton'
    contents: str = ''
    address: str = ''
    recipient: str = ''
    weight_kg: float = .4
    deadline: float = 180.0
    received_at: float | None = None
    receipt_confirmed: bool = False
    priority: int = 1
    release: float = 0.0
    mode: str | None = None
    ground_released: bool = False
    meeting_last_attempt: float = -1.
    meeting_revision: int = 0
    meeting_map: str = ''
    meeting_ground_arrival: float | None = None
    meeting_uav_arrival: float | None = None
    meeting_wait_s: float = 0.
    meeting_estimates: dict = field(default_factory=dict)
    numbered_destination: str = ''
    recognized_priority: int | None = None
    delivery_goals: dict = field(default_factory=dict)
    transfer: list = field(default_factory=list)
    uav_route: list = field(default_factory=list)
    route_phase: str = ''
    waypoint_index: int = 0
    route_history: list = field(default_factory=list)


class Fleet:
    def __init__(self, ground_count=1, uav_count=1, package_count=3, mode='relay', seed=0, randomize=False, motion=None, cargo=None, priorities=None, scheduler=None, dispatcher=None):
        for name, value in [('ground_count', ground_count), ('uav_count', uav_count), ('package_count', package_count)]:
            if type(value) is not int or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if mode not in MODES:
            raise ValueError('mode must be relay, scout, or joint')
        self.ground_speed=.9;self.uav_speed=1.3
        self.mode = mode
        self.dispatcher=dispatcher
        self.seed, self.randomize = seed, randomize
        self.cargo = cargo if cargo is not None else KinematicCargo()
        self.motion = motion if motion is not None else SafeKinematicMotion()
        self.scheduler = scheduler if scheduler is not None else PriorityDeadlineScheduler()
        if priorities is not None and (len(priorities)!=package_count or any(type(p) is not int or p not in (1,2,3) for p in priorities)):
            raise ValueError("priorities must contain one integer 1..3 per package")
        self.time = 0.0
        self.metrics={}
        self.operation_limit = 0.0
        self.events = []
        self.active_warnings = set()
        self.warning_count = 0
        self.agents = {}
        for kind, count, z in [('ground', ground_count, .25), ('uav', uav_count, .20)]:
            for i in range(count):
                name = f'{kind}_{i}'
                self.agents[name] = Agent(name, kind, [4.5, i * LANE_SPACING + (1.0 if kind=='uav' else 0.), z])
        self.jobs = {}
        order_colors=[list(COLORS)[i%3] for i in range(package_count)]
        if randomize: random.Random(seed).shuffle(order_colors)
        for i in range(package_count):
            source = [3., i * LANE_SPACING, .08]
            color = order_colors[i]
            kind, contents, weight = [('medical_case', 'Medical supplies', .6),
                                      ('grocery_crate', 'Fresh groceries', .8),
                                      ('carton', 'Household parcel', .4)][list(COLORS).index(color)]
            self.jobs[f'package_{i}'] = Job(f'package_{i}', color, source, source.copy(),
                package_type=kind, contents=contents, weight_kg=weight,
                address=f'Block {i+1:02d} / {RECIPIENTS[color]}', recipient=RECIPIENTS[color],
                deadline=180.+i*30.,priority=priorities[i] if priorities is not None else 1)

        self.restricted_areas = areas_for_jobs(self.jobs.values())
        if hasattr(self.motion,'set_restrictions'): self.motion.set_restrictions(self.restricted_areas)

    def event(self, job, name, **extra):
        self.events.append(dict(time=round(self.time, 3), package=job.name, event=name, **extra))

    def transition(self, job, phase):
        job.phase, job.elapsed = phase, 0.0
        self.event(job, phase)

    def decide(self, package, destination, reason=''):
        if package not in self.jobs:return False
        job=self.jobs[package]
        if job.phase!='observe' or job.destination is not None:return False
        if job.numbered_destination:
            priority={'blue':1,'yellow':2,'red':3}.get(destination)
            if priority is None:return False
            # Address comes from order metadata; priority color never selects an address.
            job.recognized_priority=priority;job.destination=job.numbered_destination
        else:
            if destination not in COLORS:return False
            job.destination=destination
        job.reason=str(reason)[:300]
        self.event(job,'decision',destination=job.destination,recognized_priority=job.recognized_priority,reason=job.reason)
        return True

    @staticmethod
    def correct_delivery(job):
        return job.destination==(job.numbered_destination or job.color)

    @staticmethod
    def goal(job):
        # Separate bays per task prevent stacking delivered packages.
        if job.delivery_goals:return job.delivery_goals[job.destination].copy()
        return [-3.0 - list(COLORS).index(job.destination) * .8, job.source[1], .08]

    def home(self, agent):
        if hasattr(self,'homes'):return self.homes[agent.name].copy()
        index = int(agent.name.rsplit('_', 1)[1])
        return [4.5, index * LANE_SPACING + (1.0 if agent.kind=='uav' else 0.), .25 if agent.kind == 'ground' else (.055 if self.motion.name=='gazebo_diffdrive_rotor_feedback' else .20)]

    def delivery_mode(self,job):
        city=getattr(self,'city',None)
        if getattr(city,'layout',None)=='blocks' and job.delivery_goals and all(
                p[2]<.2 and self.motion.street_network.covers(p,p) for p in job.delivery_goals.values()):
            return 'ground'
        return self.mode

    def configure_speeds(self,ground=2.4,uav=5.2):
        if any(not math.isfinite(v) or v<=0 for v in (ground,uav)):
            raise ValueError('ground_speed and uav_speed must be finite positive m/s')
        self.ground_speed=float(ground);self.uav_speed=float(uav)
        if self.motion.name=='gazebo_diffdrive_rotor_feedback':self.motion.ground_speed_limit=float(ground)
    @property
    def estimated_ground_speed(self):
        return min(self.ground_speed,getattr(self.motion,'ground_speed_limit',self.ground_speed))

    def fly(self, job, uav, points, speed, dt):
        if job.route_phase != job.phase:
            job.uav_route = [list(p) for p in points]
            job.route_phase, job.waypoint_index = job.phase, 0
            job.route_history.append(dict(phase=job.phase, points=[uav.pos.copy()]+[p.copy() for p in job.uav_route]))
            self.event(job, 'flight_plan', phase=job.phase, waypoints=job.uav_route)
        if job.waypoint_index >= len(job.uav_route): return True
        point=job.uav_route[job.waypoint_index]
        dx,dy=point[0]-uav.pos[0],point[1]-uav.pos[1]
        if abs(dx)+abs(dy)>.0001:
            desired=math.atan2(dy,dx)
            delta=(desired-uav.yaw+math.pi)%(2*math.pi)-math.pi
            uav.yaw += max(-1.5*dt,min(1.5*dt,delta))
        # Slow near a waypoint instead of turning at full speed.
        velocity=min(speed,max(.25,math.dist(uav.pos,point)*1.8))
        if self.motion.move(uav,point,velocity,dt):
            job.waypoint_index+=1
        return job.waypoint_index>=len(job.uav_route)

    def joint_route(self,start,target,lane_y):
        # A shared load must clear both ground obstacles and the restricted airspace.
        if self.motion.name!='gazebo_diffdrive_rotor_feedback':return plan_outbound(start,target,self.restricted_areas,lane_y)
        from .airspace import RestrictedArea
        pad=.19 if self.motion.name=='gazebo_diffdrive_rotor_feedback' else -.06
        ground_areas=[RestrictedArea('ground_'+o.name,o.low[0]-pad,o.high[0]+pad,o.low[1]-pad,o.high[1]+pad) for o in getattr(self.motion,'obstacles',[])]
        return plan_outbound(start,target,self.restricted_areas+ground_areas,lane_y)

    def monitor_airspace(self):
        alerts=set()
        uavs=[a for a in self.agents.values() if a.kind=='uav']
        for u in uavs:
            for area in self.restricted_areas:
                if area.contains(u.pos): alerts.add(('area',u.name,area.name))
        for i,a in enumerate(uavs):
            for b in uavs[i+1:]:
                if math.dist(a.pos,b.pos)<.65: alerts.add(('proximity',a.name,b.name))
        for alert in sorted(alerts-self.active_warnings):
            self.events.append(dict(time=round(self.time,3),event='airspace_warning',kind=alert[0],objects=list(alert[1:])))
            self.warning_count+=1
        for alert in sorted(self.active_warnings-alerts):
            self.events.append(dict(time=round(self.time,3),event='airspace_clear',kind=alert[0],objects=list(alert[1:])))
        self.active_warnings=alerts

    def tick(self, dt):
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError('dt must be positive and finite')
        self.motion.begin_step(dt,self.agents)
        self.cargo.begin_step()
        self.time += dt
        perception=getattr(self,'site_perception',None)
        if perception is not None:
            for job in self.jobs.values():
                if job.phase in ('preflight','approach') and not perception.available('pickup_'+'AB'[self.city.pickup_index(job.source)]):
                    for name in (job.ground,job.uav):
                        if name in self.agents:self.agents[name].task=None
                    self.event(job,'pickup_diverted',reason='画像認識で集荷先が利用不可。別センターの注文を再選択')
                    job.ground=job.uav=None;self.transition(job,'queued')
        selection=self.dispatcher.select(self) if self.dispatcher is not None else None
        for job in self.scheduler.rank(self.jobs.values(),self.time):
            if job.phase == 'queued':
                if hasattr(getattr(self,'city',None),'pickup_available') and not self.city.pickup_available(self,job):continue
                if self.time<job.release or (self.operation_limit and self.time>=self.operation_limit):continue
                if self.dispatcher is not None and (selection is None or job.name!=selection[0]):continue
                free = {k: sorted((a for a in self.agents.values() if a.kind == k and a.task is None),
                                  key=lambda a: (a.uses, a.name)) for k in ('ground', 'uav')}
                direct=self.delivery_mode(job)=='ground'
                prefetch=hasattr(self,'meeting') and self.mode=='relay'
                if not free['ground'] or (not direct and not free['uav'] and not prefetch):
                    continue
                g=self.agents[selection[1]] if selection is not None else free['ground'][0]
                u=None if direct else (self.agents.get(selection[2]) if selection is not None else (free['uav'][0] if free['uav'] else None))
                job.mode='ground' if direct else (selection[3] if selection is not None else self.mode)
                for a in [a for a in (g,u) if a is not None]:
                    a.task = job.name
                    a.uses += 1
                job.ground, job.uav = g.name, u.name if u else None
                self.event(job, 'assigned', ground=g.name, uav=u.name if u else None, mode=job.mode,
                           address=job.address, contents=job.contents, weight_kg=job.weight_kg, deadline=job.deadline,priority=job.priority)
                if self.dispatcher is not None:self.event(job,'dispatch_decision',**self.dispatcher.last_decision)
                self.transition(job, 'preflight')
            if job.phase in ('delivered', 'failed'):
                continue
            job.elapsed += dt
            g, u = self.agents[job.ground], self.agents.get(job.uav)
            sx, y, _ = job.source
            tx,ty,_=job.transfer or [0.,y,.08]
            if job.phase == 'preflight':
                if job.route_phase != 'approach' and u is not None and not hasattr(self,'meeting'):
                    job.uav_route=[[u.pos[0],u.pos[1],CRUISE],[sx,u.pos[1],CRUISE],[sx,y,CRUISE],[sx,y,1.8]]
                    job.route_phase,job.waypoint_index='approach',0
                    job.route_history.append(dict(phase='approach',points=[u.pos.copy()]+[p.copy() for p in job.uav_route]))
                    self.event(job,'flight_plan',phase='approach',waypoints=job.uav_route)
                if job.elapsed >= 1.0:
                    self.event(job, 'simulated_flight_authorized', scenario='ground_uav_rendezvous_v1' if hasattr(self,'meeting') else SCENARIO_ID)
                    self.transition(job, 'approach')
            elif job.phase == 'approach':
                # Transit at safe observation height before any pickup.
                ga = self.motion.move(g, [sx + .6, y, .25], self.ground_speed, dt) if job.mode != 'scout' else True
                ua = True if u is None or hasattr(self,'meeting') else self.fly(job,u,[[u.pos[0],u.pos[1],CRUISE],[sx,u.pos[1],CRUISE],[sx,y,CRUISE],[sx,y,1.8]],self.uav_speed,dt)
                if ga and ua:
                    self.transition(job, 'observe')
            elif job.phase == 'observe' and job.destination is not None:
                self.transition(job, 'ground_pickup')
            elif job.phase == 'ground_pickup':
                if self.motion.move(g, [sx + (.45 if self.motion.name=='gazebo_diffdrive_rotor_feedback' else 0.), y, .25], self.ground_speed, dt):
                    if job.carrier is None:
                        job.carrier = g.name
                        self.event(job, 'pickup', carrier=g.name)
                    if job.mode!='ground' and hasattr(self,'meeting'):
                        if u is None:
                            self.transition(job,'await_uav')
                            self.event(job,'await_uav',agent=g.name,reason='先行集荷完了。利用可能なUAVを待つ')
                            self.cargo.follow(job,g);continue
                        if self.time-job.meeting_last_attempt<1:
                            self.cargo.follow(job,g);continue
                        job.meeting_last_attempt=self.time
                        if not self.meeting.plan(self,job):
                            self.cargo.follow(job,g);continue
                    self.transition(job, {'ground':'ground_delivery','scout': 'ground_delivery', 'relay': 'to_transfer', 'joint': 'joint_lift'}[job.mode])
            elif job.phase == 'await_uav':
                available=sorted((a for a in self.agents.values() if a.kind=='uav' and a.task is None),key=lambda a:(a.uses,a.name))
                if available:
                    u=available[0];u.task=job.name;u.uses+=1;job.uav=u.name
                    self.event(job,'uav_assigned',agent=u.name,ground=g.name)
                    self.transition(job,'ground_pickup')
            elif job.phase == 'to_transfer':
                if hasattr(self,'meeting'):
                    from .rendezvous import fingerprint
                    if self.meeting.name!='fixed' and fingerprint(self)!=job.meeting_map:
                        if not self.meeting.plan(self,job,'map_changed'):
                            self.event(job,'meeting_replan_wait');continue
                        tx,ty,_=job.transfer
                ga = self.motion.move(g, [tx+(.25 if self.motion.name=='gazebo_diffdrive_rotor_feedback' else 0.), ty, .25], self.ground_speed, dt)
                if hasattr(self,'meeting'):
                    points=job.uav_route
                    if job.route_phase!=job.phase:
                        points=plan_outbound(u.pos,[tx,ty,1.8],self.restricted_areas,y)+[[tx,ty,1.8]]
                    ua=self.fly(job,u,points,self.uav_speed,dt)
                    if ga and job.meeting_ground_arrival is None:job.meeting_ground_arrival=self.time;self.event(job,'meeting_arrival',agent=g.name)
                    if ua and job.meeting_uav_arrival is None:job.meeting_uav_arrival=self.time;self.event(job,'meeting_arrival',agent=u.name)
                    if ga!=ua:job.meeting_wait_s+=dt
                else:ua = self.fly(job,u,[[sx,y,CRUISE],[sx,ty-.9,CRUISE],[tx,ty-.9,CRUISE],[tx,ty,1.8]],self.uav_speed,dt)
                if ga and ua:
                    self.transition(job, 'handoff')
            elif job.phase == 'handoff':
                # Ground must stop; UAV descends above the parcel before ownership changes.
                if self.motion.move(u, ([job.pos[0],job.pos[1],job.pos[2]+.30] if self.motion.name=='gazebo_diffdrive_rotor_feedback' else [tx,ty,.82]), .6, dt) and job.elapsed > .5:
                    job.carrier = u.name
                    self.event(job, 'handoff_complete', sender=g.name, receiver=u.name,meeting_position=job.transfer.copy(),meeting_wait_s=job.meeting_wait_s)
                    job.ground_released=True
                    if g.task==job.name:g.task=None
                    self.event(job,'ground_released',agent=g.name,reason='受け渡し完了。次の集荷を受け付ける')
                    self.transition(job, 'uav_lift')
            elif job.phase == 'uav_lift':
                if self.fly(job,u,[[tx,ty,CRUISE]],.8,dt):
                    self.transition(job, 'uav_delivery')
            elif job.phase == 'uav_delivery':
                target = self.goal(job)
                points=plan_outbound(u.pos,target,self.restricted_areas,y) if job.route_phase != job.phase else job.uav_route
                if self.fly(job,u,points,self.uav_speed,dt):
                    self.transition(job, 'uav_drop')
            elif job.phase == 'uav_drop':
                target = self.goal(job)
                if self.fly(job,u,[[target[0],target[1],target[2]+1.17]],.6,dt):
                    self.transition(job, 'lower_package')
            elif job.phase == 'lower_package':
                target=self.goal(job)
                # Visual winch lowering, not a cable dynamics simulation.
                if self.cargo.lower(job,target,dt):
                    self.deliver(job)
            elif job.phase == 'joint_lift':
                if self.fly(job,u,[[g.pos[0]+(.45*math.cos(g.yaw) if self.motion.name=='gazebo_diffdrive_rotor_feedback' else 0),g.pos[1]+(.45*math.sin(g.yaw) if self.motion.name=='gazebo_diffdrive_rotor_feedback' else 0),CRUISE]],.8,dt):
                    self.transition(job, 'joint_delivery')
            elif job.phase in ('ground_delivery', 'joint_delivery'):
                target = self.goal(job)
                if job.mode != 'joint':
                    arrived = self.motion.move(g, [target[0]+(.45 if self.motion.name=='gazebo_diffdrive_rotor_feedback' else 0.), target[1], .25], self.ground_speed, dt)
                else:
                    if job.route_phase != job.phase:
                        job.uav_route=self.joint_route(g.pos,target,y)
                        job.route_phase,job.waypoint_index=job.phase,0
                        job.route_history.append(dict(phase=job.phase,points=[p.copy() for p in job.uav_route]))
                        self.event(job,'flight_plan',phase=job.phase,waypoints=job.uav_route)
                    p=job.uav_route[min(job.waypoint_index,len(job.uav_route)-1)]
                    arrived=self.motion.move(g,[p[0]+(.45 if self.motion.name=='gazebo_diffdrive_rotor_feedback' and job.waypoint_index>=len(job.uav_route)-1 else 0),p[1],.25],self.ground_speed,dt)
                    if arrived and job.waypoint_index<len(job.uav_route): job.waypoint_index+=1
                    arrived=arrived and job.waypoint_index>=len(job.uav_route)
                if job.mode == 'joint':
                    # Kinematic cooperative support: same horizontal load trajectory.
                    # Shared-load experiment follows ground route; not PODIUM's baseline flight.
                    synchronized=self.motion.move(u,[g.pos[0]+(.45*math.cos(g.yaw) if self.motion.name=='gazebo_diffdrive_rotor_feedback' else 0),g.pos[1]+(.45*math.sin(g.yaw) if self.motion.name=='gazebo_diffdrive_rotor_feedback' else 0),CRUISE],self.uav_speed,dt)
                    arrived=arrived and synchronized
                    if self.motion.name!='gazebo_diffdrive_rotor_feedback' or synchronized or '+' in (job.carrier or ''):job.carrier = g.name + '+' + u.name
                if arrived and (self.motion.name!='gazebo_diffdrive_rotor_feedback' or self.motion.face(g,math.pi)):
                    self.deliver(job)
            elif job.phase == 'receiving':
                if self.motion.name=='gazebo_diffdrive_rotor_feedback' and math.dist(job.pos,self.goal(job))>.35:
                    if job.elapsed>6:
                        self.event(job,'physical_delivery_missed',position=job.pos.copy(),target=self.goal(job))
                        self.transition(job,'failed')
                    continue
                if job.elapsed >= 1.2:
                    job.receipt_confirmed = True
                    job.received_at = self.time
                    self.event(job, 'receipt_confirmed', address=job.address,
                               recipient=RECIPIENTS.get(job.destination,job.recipient), correct=self.correct_delivery(job),
                               on_time=self.time <= job.deadline)
                    self.transition(job, 'return_clear')
            elif job.phase == 'return_clear':
                return_goal=self.home(g) if hasattr(self.motion,'street_network') else [self.home(g)[0],y,.25]
                ga = True if job.ground_released else self.motion.move(g, return_goal, self.ground_speed, dt)
                ua = True if u is None else self.fly(job,u,[[u.pos[0],u.pos[1],RETURN_ALTITUDE]],.8,dt)
                if ga and ua:
                    self.transition(job, 'return_home')
            elif job.phase == 'return_home':
                gh, uh = self.home(g), self.home(u) if u else None
                ga = True if job.ground_released else self.motion.move(g, gh, self.ground_speed, dt)
                ua = True if u is None else self.fly(job,u,[[u.pos[0],y-.9,RETURN_ALTITUDE],[uh[0],y-.9,RETURN_ALTITUDE],[uh[0],uh[1],RETURN_ALTITUDE]],self.uav_speed,dt)
                if ga and ua:
                    self.transition(job, 'dock')
            elif job.phase == 'dock':
                landed=True if u is None else self.fly(job,u,[self.home(u)],.6,dt)
                if landed and (u is None or self.motion.name!='gazebo_diffdrive_rotor_feedback' or u.pos[2]<.065):
                    self.transition(job, 'delivered')
                    for agent in [a for a in (g,u) if a is not None]:
                        if agent.task==job.name:agent.task = None
                    self.event(job, 'agents_available')
            if job.phase not in ('delivered','lower_package') and job.carrier:
                owner = u if u is not None and job.carrier == u.name else g
                self.cargo.follow(job,owner)

        # Released UGVs are independent of the old UAV mission. If no next job
        # was selected, return them without issuing commands from that old job.
        self.ground_returning=[]
        for agent in self.agents.values():
            if agent.kind=='ground' and agent.task is None:
                if not self.motion.move(agent,self.home(agent),self.ground_speed,dt):self.ground_returning.append(agent.name)
        self.motion.end_step(dt,self.agents)
        for event in getattr(self.motion,'events',[]):
            self.events.append(dict(time=round(self.time,3),**event))
        self.monitor_airspace()

    def deliver(self, job):
        self.cargo.place(job,self.goal(job))
        job.carrier = None
        self.transition(job, 'receiving')
        self.event(job, 'delivery_checked', correct=self.correct_delivery(job))

    @property
    def done(self):
        if getattr(self,'ground_returning',[]):return False
        if self.operation_limit and self.time>=self.operation_limit:
            return all(j.phase in ('queued','delivered') for j in self.jobs.values())
        return all(j.phase == 'delivered' for j in self.jobs.values())

    def observation(self):
        """Nonprivileged high-level policy input; no true color/recipient/address."""
        return dict(time=self.time,
            agents={a.name:dict(kind=a.kind,position=a.pos.copy(),task=a.task) for a in self.agents.values()},
            jobs={j.name:dict(position=j.pos.copy(),phase=j.phase,destination=j.destination,priority=j.priority,deadline=j.deadline)
                  for j in self.jobs.values()})

    def snapshot(self):
        from dataclasses import asdict
        from .service_budget import summary
        return dict(ground_returning=getattr(self,'ground_returning',[]),mode=self.mode, seed=self.seed, randomized=self.randomize, movement_speeds=dict(ground_m_s=self.estimated_ground_speed,uav_m_s=self.uav_speed), control_backend=self.motion.name,scheduler=self.scheduler.name,
                    collision_holds=getattr(self.motion,'hold_count',0),obstacle_detours=getattr(self.motion,'detour_count',0),scenario='ground_uav_rendezvous_v1' if hasattr(self,'meeting') else SCENARIO_ID, scenario_source=self.metrics['rendezvous_scenario']['reference'] if hasattr(self,'meeting') else SCENARIO_URL,
                    site_perception=getattr(self,'site_perception',None).snapshot() if getattr(self,'site_perception',None) else None,site_blockers=getattr(self,'site_survey',None).blockers() if getattr(self,'site_survey',None) else {},city=getattr(self,'city_manifest',None),metrics=self.metrics, dispatch_policy=getattr(self.dispatcher,'name','priority_then_deadline'),
                    warning_count=self.warning_count, active_warnings=[list(a) for a in sorted(self.active_warnings)],
                    time=self.time, complete=self.done,ground_routes=getattr(self.motion,'route_preview',{}),
                    service_budget=summary(self),
                    dispatch_decision=getattr(self.dispatcher,'last_decision',None),
                    agents={k: asdict(a) for k, a in self.agents.items()},
                    packages={k: asdict(j) for k, j in self.jobs.items()})
