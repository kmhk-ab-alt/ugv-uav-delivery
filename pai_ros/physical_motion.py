"""Feedback waypoint tracking. Positions come only from Gazebo odometry.
Safety produces waypoint/hold decisions; commands are body-frame velocities.
"""
import math
from .safety import SafeKinematicMotion

def wrap(angle):return (angle+math.pi)%(2*math.pi)-math.pi

def track(kind,position,yaw,target,speed):
    dx,dy,dz=[b-a for a,b in zip(position,target)]
    distance=math.hypot(dx,dy)
    if kind=='ground':
        error=wrap(math.atan2(dy,dx)-yaw) if distance>.025 else 0
        forward=min(speed,.8*distance,math.sqrt(2*.35*max(0.,distance-.025)))*max(0.,math.cos(error)) if abs(error)<.65 else 0.
        return [forward,0.,0.,max(-1.6,min(1.6,2.5*error))]
    vx,vy,vz=.7*dx,.7*dy,1.2*dz
    norm=math.sqrt(vx*vx+vy*vy+vz*vz)
    limit=min(speed,math.sqrt(2*.5*math.sqrt(dx*dx+dy*dy+dz*dz)))
    if norm>limit:vx,vy,vz=[v*limit/norm for v in (vx,vy,vz)]
    return [math.cos(yaw)*vx+math.sin(yaw)*vy,-math.sin(yaw)*vx+math.cos(yaw)*vy,vz,0.]

class PhysicalMotion(SafeKinematicMotion):
    name='gazebo_diffdrive_rotor_feedback'
    ground_speed_limit=.5
    def __init__(self):super().__init__();self.velocity_commands={};self.measured_yaw={};self.hold_targets={};self.integral_z={};self.safety_holds={};self.last_positions={};self.ground_velocity={};self.measured_speed={}
    def plan(self,start,target,kind):
        # Leave a tracking margin beyond the collision margin at grid corners.
        from .safety import Obstacle
        original=self.obstacles
        self.obstacles=[Obstacle(o.name,tuple(v-(.25 if i<2 else 0) for i,v in enumerate(o.low)),tuple(v+(.25 if i<2 else 0) for i,v in enumerate(o.high))) for o in original]
        try:return super().plan(start,target,kind)
        finally:self.obstacles=original
    def begin_step(self,dt,agents):
        for a in agents.values():
            previous=self.last_positions.get(a.name)
            self.measured_speed[a.name]=math.dist(a.pos,previous)/dt if previous is not None and dt>0 else 0.
            if a.kind=='ground':
                self.ground_velocity[a.name]=math.dist(a.pos[:2],previous[:2])/dt if previous is not None and dt>0 else 0.
            self.last_positions[a.name]=a.pos.copy()
        super().begin_step(dt,agents)
        self.velocity_commands={}
        for a in agents.values():
            if a.name not in getattr(self,'payloads',set()):self.integral_z[a.name]=0.
            target=self.hold_targets.get(a.name,a.pos)
            self.velocity_commands[a.name]=track(a.kind,a.pos,self.measured_yaw.get(a.name,a.yaw),target,.5) if a.kind=='uav' else [0.,0.,0.,0.]
            if a.kind=='uav':self.velocity_commands[a.name][2]+=self.integral_z.get(a.name,0)
    def move(self,agent,target,speed,dt):
        if agent.kind=='ground':speed=min(speed,self.ground_speed_limit)
        actual=agent.pos.copy();tolerance=.10 if agent.kind=='ground' else (.025 if target[2]<.1 else .12)
        distance=math.dist(actual,target)
        settled=(self.ground_velocity if agent.kind=='ground' else self.measured_speed).get(agent.name,0.)<.12
        if distance<tolerance and not settled:
            self.hold_targets[agent.name]=list(target)
            self.velocity_commands[agent.name]=[0.,0.,0.,0.] if agent.kind=='ground' else track(agent.kind,actual,self.measured_yaw.get(agent.name,agent.yaw),target,.3)
            if agent.kind=='uav':self.velocity_commands[agent.name][2]+=self.integral_z.get(agent.name,0.)
            self.commands.append(dict(agent=agent.name,command_type='brake_for_waypoint',reason='settle before arrival'))
            return False
        if distance<tolerance:
            self.safety_holds.pop(agent.name,None)
            self.hold_targets[agent.name]=list(target)
            self.velocity_commands[agent.name]=track(agent.kind,actual,self.measured_yaw.get(agent.name,agent.yaw),target,.4) if agent.kind=='uav' else [0.,0.,0.,0.]
            if agent.kind=='uav':self.velocity_commands[agent.name][2]+=self.integral_z.get(agent.name,0.)
            return True
        # Physical tracking converges within a tolerance, not bit-exact grid points.
        for key,path in self.paths.items():
            if key[0]==agent.name:
                while path and math.dist(actual,path[0])<tolerance:
                    if agent.kind=='ground' and not settled:
                        self.velocity_commands[agent.name]=[0.,0.,0.,0.]
                        self.commands.append(dict(agent=agent.name,command_type='brake_for_waypoint',reason='settle before turning'))
                        return False
                    path.pop(0)
        if agent.name in self.escape and math.dist(actual,self.escape[agent.name])<tolerance:self.escape.pop(agent.name)
        n=len(self.commands)
        # Do not cut a corner when the final goal first becomes visible. Finish the
        # current street waypoint before turning; the chassis has braking inertia.
        path=self.paths.get((agent.name,tuple(target)))
        steering_target=path[0] if agent.kind=='ground' and path else target
        super().move(agent,steering_target,speed,max(.25,dt))
        agent.pos[:]=actual
        generated=self.commands[n:]
        waypoint=next((c['target'] for c in reversed(generated) if c['command_type']=='position_target'),None)
        if waypoint is None:
            hold=self.safety_holds.setdefault(agent.name,actual.copy())
            target_hold=[actual[0],actual[1],hold[2]]
            self.hold_targets[agent.name]=target_hold
            velocity=track(agent.kind,actual,self.measured_yaw.get(agent.name,agent.yaw),target_hold,speed)
        else:
            self.safety_holds.pop(agent.name,None)
            self.hold_targets[agent.name]=list(waypoint)
            velocity=track(agent.kind,actual,self.measured_yaw.get(agent.name,agent.yaw),waypoint,speed)
        if agent.kind=='uav':
            error=(waypoint if waypoint is not None else target_hold)[2]-actual[2]
            loaded=agent.name in getattr(self,'payloads',set())
            integral=self.integral_z.get(agent.name,0)
            if waypoint is None or abs(error)<.5:integral+=error*dt*.7
            self.integral_z[agent.name]=max(0. if loaded else -.7,min(.7,integral))
            velocity[2]=max(-speed-.7,min(speed+.7,velocity[2]+self.integral_z[agent.name]))
        self.velocity_commands[agent.name]=velocity
        return False
    def face(self,agent,yaw):
        error=wrap(yaw-self.measured_yaw.get(agent.name,agent.yaw))
        self.velocity_commands[agent.name]=[0.,0.,0.,max(-1.2,min(1.2,2*error))]
        return abs(error)<.12
    def end_step(self,dt,agents):
        self.commands += [dict(agent=name,command_type='body_twist',linear=v[:3],yaw_rate=v[3],unit='m/s',yaw_unit='rad/s',feedback='gazebo_odometry') for name,v in self.velocity_commands.items()]
