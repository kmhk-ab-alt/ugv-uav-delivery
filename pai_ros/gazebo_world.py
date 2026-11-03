"""ROS 2 / ros_gz adapter for pose visualization or physical feedback control.
Gazebo owns simulation time and renders images; no synthetic camera fallback.
"""
import math
import threading
import time
import numpy as np
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool,String
from ros_gz_interfaces.srv import SetEntityPose
from rclpy.qos import qos_profile_sensor_data
from .safety import Obstacle

class GazeboWorld:
    viewer=None
    def __init__(self,fleet,ros_node,render=True,physical=False,lockstep=False):
        self.lockstep=lockstep;self.step_sent=0.;self.step_target=None;self.startup_feedback_retries=0
        if lockstep and not physical:raise ValueError('lockstep requires physical mode')
        self.fleet=fleet; self.node=ros_node; self.lock=threading.Lock(); self.pending=[]
        self.client=ros_node.create_client(SetEntityPose,'/world/pai_delivery/set_pose')
        if not self.client.wait_for_service(timeout_sec=45):raise RuntimeError('Gazebo pose service not available')
        self.cargo_pub=self.node.create_publisher(String,'/pai/cargo/command',10) if physical else None
        self.parcel_odom={};self.measured_distance={};self.previous_position={}
        self.last_visual_pose={};self.physical=physical;self.active_since={};self.odom={};self.twist_pubs={};self.enable_pubs={}
        if physical:
            for name,a in fleet.agents.items():
                self.node.create_subscription(Odometry,f'/model/{name}/odometry',lambda msg,n=name:self.odometry(n,msg),qos_profile_sensor_data)
                topic=f'/model/{name}/cmd_vel' if a.kind=='ground' else f'/{name}/cmd_vel'
                self.twist_pubs[name]=self.node.create_publisher(Twist,topic,10)
                if a.kind=='uav':self.enable_pubs[name]=self.node.create_publisher(Bool,f'/{name}/enable',10)
        if physical:
            for name in fleet.jobs:self.node.create_subscription(Odometry,f'/model/{name}/odometry',lambda msg,n=name:self.parcel_odometry(n,msg),qos_profile_sensor_data)
        self.images={};self.image_stamps={}; self.now=None; self.last=None
        self.node.create_subscription(Clock,'/clock',self.clock,qos_profile_sensor_data)
        if render:
            names=list(fleet.jobs)
            if hasattr(fleet,'city') and getattr(fleet.city,'layout','')=='blocks':
                from .site_perception import sites_for
                names += ['site_'+name for name in sites_for(fleet.city)]
            for name in names:
                self.node.create_subscription(Image,f'/pai_gz/{name}/image',lambda msg,n=name:self.camera(n,msg),qos_profile_sensor_data)
        # Same obstacle geometry used by the existing kinematic safety layer.
        # Keep the tested obstacle extractor independent of MuJoCo.
        obstacles=[]
        for i,j in enumerate(fleet.jobs.values()):
            from .fleet import COLORS
            x=-3.-list(COLORS).index(j.color)*.8; y=j.source[1]+1.7
            obstacles.append(Obstacle(f'building_{i}',(x-.74,y-.7,0),(x+.74,y+.7,2.3)))
        if fleet.jobs:
            y=next(iter(fleet.jobs.values())).source[1]
            obstacles.append(Obstacle('trolley',(1.3,y-.2,.04),(1.8,y+.2,.60)))
        if hasattr(fleet,'city'):obstacles=list(fleet.city.obstacles)
        if hasattr(fleet,'rendezvous_obstacles'):obstacles+=fleet.rendezvous_obstacles
        if physical:obstacles=[Obstacle(o.name,tuple(x-.2 for x in o.low),tuple(x+.2 for x in o.high)) for o in obstacles]
        if hasattr(fleet.motion,'set_obstacles'): fleet.motion.set_obstacles(obstacles)
        if lockstep:
            from gz.transport13 import Node as GzNode
            self.step_node=GzNode()
            self.advance()
    def advance(self):
        from gz.msgs10.world_control_pb2 import WorldControl
        from gz.msgs10.boolean_pb2 import Boolean
        ok,response=self.step_node.request('/world/pai_delivery/control',WorldControl(pause=True,multi_step=25),WorldControl,Boolean,5000)
        if not ok or not response.data:raise RuntimeError('Gazebo rejected lockstep request')
        self.step_sent=time.monotonic()
        self.step_target=None if self.now is None else self.now+.05

    def parcel_odometry(self,name,msg):
        p=msg.pose.pose.position
        self.fleet.jobs[name].pos[:]=[p.x,p.y,p.z]
        self.parcel_odom[name]=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
    def odometry(self,name,msg):
        p=msg.pose.pose.position;q=msg.pose.pose.orientation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        position=[p.x,p.y,p.z]
        self.fleet.agents[name].orientation=[q.x,q.y,q.z,q.w]
        if name in self.previous_position:self.measured_distance[name]=self.measured_distance.get(name,0)+math.dist(position,self.previous_position[name])
        self.previous_position[name]=position
        self.fleet.metrics['measured_distance_m']=dict(self.measured_distance)
        self.odom[name]=(position,yaw,msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9)
    def clock(self,msg):
        with self.lock: self.now=msg.clock.sec+msg.clock.nanosec*1e-9
    def camera(self,name,msg):
        if msg.width!=192 or msg.height!=192 or msg.encoding not in ('rgb8','RGB8'):
            return
        data=np.frombuffer(bytes(msg.data),dtype=np.uint8).reshape(msg.height,msg.step)
        pixels=data[:,:msg.width*3].reshape(msg.height,msg.width,3).copy()
        with self.lock:
            self.images[name]=pixels
            self.image_stamps[name]=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
    def step_time(self):
        with self.lock: clock_now=self.now
        if self.lockstep:
            if clock_now is None:
                if time.monotonic()-self.step_sent>2:self.advance()
                return 0
            if self.last is None:
                # Bootstrap only; no task commands have been issued yet.
                self.last=clock_now;self.advance();return 0
            if self.fleet.time==0 and (len(self.odom)!=len(self.fleet.agents) or len(self.parcel_odom)!=len(self.fleet.jobs)):
                if (self.step_target is not None and clock_now>=self.step_target-1e-6) or time.monotonic()-self.step_sent>2:
                    self.last=clock_now;self.advance()
                return 0
            waiting=(self.step_target is not None and clock_now<self.step_target-1e-6) or clock_now<=self.last or len(self.odom)!=len(self.fleet.agents) or any(clock_now-v[2]>.035 for v in self.odom.values())
            if waiting:
                waited=time.monotonic()-self.step_sent
                # During camera bootstrap only, no vehicle has a task or motion
                # command. A dropped final feedback sample otherwise deadlocks
                # a paused simulator: it cannot publish a newer sample until stepped.
                idle_startup=self.fleet.time==0 and all(a.task is None for a in self.fleet.agents.values())
                batch_finished=self.step_target is not None and clock_now>=self.step_target-1e-6
                retries=getattr(self,'startup_feedback_retries',0)
                if idle_startup and batch_finished and waited>2 and retries<3:
                    self.startup_feedback_retries=retries+1
                    self.last=clock_now;self.advance();return 0
                if waited>30:
                    stale={n:round(clock_now-v[2],4) for n,v in self.odom.items() if clock_now-v[2]>.035}
                    missing=sorted(set(self.fleet.agents)-set(self.odom))
                    raise RuntimeError(f'Gazebo lockstep feedback timeout; simulation remains paused (clock={clock_now}, target={self.step_target}, last={self.last}, missing={missing}, stale_seconds={stale}, startup_retries={retries})')
                return 0
        if self.pending and self.physical:
            completed=[f for f in self.pending if f.done()]
            if any(not f.result().success for f in completed):raise RuntimeError('Gazebo rejected visual marker pose')
            self.pending=[f for f in self.pending if not f.done()]
        elif self.pending:
            if not all(f.done() for f in self.pending):
                if time.monotonic()-self.sent_at>10:raise RuntimeError('Gazebo pose response timeout')
                return 0
            if not all(f.result().success for f in self.pending):raise RuntimeError('Gazebo rejected pose command')
            self.pending=[]
        if clock_now is None or (self.last is not None and clock_now<=self.last):return 0
        if self.physical:
            self.fleet.motion.payloads={name for j in self.fleet.jobs.values() if j.carrier for name in j.carrier.split('+')}
            if len(self.odom)!=len(self.fleet.agents):return 0
            if len(self.parcel_odom)!=len(self.fleet.jobs):return 0
            if any(j.carrier and clock_now-self.parcel_odom[j.name]>1 for j in self.fleet.jobs.values()):raise RuntimeError('Stale carried-payload odometry')
            for name,(position,yaw,received) in self.odom.items():
                agent=self.fleet.agents[name]
                active=agent.task is not None and self.fleet.jobs[agent.task].phase!='preflight'
                if hasattr(self.fleet,'meeting') and agent.kind=='uav' and agent.task is not None and self.fleet.jobs[agent.task].phase in ('approach','observe','ground_pickup'):active=False
                if active:self.active_since.setdefault(name,clock_now)
                else:self.active_since.pop(name,None)
                if active and clock_now-received>1 and clock_now-self.active_since[name]>1:raise RuntimeError('Stale Gazebo odometry: '+name)
                self.fleet.agents[name].pos[:]=position
                self.fleet.agents[name].yaw=yaw
                self.fleet.motion.measured_yaw[name]=yaw
        with self.lock: now=self.now
        if now is None: return 0
        if self.last is None: self.last=now; return 0
        dt=now-self.last
        if dt<=0: return 0
        # Consume at most one integration step; retain backlog rather than jump over obstacles.
        if self.physical and dt>.5:raise RuntimeError('Physics controller stalled for more than 0.5 simulation seconds')
        dt=dt if self.physical else min(.02,dt)
        self.last=now if self.physical else self.last+dt
        return dt
    def sync(self):
        self.sent_at=time.monotonic()
        if self.physical:
            for name,v in self.fleet.motion.velocity_commands.items():
                msg=Twist();msg.linear.x,msg.linear.y,msg.linear.z=v[:3];msg.angular.z=v[3]
                self.twist_pubs[name].publish(msg)
            for name,pub in self.enable_pubs.items():
                a=self.fleet.agents[name];j=self.fleet.jobs.get(a.task)
                enabled=j is not None and not (j.phase=='dock' and a.pos[2]<.08)
                if hasattr(self.fleet,'meeting') and j is not None and j.phase in ('preflight','approach','observe','ground_pickup'):enabled=False
                pub.publish(Bool(data=enabled))
        if self.physical:
            for command in self.fleet.cargo.commands:
                if command['command_type']=='physical_grip':
                    text=f"attach {command['package']} {command['carrier']} {command['length']} {command['mass']}"
                elif command['command_type']=='physical_release':text=f"detach {command['package']}"
                else:continue
                self.cargo_pub.publish(String(data=text))
        entities=[] if self.physical else list(self.fleet.agents.values())+list(self.fleet.jobs.values())
        if hasattr(self.fleet,'meeting'):
            from types import SimpleNamespace
            entities += [SimpleNamespace(name='meeting_'+j.name,pos=[j.transfer[0],j.transfer[1],.025 if j.meeting_revision and not j.receipt_confirmed else -1.]) for j in self.fleet.jobs.values() if not self.physical or j.meeting_revision]
        scenario=getattr(self,'scenario',None)
        if scenario is not None:
            from types import SimpleNamespace
            entities += [SimpleNamespace(name='temporary_roadwork',pos=scenario.visual_position),SimpleNamespace(name='temporary_airspace',pos=scenario.air_position)]
        survey=getattr(self.fleet,'site_survey',None)
        if survey is not None:
            from types import SimpleNamespace
            entities += [SimpleNamespace(name='site_blocker_'+name,pos=pos) for name,pos in survey.blockers().items()]
        for entity in entities:
            pose_key=tuple(entity.pos)+(getattr(entity,'yaw',0),)
            if self.physical and self.last_visual_pose.get(entity.name)==pose_key:continue
            if self.physical:self.last_visual_pose[entity.name]=pose_key
            request=SetEntityPose.Request();request.entity.name=entity.name;request.entity.type=2
            pose=request.pose
            pose.position.x,pose.position.y,pose.position.z=entity.pos
            yaw=getattr(entity,'yaw',0);pose.orientation.z=math.sin(yaw/2);pose.orientation.w=math.cos(yaw/2)
            self.pending.append(self.client.call_async(request))
        if self.lockstep:self.advance()
    def image(self,package):
        with self.lock:
            pixels=self.images.get(package)
            return None if pixels is None else pixels.copy()
    def close(self):
        for pub in self.twist_pubs.values():pub.publish(Twist())
        for pub in self.enable_pubs.values():pub.publish(Bool(data=False))

