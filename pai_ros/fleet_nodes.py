"""ROS 2 bridge for fleet state, images, agent poses, and VLM decisions."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import time
import uuid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock as RclClock, ClockType
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from .fleet import Fleet
from .recording import EpisodeRecorder
from .scheduling import parse_priorities
from .cargo_vlm import offline_decide, vlm_decide, api_key_for, DEFAULT_MODEL, DEFAULT_BASE_URL

DEFAULTS = dict(scene_perception='off',site_events=True,speed_scale=1.0,ground_speed=2.4,uav_speed=5.2,lockstep=False,city_layout='lanes',operation_limit=0.0,budget_policy='learned',budget_checkpoint='',meeting_policy='off',meeting_checkpoint='',city_seed=-1,city_family='train',local_map=False,pixel_test=False,dispatch_policy='',checkpoint='',adaptive_modes=False,disruptions=False,physical=False, backend='mujoco', ground_count=1, uav_count=1, package_count=3, mode='relay',
                headless=True, stub=False, seed=0, randomize=False, record_dir='', priorities='', model=DEFAULT_MODEL,
                base_url=DEFAULT_BASE_URL, mission_timeout=600.0)

class FleetSimulation(Node):
    def __init__(self):
        super().__init__('fleet_sim')
        for k, v in DEFAULTS.items():
            self.declare_parameter(k, v)
        get = lambda k: self.get_parameter(k).value
        from .physical_motion import PhysicalMotion
        from .fleet_dispatch import FleetDispatch
        dispatcher=FleetDispatch(get('dispatch_policy'),get('checkpoint'),get('adaptive_modes'),get('seed')) if get('dispatch_policy') else None
        self.fleet = Fleet(get('ground_count'), get('uav_count'), get('package_count'), get('mode'), seed=get('seed'), randomize=get('randomize'),priorities=parse_priorities(get('priorities'),get('package_count')),motion=PhysicalMotion() if get('physical') else None,dispatcher=dispatcher)
        self.fleet.configure_speeds(get('ground_speed')*get('speed_scale'),get('uav_speed')*get('speed_scale'))
        if get('city_seed')>=0:
            from .city import generate_city
            generate_city(get('city_seed'),get('package_count'),get('city_family'),get('city_layout')).apply(self.fleet)
            if get('randomize'):
                from .city import configure_orders
                configure_orders(self.fleet,get('seed'))
                if get('priorities'):
                    for j,priority in zip(self.fleet.jobs.values(),parse_priorities(get('priorities'),get('package_count'))):j.priority=priority
        if get('operation_limit'):
            if get('mode')!='relay' or get('adaptive_modes'):raise ValueError('service budget requires relay without adaptive modes')
            from .service_budget import configure,orders,BudgetDispatch
            configure(self.fleet,get('operation_limit'));orders(self.fleet,get('seed'))
            if get('priorities'):
                for j,p in zip(self.fleet.jobs.values(),parse_priorities(get('priorities'),get('package_count'))):j.priority=p
            budget_checkpoint=get('budget_checkpoint')
            if get('budget_policy')=='learned' and not budget_checkpoint:
                from ament_index_python.packages import get_package_share_directory
                from pathlib import Path
                budget_checkpoint=str(Path(get_package_share_directory('pai_ros'))/('checkpoints/site_dispatch/best.npz' if get('scene_perception')!='off' else 'checkpoints/budget_blocks/best.npz' if get('city_layout')=='blocks' else 'checkpoints/budget/best.npz'))
            self.fleet.dispatcher=BudgetDispatch(get('budget_policy'),budget_checkpoint,seed=get('seed'))
        if get('physical'):
            from .physical_cargo import PhysicalCargo
            self.fleet.cargo=PhysicalCargo()
        self.gazebo = get('backend') == 'gazebo'
        if get('backend') not in ('mujoco', 'gazebo'):
            raise ValueError('backend must be mujoco or gazebo')
        if get('meeting_policy')!='off':
            from .rendezvous import configure
            configure(self.fleet,get('meeting_policy'),get('meeting_checkpoint'))
        if self.gazebo:
            from .gazebo_world import GazeboWorld
            self.world = GazeboWorld(self.fleet, self, render=not get('stub') or get('scene_perception')=='vlm',physical=get('physical'),lockstep=get('lockstep'))
        else:
            from .fleet_world import FleetWorld
            self.world = FleetWorld(self.fleet, get('headless'), render=not get('stub'))
        from .live_scenario import LiveScenario
        self.scenario=LiveScenario(self.fleet) if get('disruptions') else None
        if self.scenario is not None:self.world.scenario=self.scenario
        self.mapper=None
        if get('local_map'):
            if not hasattr(self.fleet,'city'):raise ValueError('local_map requires city_seed')
            from .city import LocalObstacleMap
            self.mapper=LocalObstacleMap(self.fleet)
        self.stub = get('stub')
        self.site_survey=None;self.survey_status_time=0.
        if get('scene_perception')!='off':
            if get('backend')!='gazebo' or get('city_layout')!='blocks':raise ValueError('Site perception requires the Gazebo blocks scene')
            from .site_perception import SiteSurvey
            mode=get('scene_perception');key=api_key_for(get('base_url')) if mode=='vlm' else None
            self.site_survey=SiteSurvey(self.fleet,mode,[x.strip() for x in get('model').split(',') if x.strip()],get('base_url'),key,get('site_events'))
            self.fleet.site_survey=self.site_survey
        config={k:get(k) for k in DEFAULTS if k!='record_dir'}
        self.recorder=EpisodeRecorder(get('record_dir'),config) if get('record_dir') else None
        self.timeout = get('mission_timeout')
        if self.timeout <= 0:
            raise ValueError('mission_timeout must be positive')
        self.session = uuid.uuid4().hex
        self.started = time.monotonic()
        self.frames = 0
        self.finished = False
        self.failed = False
        self.errors = {}
        self.state_pub = self.create_publisher(String, 'fleet/state', 10)
        self.event_pub = self.create_publisher(String, 'fleet/events', 100)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.image_pubs = {j: self.create_publisher(Image, f'{j}/camera/image_raw', 10) for j in self.fleet.jobs}
        self.pose_pubs = {a: self.create_publisher(PoseStamped, f'{a}/pose', 10) for a in self.fleet.agents}
        self.create_subscription(String, 'fleet/decisions', self.decide, 100)
        self.timer = self.create_timer(.02, self.tick,
                                      clock=RclClock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(f'mode={self.fleet.mode}, ground={get("ground_count")}, uav={get("uav_count")}')

    def decide(self, msg):
        try:
            d = json.loads(msg.data)
            if not isinstance(d, dict) or d.get('session') != self.session:
                return
            package = d.get('package')
            if not isinstance(package, str) or package not in self.fleet.jobs:
                return
            if d.get('error'):
                self.errors[package] = str(d['error'])[:300]
            else:
                self.fleet.decide(package, d.get('destination'), d.get('reason', ''))
        except (ValueError, TypeError):
            self.get_logger().warning('Ignored invalid fleet decision')

    def stamp(self):
        from builtin_interfaces.msg import Time
        ns = int(self.fleet.time * 1e9)
        return Time(sec=ns // 1000000000, nanosec=ns % 1000000000)

    def observe(self, j):
        if self.stub:
            # Explicit fixture; no renderer, network, or learned model.
            from .fleet import COLORS,PRIORITY_COLORS
            rgb=PRIORITY_COLORS[j.priority] if j.numbered_destination else COLORS[j.color]
            pixels = np.full((32, 32, 3), np.array(rgb) * 255, dtype=np.uint8)
        else:
            pixels = self.world.image(j.name)
        if pixels is None:
            return  # Wait for a real Gazebo camera frame; never invent an image.
        if self.recorder: self.recorder.image(self.fleet,j.name,pixels)
        msg = Image()
        msg.header.frame_id = f'{self.session}/{j.name}/camera'
        msg.header.stamp = self.stamp()
        msg.height, msg.width = pixels.shape[:2]
        msg.encoding, msg.step = 'rgb8', pixels.shape[1] * 3
        msg.data = pixels.tobytes()
        self.image_pubs[j.name].publish(msg)
        if not j.observed:
            j.observed = True
            self.fleet.event(j, 'inspection', observer=j.uav if (j.mode or self.fleet.mode) in ('scout', 'joint') else j.ground)

    def publish_state(self):
        state = self.fleet.snapshot()
        state.update(session=self.session, errors=self.errors, failed=self.failed,presentation=dict(stub=self.stub,physical=self.gazebo and self.world.physical))
        self.state_pub.publish(String(data=json.dumps(state)))
        for a in self.fleet.agents.values():
            msg = PoseStamped()
            msg.header.frame_id, msg.header.stamp = 'map', self.stamp()
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = a.pos
            msg.pose.orientation.z = math.sin(a.yaw/2)
            msg.pose.orientation.w = math.cos(a.yaw/2)
            self.pose_pubs[a.name].publish(msg)

    def tick(self):
        if self.finished:
            return
        if self.world.viewer and not self.world.viewer.is_running():
            self.failed, self.finished = True, True
        elif time.monotonic() - self.started > self.timeout:
            self.failed, self.finished = True, True
            self.get_logger().error('Mission timeout: unresolved cargo remains in place')
        else:
            if self.site_survey is not None:
                try:ready=self.site_survey.poll(self.world)
                except Exception as exc:
                    self.errors['site_perception']=str(exc);self.failed=self.finished=True;ready=False
                if not ready:
                    # No mission progression while awaiting an image decision.
                    # When cameras need a fresh frame, advance only holding physics.
                    if self.site_survey.future is None and not self.finished:
                        dt=self.world.step_time()
                        if dt>0:
                            self.fleet.motion.begin_step(dt,self.fleet.agents);self.fleet.cargo.begin_step()
                            for agent in self.fleet.agents.values():
                                self.fleet.motion.move(agent,agent.pos.copy(),.4,dt)
                            self.world.sync()
                    if time.monotonic()-self.survey_status_time>.3:
                        self.publish_state();self.survey_status_time=time.monotonic()
                    if self.finished:self.get_logger().error('Site recognition failed; no guessed availability: '+str(self.errors))
                    return
            if self.scenario is not None:self.scenario.tick(self.fleet)
            if self.mapper is not None:self.mapper.tick(self.fleet)
            before=self.fleet.observation()
            dt = self.world.step_time() if self.gazebo else .02
            if dt <= 0:
                return
            self.fleet.tick(dt)
            if self.recorder: self.recorder.step(self.fleet,before)
            self.world.sync()
            if self.frames % 25 == 0:
                for j in self.fleet.jobs.values():
                    if j.phase == 'observe' and j.destination is None and j.name not in self.errors:
                        self.observe(j)
            if self.fleet.done:
                self.failed = any(j.receipt_confirmed and not self.fleet.correct_delivery(j) for j in self.fleet.jobs.values())
                self.finished = True
            # Stop on a decision failure; never silently substitute a destination.
            for j in self.fleet.jobs.values():
                if j.phase=='failed':self.errors[j.name]='physical delivery missed receiving zone'
            if self.errors:
                self.failed, self.finished = True, True
                self.get_logger().error('Mission failed: ' + json.dumps(self.errors))
        for event in self.fleet.events:
            payload = json.dumps(event)
            self.event_pub.publish(String(data=payload))
            self.get_logger().info(payload)
        self.fleet.events.clear()
        if not self.gazebo:
            self.clock_pub.publish(Clock(clock=self.stamp()))
        if self.frames % (2 if self.get_parameter('lockstep').value else 5) == 0 or self.finished:
            self.publish_state()
        if self.finished:
            result = dict(event='mission_finished', success=not self.failed,
                          delivered=sum(j.phase == 'delivered' for j in self.fleet.jobs.values()),
                          total=len(self.fleet.jobs),service_budget=__import__('pai_ros.service_budget',fromlist=['summary']).summary(self.fleet))
            self.event_pub.publish(String(data=json.dumps(result)))
            self.get_logger().info(json.dumps(result))
        self.frames += 1

    def destroy_node(self):
        if self.site_survey is not None:self.site_survey.close()
        self.world.close()
        if self.recorder: self.recorder.close(self.fleet,self.errors)
        return super().destroy_node()

class CargoPlanner(Node):
    def __init__(self):
        super().__init__('cargo_planner')
        for k, v in DEFAULTS.items():
            self.declare_parameter(k, v)
        self.stub = self.get_parameter('stub').value
        self.priority_labels=self.get_parameter('city_layout').value=='blocks'
        count = self.get_parameter('package_count').value
        if type(count) is not int or count < 1:
            raise ValueError('package_count must be positive')
        self.models = [x.strip() for x in self.get_parameter('model').value.split(',') if x.strip()]
        self.url = self.get_parameter('base_url').value
        self.pixel_test=self.get_parameter('pixel_test').value
        self.api_key = None if self.stub or self.pixel_test else api_key_for(self.url)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = {}
        self.cache = {}
        self.pub = self.create_publisher(String, 'fleet/decisions', 100)
        self.subs = [self.create_subscription(Image, f'package_{i}/camera/image_raw', self.observe, 10)
                     for i in range(count)]
        self.create_timer(.05, self.poll)

    def observe(self, msg):
        try:
            session, package, _ = msg.header.frame_id.split('/')
            key = (session, package)
            if key in self.cache:
                self.pub.publish(String(data=json.dumps(self.cache[key])))
                return
            if key in self.pending:
                return
            if msg.encoding != 'rgb8' or msg.step < msg.width * 3 or not msg.height or not msg.width:
                raise ValueError('expected nonempty rgb8 image')
            pixels = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
            pixels = pixels[:, :msg.width*3].reshape(msg.height, msg.width, 3).copy()
            if self.priority_labels:
                from .cargo_vlm import priority_decide
                future=self.pool.submit(priority_decide,pixels,self.models,self.url,self.api_key,offline=self.stub or self.pixel_test)
            elif self.stub or self.pixel_test:
                future = self.pool.submit(offline_decide, pixels)
            else:
                future = self.pool.submit(vlm_decide, pixels, self.models, self.url,
                                          self.api_key)
            self.pending[key] = future
        except (TypeError, ValueError):
            self.get_logger().warning('Ignored malformed camera message')

    def poll(self):
        for key, future in list(self.pending.items()):
            if not future.done():
                continue
            result = dict(session=key[0], package=key[1])
            try:
                destination, reason = future.result()
                result.update(destination=destination, reason=reason)
            except Exception as exc:
                result['error'] = str(exc)[:300]
            self.cache[key] = result
            self.pub.publish(String(data=json.dumps(result)))
            del self.pending[key]

    def destroy_node(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

def run(node_type, args=None):
    rclpy.init(args=args)
    node = None
    failed = False
    try:
        node = node_type()
        while rclpy.ok() and not getattr(node, 'finished', False):
            rclpy.spin_once(node, timeout_sec=.1)
        failed = getattr(node, 'failed', False)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        failed=True
        if node is not None:
            node.failed=True
            if hasattr(node,'errors'):node.errors['runtime']=str(exc)
        raise
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if failed:
        raise SystemExit(1)

def sim_main(args=None):
    run(FleetSimulation, args)

def planner_main(args=None):
    run(CargoPlanner, args)
