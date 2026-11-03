"""Single owner of MuJoCo state. Physics never waits on VLM inference."""
import json
import random
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock as RclClock, ClockType
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from .mujoco_backend import MujocoWorld, MujocoGo2Controller, MujocoCameraPerception, VARIANTS
from .go2_tricks import STAND, TRICKS
from .protocol import decode_command

class Simulation(Node):
    def __init__(self):
        super().__init__('pai_sim')
        for name, value in [('menagerie_dir', ''), ('headless', True), ('stub', False),
                            ('rounds', 8), ('decision_timeout', 45.0)]:
            self.declare_parameter(name, value)
        get = lambda name: self.get_parameter(name).value
        self.world = MujocoWorld(get('menagerie_dir'), headless=get('headless'))
        self.controller = MujocoGo2Controller(self.world)
        self.camera = None if get('stub') else MujocoCameraPerception(self.world)
        self.stub = get('stub')
        self.rounds = get('rounds')
        self.timeout = get('decision_timeout')
        self.images = self.create_publisher(Image, 'camera/image_raw', 10)
        self.joints = self.create_publisher(JointState, 'joint_states', 10)
        self.status = self.create_publisher(String, 'status', 10)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.create_subscription(String, 'reaction', self.on_command, 10)
        self.round = 0
        self.phase = 'settle'
        self.phase_start = 0.0
        self.command = None
        self.launched = False
        self.frames = 0
        self.done = False
        self.deadline = 0.0
        self.variant = ''
        self.timer = self.create_timer(float(self.world.model.opt.timestep), self.tick,
                                      clock=RclClock(clock_type=ClockType.STEADY_TIME))

    def report(self, event, **extra):
        payload = json.dumps(dict(round=self.round, event=event, **extra))
        self.status.publish(String(data=payload))
        self.get_logger().info(payload)

    def on_command(self, msg):
        if self.phase != 'approach' or self.command is not None:
            return
        try:
            self.command = decode_command(msg.data, self.round)
        except (ValueError, TypeError, AttributeError):
            self.get_logger().warning('Ignored invalid or stale reaction')

    def spawn(self):
        if self.round >= self.rounds:
            self.report('complete')
            self.done = True
            return
        self.round += 1
        self.variant = random.choice(VARIANTS)
        self.world.set_obstacle(self.variant)
        self.command = None
        self.launched = False
        self.phase = 'approach'
        self.deadline = time.monotonic() + self.timeout
        self.report('spawn')

    def image(self):
        msg = Image()
        # Round identifier ties each observation to its reaction, including delayed results.
        msg.header.frame_id = str(self.round)
        msg.header.stamp = self.stamp()
        if self.camera is None:
            msg.height = msg.width = 1
            msg.data = [0, 0, 0]
            # Stub-only fixture. The VLM never receives ground-truth labels.
            msg.header.frame_id += ':' + self.variant
        else:
            pixels = np.ascontiguousarray(self.camera.observe().image, dtype=np.uint8)
            msg.height, msg.width = pixels.shape[:2]
            msg.data = pixels.tobytes()
        msg.encoding = 'rgb8'
        msg.step = msg.width * 3
        self.images.publish(msg)

    def stamp(self):
        from builtin_interfaces.msg import Time
        ns = int(self.world.data.time * 1e9)
        return Time(sec=ns // 1000000000, nanosec=ns % 1000000000)

    def tick(self):
        if not self.world.running():
            self.done = True
        if self.done:
            return
        now = float(self.world.data.time)
        target = STAND
        if self.phase == 'settle' and now - self.phase_start >= 1.0:
            self.spawn()
        if self.done:
            return
        if self.phase == 'approach':
            # Repeat observations so late-starting subscribers can recover.
            if self.frames % 50 == 0:
                self.image()
            near = self.world.obstacle_x() <= 0.48
            self.world.drive_obstacle(0.0 if near else -0.6)
            if self.command is None and time.monotonic() >= self.deadline:
                self.command = ('dodge', 'decision timeout')
            if near and self.command is not None:
                self.phase = 'react'
                self.phase_start = now
                self.report('reaction', action=self.command[0], reason=self.command[1])
        if self.phase == 'react':
            elapsed = now - self.phase_start
            if elapsed >= 0.9:
                self.phase = 'settle'
                self.phase_start = now
            else:
                action = self.command[0]
                target = TRICKS[action](elapsed, 0.9)
                if action == 'smash' and not self.launched and elapsed >= 0.09:
                    self.world.launch_obstacle(4.5, 2.5)
                    self.launched = True
        self.controller._pd(target)
        if not self.world.step():
            self.done = True
        self.clock_pub.publish(Clock(clock=self.stamp()))
        if self.frames % 10 == 0:
            msg = JointState()
            msg.header.stamp = self.stamp()
            msg.name = [self.world.model.joint(i).name for i in range(1, 13)]
            msg.position = self.world.data.qpos[7:19].tolist()
            msg.velocity = self.world.data.qvel[6:18].tolist()
            msg.effort = self.world.data.ctrl.tolist()
            self.joints.publish(msg)
        self.frames += 1

    def destroy_node(self):
        if self.camera is not None:
            self.camera._renderer.close()
        self.world.close()
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Simulation()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
