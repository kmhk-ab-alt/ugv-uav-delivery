"""VLM worker; callbacks remain responsive during network requests."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from .interfaces import Observation
from .planners import VLMReactionPlanner

class ReactionPlanner(Node):
    def __init__(self):
        super().__init__('pai_planner')
        for name, value in [('stub', False), ('model', 'qwen2.5vl:3b'),
                            ('base_url', 'http://localhost:11434/v1')]:
            self.declare_parameter(name, value)
        self.stub = self.get_parameter('stub').value
        self.planner = VLMReactionPlanner(
            [x.strip() for x in self.get_parameter('model').value.split(',') if x.strip()],
            base_url=self.get_parameter('base_url').value,
            api_key=os.environ.get('OPENROUTER_API_KEY', 'ollama'))
        self.pub = self.create_publisher(String, 'reaction', 10)
        self.create_subscription(Image, 'camera/image_raw', self.observe, 10)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.round = None
        self.completed_round = 0
        self.create_timer(0.05, self.poll)

    def observe(self, msg):
        try:
            parts = msg.header.frame_id.split(':', 1)
            round_id = int(parts[0])
            if round_id <= self.completed_round or self.pending is not None:
                return
            if msg.encoding != 'rgb8' or msg.step < msg.width * 3:
                raise ValueError('expected rgb8')
            raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
            pixels = raw[:, :msg.width * 3].reshape(msg.height, msg.width, 3).copy()
            self.round = round_id
            if self.stub:
                from .interfaces import Command
                action = 'smash' if len(parts) == 2 and parts[1] == 'barrier' else 'dodge'
                self.pending = self.pool.submit(lambda: Command(action, 'stub'))
            else:
                self.pending = self.pool.submit(self.planner.decide, Observation(image=pixels))
        except (ValueError, TypeError):
            self.get_logger().warning('Ignored malformed image')

    def poll(self):
        if self.pending is None or not self.pending.done():
            return
        try:
            cmd = self.pending.result()
            payload = dict(round=self.round, action=cmd.trick, reason=cmd.reason)
        except Exception as exc:
            self.get_logger().error(str(exc))
            payload = dict(round=self.round, action='dodge', reason='planner error')
        self.pub.publish(String(data=json.dumps(payload)))
        self.completed_round = self.round
        self.pending = None

    def destroy_node(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ReactionPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
