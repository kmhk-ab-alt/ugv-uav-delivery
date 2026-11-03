"""Replaceable motion boundary. Physical controllers must implement this contract.

The shipped backend directly integrates positions; it is explicitly NOT physics.
"""
from typing import Protocol
import math

class MotionBackend(Protocol):
    name: str
    commands: list
    def begin_step(self, dt, agents): ...
    def move(self, agent, target, speed, dt) -> bool: ...
    def end_step(self, dt, agents): ...

class KinematicMotion:
    name='kinematic_demo'
    def __init__(self): self.commands=[]
    def begin_step(self, dt, agents): self.commands=[]
    def move(self, agent, target, speed, dt):
        self.commands.append(dict(agent=agent.name,target=list(target),speed_limit=speed,
                                  command_type='position_target',unit='m',speed_unit='m/s'))
        distance=math.dist(agent.pos,target)
        if distance<=speed*dt:
            agent.pos[:]=target
            return True
        for i in range(3): agent.pos[i]+=(target[i]-agent.pos[i])*speed*dt/distance
        return False
    def end_step(self, dt, agents): pass

class CargoBackend(Protocol):
    commands: list
    def begin_step(self): ...
    def follow(self, job, owner): ...
    def lower(self, job, target, dt) -> bool: ...
    def place(self, job, target): ...

class KinematicCargo:
    """Attachment/lowering visualization; no contact or tension model."""
    def __init__(self): self.commands=[]
    def begin_step(self): self.commands=[]
    def follow(self, job, owner):
        target=[owner.pos[0],owner.pos[1],owner.pos[2]-.30 if owner.kind=='uav' else .54]
        self.commands.append(dict(command_type='cargo_follow',package=job.name,carrier=owner.name,target=target))
        job.pos[:]=target
    def lower(self, job, target, dt):
        self.commands.append(dict(command_type='cargo_lower',package=job.name,target=list(target),speed_limit=.28))
        job.pos[2]=max(target[2],job.pos[2]-.28*dt)
        return job.pos[2]<=target[2]+1e-8
    def place(self, job, target):
        self.commands.append(dict(command_type='cargo_place',package=job.name,target=list(target)))
        job.pos[:]=target
