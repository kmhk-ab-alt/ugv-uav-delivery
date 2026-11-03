"""Compliant gripper commands; payload state is measured in Gazebo.
No payload position integration or per-frame teleportation.
"""
class PhysicalCargo:
    def __init__(self):self.commands=[];self.owners={};self.lengths={}
    def begin_step(self):self.commands=[]
    def follow(self,job,owner):
        carrier=job.carrier if job.carrier and '+' in job.carrier else owner.name
        if self.owners.get(job.name)!=carrier:self.lengths[job.name]=2.68 if '+' in carrier else .3
        self.owners[job.name]=carrier
        self.commands.append(dict(command_type='physical_grip',package=job.name,carrier=carrier,length=self.lengths[job.name],mass=job.weight_kg))
    def lower(self,job,target,dt):
        name=job.name;self.lengths[name]=min(1.45,self.lengths.get(name,.3)+.28*dt)
        self.commands.append(dict(command_type='physical_grip',package=name,carrier=self.owners[name],length=self.lengths[name],mass=job.weight_kg))
        return job.pos[2]<=target[2]+.055
    def place(self,job,target):
        self.commands.append(dict(command_type='physical_release',package=job.name));self.owners.pop(job.name,None)
