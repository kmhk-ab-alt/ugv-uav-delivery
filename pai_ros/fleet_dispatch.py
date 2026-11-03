"""Native 3D dispatch adapter for the shared 19-input candidate scorer.
Same feature schema as abstract training, with explicitly estimated 3D ETAs.
No parcel color, hidden destination, or future disruption is a policy input.
"""
import hashlib
import math
from pathlib import Path
import numpy as np
from .dispatch_env import FEATURES,MODES
from .dispatch_model import ScoringPolicy
from .dispatch_policy import PriorityDeadlinePolicy,EarliestDeadlinePolicy,RandomPolicy

class FleetDispatch:
    def __init__(self,policy='priority',checkpoint='',adaptive_modes=False,seed=0):
        self.adaptive_modes=adaptive_modes;self.name=policy;self.checkpoint_sha256=None;self.last_decision=None
        if policy=='learned':
            if not checkpoint:raise ValueError('learned policy requires --checkpoint')
            self.policy=ScoringPolicy.load(checkpoint);self.checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        elif policy=='priority':self.policy=PriorityDeadlinePolicy()
        elif policy=='deadline':self.policy=EarliestDeadlinePolicy()
        elif policy=='random':self.policy=RandomPolicy(seed)
        else:raise ValueError('unknown dispatch policy')
    def observation(self,fleet):
        jobs=list(fleet.jobs.values());gs=[a for a in fleet.agents.values() if a.kind=='ground'];us=[a for a in fleet.agents.values() if a.kind=='uav']
        size=1+len(jobs)*len(gs)*len(us)*3
        f=np.zeros((size,len(FEATURES)),dtype=np.float32);mask=np.zeros(size,dtype=np.int8);mask[0]=1;f[0,-1]=1;actions={}
        for oi,j in enumerate(jobs):
            if j.phase!='queued' or fleet.time<j.release:continue
            for gi,g in enumerate(gs):
                if g.task is not None:continue
                for ui,u in enumerate(us):
                    if u.task is not None:continue
                    for mi,mode in enumerate(MODES):
                        if not self.adaptive_modes and mode!=fleet.mode:continue
                        if mode=='relay' and j.weight_kg>1.5:continue
                        if mode=='scout' and j.weight_kg>2.5:continue
                        if mode=='joint' and j.weight_kg>4:continue
                        approach=math.dist(g.pos,[j.source[0]+.6,j.source[1],.25])/.9
                        flight=math.dist(u.pos,[j.source[0],j.source[1],1.8])/1.0+8
                        # Routing label is deliberately unknown until camera inference.
                        last_mile=7.6
                        closure_delay=8*int(bool(getattr(fleet,'disruption_active',False)))
                        eta=max(approach,flight)+10+last_mile/(1.1 if mode=='relay' else .75)+closure_delay
                        dist=math.dist(g.pos,j.source)+math.dist(u.pos,j.source)+2*last_mile+6.4
                        if hasattr(fleet,'city'):
                            from .city_eta import estimate
                            travel,route_distance=estimate(fleet,j,mode)
                            eta=max(approach,flight)+travel+closure_delay
                            dist=math.dist(g.pos,j.source)+math.dist(u.pos,j.source)+route_distance
                        a=1+(((oi*len(gs)+gi)*len(us)+ui)*3+mi);mask[a]=1;actions[a]=(j.name,g.name,u.name,mode)
                        f[a,:9]=[j.priority/3,j.weight_kg/4,(j.deadline-fleet.time)/120,eta/120,1,1,2.5/4,1.5/4,dist/100];f[a,9+mi]=1
        active=sum(a.task is None for a in fleet.agents.values())/len(fleet.agents)
        state=np.array([fleet.time/240,sum(j.phase=='queued' and fleet.time>=j.release for j in jobs)/len(jobs),sum(j.phase=='delivered' for j in jobs)/len(jobs),active,int(bool(getattr(fleet,'disruption_active',False)))/12,0],dtype=np.float32)
        return dict(candidates=f,action_mask=mask,global_state=state),actions
    def select(self,fleet):
        obs,actions=self.observation(fleet)
        if not actions:return None
        action=self.policy.select(obs)
        self.last_decision=dict(policy=self.name,checkpoint_sha256=self.checkpoint_sha256,adapter='fleet_3d_v1',action=action,
            selected=actions.get(action),features=obs['candidates'][action].tolist(),global_state=obs['global_state'].tolist())
        return actions.get(action)
