"""Ground-to-UAV relay with map-generated rendezvous sites (no road vehicle).
Selection happens after inspection; the UAV waits at home until pickup completes.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np
from .safety import Obstacle,SafeKinematicMotion
from .airspace import CRUISE,plan_outbound
from .dispatch_model import ScoringPolicy

FEATURES=('ground_eta','uav_eta','waiting','delivery_flight','delivery_eta','receipt_eta','distance','deadline_slack','priority','weight','source_dx','source_dy','fixed_site')
SCENARIO_URL='https://newsroom.kddi.com/news/detail/kddi_nr-356_3625.html'

def path_length(points):return sum(math.dist(a,b) for a,b in zip(points,points[1:]))

def fingerprint(fleet):
    return hashlib.sha256(repr((getattr(getattr(fleet,'site_perception',None),'version',0),[(o.name,o.low,o.high) for o in fleet.motion.obstacles],[(a.name,a.xmin,a.xmax,a.ymin,a.ymax) for a in fleet.restricted_areas])).encode()).hexdigest()

class MeetingNetwork(ScoringPolicy):
    def save(self,path,metadata=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,**self.params,features=np.array(FEATURES),metadata=np.array(json.dumps(dict(adapter='rendezvous_v1',**(metadata or {})))))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if tuple(d['features'].tolist())!=FEATURES:raise ValueError('This is not a rendezvous checkpoint')
            if json.loads(str(d['metadata'])).get('adapter')!='rendezvous_v1':raise ValueError('wrong rendezvous adapter')
            if d['w1'].ndim!=2 or d['w1'].shape[0]!=19:raise ValueError('wrong rendezvous input shape')
            model=cls(hidden=d['w1'].shape[1]);model.params={k:d[k].copy() for k in model.params}
        h=model.params['w1'].shape[1]
        if model.params['b1'].shape!=(h,) or model.params['w2'].shape!=(h,) or not all(np.isfinite(v).all() for v in model.params.values()):raise ValueError('invalid rendezvous weights')
        return model

class Rendezvous:
    def __init__(self,policy='eta',checkpoint='',model=None):
        if policy not in ('fixed','eta','learned'):raise ValueError('invalid meeting policy')
        self.name=policy;self.network=model;self.checkpoint_sha256=None
        if policy=='learned' and model is None:
            if not checkpoint:raise ValueError('learned meeting policy needs a checkpoint')
            self.network=MeetingNetwork.load(checkpoint);self.checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.maps={};self.collect=False;self.examples=[]
    def candidates(self,fleet,job):
        if job.destination is None:raise ValueError('inspect parcel before planning a meeting')
        fixed=getattr(fleet,'fixed_meetings',{}).get(job.name,job.transfer or [0.,job.source[1],.08])
        points=[('fixed',list(fixed))];goal=fleet.goal(job)
        if getattr(getattr(fleet,'city',None),'layout','lanes')=='blocks':
            return [('fixed' if math.dist(p,fixed)<.01 else f'meeting_{i}',p.copy())
                    for i,p in enumerate(fleet.city.meeting_sites) if getattr(fleet,'site_perception',None) is None or not fleet.site_perception.pending]
        for fraction in (.15,.35,.55,.75,.9):
            x=job.source[0]+fraction*(goal[0]-job.source[0])
            for offset in (0.,-1.1,1.1):points.append((f'site_{fraction}_{offset}',[x,job.source[1]+offset,.08]))
        # Include an already agreed site when a live disruption changes the map.
        if job.meeting_revision and not any(math.dist(job.transfer,p)<.01 for _,p in points):points.append(('current',job.transfer.copy()))
        return points
    def observation(self,fleet,job):
        if job.destination is None:raise ValueError('inspect parcel before planning a meeting')
        g=fleet.agents[job.ground];u=fleet.agents[job.uav];goal=fleet.goal(job)
        physical=fleet.motion.name=='gazebo_diffdrive_rotor_feedback'
        gs=fleet.estimated_ground_speed;offset=.25 if physical else 0.
        rows=[];records=[];geometry=fingerprint(fleet)
        for site,point in self.candidates(fleet,job):
            # Both bodies need clearance. A physical tracking margin is included in ground routing.
            target=[point[0]+offset,point[1],.25]
            if any(a.contains([point[0],point[1],.82],.50) for a in fleet.restricted_areas):continue
            if any(j.name!=job.name and (j.phase in ('to_transfer','handoff','uav_lift','uav_delivery','uav_drop','lower_package','receiving') or (j.phase=='return_clear' and j.ground in fleet.agents and math.dist(fleet.agents[j.ground].pos[:2],point[:2])<1.1)) and j.meeting_revision and math.dist(j.transfer,point)<1.1 for j in fleet.jobs.values()):continue
            key=(geometry,tuple(round(v,2) for v in g.pos),tuple(target))
            if key not in self.maps:
                self.maps[key]=fleet.motion.plan(g.pos,target,'ground')
            path=self.maps[key]
            if path is None:continue
            ground=path_length([g.pos]+path)
            try:
                aerial=plan_outbound(u.pos,point,fleet.restricted_areas,job.source[1])
                delivery=plan_outbound(point,goal,fleet.restricted_areas,job.source[1])
            except ValueError:continue
            if any(not fleet.motion.static_clear(a,b,'uav') for a,b in zip(aerial,aerial[1:])):continue
            if not fleet.motion.static_clear([point[0],point[1],CRUISE],[point[0],point[1],.82],'uav'):continue
            flight=path_length(aerial);last_mile=path_length(delivery)
            ge=ground/gs;ue=abs(CRUISE-u.pos[2])/.8+flight/fleet.uav_speed+(CRUISE-1.8)/.8
            wait=abs(ge-ue);completion=max(ge,ue)+last_mile/fleet.uav_speed+12
            distance=ground+flight+last_mile;slack=job.deadline-fleet.time-completion
            cost=completion+.25*wait+.03*distance+2*job.priority*max(0,-slack)
            rows.append([ge/120,ue/120,wait/60,last_mile/20,last_mile/fleet.uav_speed/120,completion/180,distance/60,slack/120,job.priority/3,job.weight_kg/1.5,(point[0]-job.source[0])/10,(point[1]-job.source[1])/5,int(site=='fixed')])
            records.append(dict(site=site,position=point,ground_eta_s=ge,uav_eta_s=ue,waiting_estimate_s=wait,receipt_eta_s=completion,distance_estimate_m=distance,cost=cost))
        if not rows:return None,[]
        obs=dict(candidates=np.array(rows,dtype=np.float32),action_mask=np.ones(len(rows),dtype=np.int8),global_state=np.array([fleet.time/300,job.priority/3,job.weight_kg/1.5,int(getattr(fleet,'disruption_active',False)),(job.deadline-fleet.time)/180,1],dtype=np.float32))
        return obs,records
    def plan(self,fleet,job,reason='pickup'):
        obs,records=self.observation(fleet,job)
        if not records:
            fleet.event(job,'meeting_unavailable',reason=reason);return False
        if self.name=='fixed':
            choices=[i for i,r in enumerate(records) if r['site']=='fixed']
            if not choices:return False
            selected=choices[0]
        elif self.name=='eta':selected=min(range(len(records)),key=lambda i:records[i]['cost'])
        else:selected=self.network.select(obs)
        if self.collect:self.examples.append((obs,selected))
        r=records[selected];previous=job.transfer.copy();job.transfer=r['position'].copy();job.meeting_revision+=1
        job.meeting_ground_arrival=None;job.meeting_uav_arrival=None;job.meeting_map=fingerprint(fleet)
        job.route_phase='';job.meeting_estimates=r
        fleet.event(job,'meeting_selected',policy=self.name,checkpoint_sha256=self.checkpoint_sha256,revision=job.meeting_revision,reason=reason,candidate_count=len(records),previous_position=previous,site_changed=math.dist(previous,job.transfer)>.01,**r)
        return True

def configure(fleet,policy='eta',checkpoint='',model=None):
    if fleet.mode!='relay' or (fleet.dispatcher is not None and fleet.dispatcher.adaptive_modes):raise ValueError('rendezvous requires relay mode without adaptive_modes')
    fleet.meeting=Rendezvous(policy,checkpoint,model);fleet.fixed_meetings={j.name:(j.transfer or [0.,j.source[1],.08]).copy() for j in fleet.jobs.values()}
    for j in fleet.jobs.values():j.transfer=fleet.fixed_meetings[j.name].copy()
    # Launch apron is separate from the cross-lane ground traffic.
    if not hasattr(fleet,'homes'):fleet.homes={a.name:fleet.home(a) for a in fleet.agents.values()}
    for a in fleet.agents.values():
        if a.kind=='uav':
            a.pos[0]+=1.2;fleet.homes[a.name][0]+=1.2
    # A road interruption stands in for a river / inaccessible final ground segment.
    # It is a bounded simulation obstacle, not a reconstruction of the KDDI site.
    ys=[j.source[1] for j in fleet.jobs.values()]
    urban=getattr(getattr(fleet,'city',None),'layout','lanes')=='blocks'
    barriers=[] if urban else [Obstacle('relay_ground_barrier',(-1.85,min(ys)-4,0),(-1.55,max(ys)+4,.65))]
    fleet.rendezvous_obstacles=barriers;fleet.motion.set_obstacles(list(fleet.motion.obstacles)+barriers)
    fleet.metrics['rendezvous_scenario']=dict(id='ground_uav_rendezvous_v1',reference=SCENARIO_URL,adaptation='ground robot and UAV only; no road vehicle',meeting_policy=policy,ground_barrier=(dict(low=barriers[0].low,high=barriers[0].high) if barriers else None))
