"""Priority-weighted receipts shared by training, ROS and presentation.
Every correct confirmed receipt earns points, regardless of receipt time.
The operation limit stops new assignments; active work continues to completion.
"""
import hashlib,json,math,random
from pathlib import Path
import numpy as np
from .dispatch_model import ScoringPolicy

POINTS={1:1,2:3,3:9}
FEATURES=('priority','value','receipt_eta','cycle_eta','budget_left','deadline_left','budget_slack','deadline_slack','ground_approach','uav_approach','released_age','queue_order','feasible')

def configure(fleet,limit):
    if not math.isfinite(limit) or limit<=0:raise ValueError('operation limit must be finite and positive')
    fleet.operation_limit=float(limit)
    fleet.metrics['service_objective']=dict(limit_s=limit,points=POINTS,rule='correct confirmed receipt earns priority points regardless of receipt time; operation limit stops new assignments')

def orders(fleet,seed):
    """Mixed queue independent of visual labels; same arrivals in every policy."""
    rng=random.Random(seed)
    for i,j in enumerate(fleet.jobs.values()):
        j.priority=rng.choice((1,1,2,2,3))
        j.release=0. if i%4!=3 else round(rng.uniform(20,60),1)
        j.deadline=round(fleet.operation_limit*rng.uniform(.65,1.),1)

def summary(fleet):
    limit=getattr(fleet,'operation_limit',0.)
    if not limit:return None
    jobs=list(fleet.jobs.values())
    def earned(j):return j.receipt_confirmed and j.destination==(j.numbered_destination or j.color) and j.received_at is not None
    rows=[]
    for p in (3,2,1):
        js=[j for j in jobs if j.priority==p]
        rows.append(dict(priority=p,points=POINTS[p],total=len(js),scored=sum(earned(j) for j in js),received=sum(j.receipt_confirmed for j in js)))
    return dict(limit_s=limit,remaining_s=max(0,limit-fleet.time),expired=fleet.time>=limit,score=sum(POINTS[j.priority] for j in jobs if earned(j)),maximum_score=sum(POINTS[j.priority] for j in jobs),scored=sum(earned(j) for j in jobs),received=sum(j.receipt_confirmed for j in jobs),unstarted=sum(j.phase=='queued' for j in jobs),by_priority=rows)

class BudgetNetwork(ScoringPolicy):
    def save(self,path,metadata=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,**self.params,features=np.array(FEATURES),metadata=np.array(json.dumps(dict(adapter='budget_dispatch_v1',**(metadata or {})))))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if tuple(d['features'].tolist())!=FEATURES or json.loads(str(d['metadata'])).get('adapter')!='budget_dispatch_v1':raise ValueError('wrong budget checkpoint schema')
            if d['w1'].ndim!=2 or d['w1'].shape[0]!=19:raise ValueError('wrong budget input shape')
            m=cls(hidden=d['w1'].shape[1]);m.params={k:d[k].copy() for k in m.params}
        h=m.params['w1'].shape[1]
        if m.params['b1'].shape!=(h,) or m.params['w2'].shape!=(h,) or not all(np.isfinite(v).all() for v in m.params.values()):raise ValueError('invalid budget weights')
        return m

class BudgetDispatch:
    adaptive_modes=False
    def __init__(self,policy='learned',checkpoint='',model=None,seed=0):
        if policy not in ('learned','untrained','value','priority','fifo','random'):raise ValueError('invalid budget policy')
        self.name='budget_'+policy;self.policy=policy;self.network=model;self.last_decision=None;self.rng=random.Random(seed);self.checkpoint_sha256=None;self.route_cache={}
        if policy=='untrained':self.network=BudgetNetwork(42)
        if policy=='learned' and model is None:
            self.network=BudgetNetwork.load(checkpoint);self.checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    def observation(self,f):
        gs=[a for a in f.agents.values() if a.kind=='ground' and a.task is None];us=[a for a in f.agents.values() if a.kind=='uav' and a.task is None]
        rows=[];records=[]
        remaining=f.operation_limit-f.time
        if remaining<=0 or not gs:return None,[]
        for i,j in enumerate(f.jobs.values()):
            if j.phase!='queued' or j.release>f.time or j.weight_kg>1.5:continue
            if f.delivery_mode(j)!='ground' and not us and (not hasattr(f,'meeting') or not any(a.kind=='uav' and a.task in f.jobs for a in f.agents.values())):continue
            if hasattr(getattr(f,'city',None),'pickup_available') and not f.city.pickup_available(f,j):continue
            # Geometry only: visual routing label is unavailable before inspection.
            transfer=getattr(f,'fixed_meetings',{}).get(j.name,j.transfer or [0,j.source[1],.08])
            goals=list(j.delivery_goals.values()) or [[-3-k*.8,j.source[1],.08] for k in range(3)]
            perception=getattr(f,'site_perception',None)
            if perception is not None and f.delivery_mode(j)!='ground':
                points=[p for i,p in enumerate(f.city.meeting_sites) if not perception.pending and f.motion.plan([j.source[0]+.45,j.source[1],.25],[p[0]+.25,p[1],.25],'ground') is not None]
                if not points:continue
                transfer=min(points,key=lambda p:math.dist(j.source,p)+sum(math.dist(p,g) for g in goals)/len(goals))
            last_mile=sum(math.dist(transfer,g) for g in goals)/len(goals)
            direct=f.delivery_mode(j)=='ground'
            for g in gs:
                for u in ([None] if direct else (us or [None])):
                    physical=f.motion.name=='gazebo_diffdrive_rotor_feedback';speed=f.estimated_ground_speed
                    approach=math.dist(g.pos,j.source)/speed
                    ge=math.dist(j.source,transfer)/speed
                    if hasattr(f.motion,'street_network'):
                        geometry=tuple((o.name,o.low,o.high) for o in f.motion.obstacles)
                        def distance(a,b):
                            key=(geometry,tuple(round(v,2) for v in a),tuple(round(v,2) for v in b))
                            if key not in self.route_cache:
                                path=f.motion.plan(a,b,'ground')
                                self.route_cache[key]=sum(math.dist(x,y) for x,y in zip([a]+path,path)) if path is not None else math.inf
                            return self.route_cache[key]
                        approach=distance(g.pos,[j.source[0]+.6,j.source[1],.25])/speed
                        ge=distance([j.source[0]+.45,j.source[1],.25],[transfer[0]+.25,transfer[1],.25])/speed
                        if not math.isfinite(approach+ge):continue
                    if direct:
                        paths=[f.motion.plan(j.source,p,'ground') for p in goals]
                        if any(p is None for p in paths):continue
                        from .rendezvous import path_length
                        delivery=sum(path_length([j.source]+p) for p in paths)/len(paths)
                        ue=0.;receipt=approach+delivery/speed+8
                        cycle=receipt+sum(math.dist(p,f.home(g)) for p in goals)/len(goals)/speed+2
                    else:
                        if u is not None:ue=math.dist(u.pos,transfer)/f.uav_speed+7
                        else:
                            # Conservative waiting estimate for pre-pickup while the UAV
                            # finishes its existing delivery and returns. Same model features.
                            delays=[]
                            for busy in f.agents.values():
                                if busy.kind!='uav':continue
                                old=f.jobs.get(busy.task);goal=f.goal(old) if old and old.destination else f.home(busy)
                                wait=(math.dist(busy.pos,goal)+math.dist(goal,f.home(busy)))/f.uav_speed+30
                                delays.append(wait+math.dist(f.home(busy),transfer)/f.uav_speed+7)
                            ue=min(delays,default=180.)
                        receipt=approach+max(ge,ue)+last_mile/f.uav_speed+23
                        cycle=receipt+math.dist(transfer,f.home(g))/speed+(math.dist(transfer,f.home(u)) if u is not None else min(math.dist(transfer,f.home(a)) for a in f.agents.values() if a.kind=='uav'))/f.uav_speed+12
                    feasible=math.isfinite(receipt) and math.isfinite(cycle)
                    value=POINTS[j.priority]
                    # Keep checkpoint dimensions, but neutralize legacy deadline
                    # and completion-before-cutoff features. No retraining implied.
                    rows.append([j.priority/3,value/9,receipt/180,cycle/240,remaining/300,0.,0.,0.,approach/120,ue/120,(f.time-j.release)/300,i/max(1,len(f.jobs)-1),float(feasible)])
                    records.append(dict(package=j.name,ground=g.name,uav=u.name if u else None,mode='ground' if direct else 'relay',priority=j.priority,value=value,receipt_eta_s=receipt,cycle_eta_s=cycle,feasible=bool(feasible),value_rate=value/max(1,cycle) if feasible else -1.,queue_order=i))
        if not rows or not any(r['feasible'] for r in records):return None,[]
        obs=dict(candidates=np.array(rows,dtype=np.float32),action_mask=np.ones(len(rows),dtype=np.int8),global_state=np.array([remaining/300,f.time/300,len(rows)/max(1,len(f.jobs)*len(gs)*len(us)),sum(j.receipt_confirmed for j in f.jobs.values())/len(f.jobs),len(gs)/len(f.agents),1.],dtype=np.float32))
        return obs,records
    def choose(self,obs,records):
        if self.policy in ('learned','untrained'):return self.network.select(obs)
        if self.policy=='random':return self.rng.randrange(len(records))
        if self.policy=='fifo':return min(range(len(records)),key=lambda i:(records[i]['queue_order'],records[i]['cycle_eta_s']))
        if self.policy=='priority':return max(range(len(records)),key=lambda i:(records[i]['priority'],-records[i]['receipt_eta_s']))
        return max(range(len(records)),key=lambda i:(records[i]['feasible'],records[i]['value_rate'],-records[i]['receipt_eta_s']))
    def select(self,f):
        perception=getattr(f,'site_perception',None)
        if perception is not None and perception.pending:return None
        obs,records=self.observation(f)
        if not records:return None
        directive=getattr(perception,'directive',None)
        selected_priority=None
        if directive:
            # Apply the command-center policy before MLP ranking. Keep the
            # checkpoint feature schema; never override physical feasibility.
            selected_priority=next((p for p in directive['priority_order'] if any(r['feasible'] and r['priority']==p for r in records)),None)
            indices=[i for i,r in enumerate(records) if r['feasible'] and r['priority']==selected_priority]
            if not indices:return None
            obs=dict(obs,candidates=obs['candidates'][indices],action_mask=obs['action_mask'][indices])
            records=[records[i] for i in indices]
        i=self.choose(obs,records);chosen=records[i]
        if self.network is not None:
            legal,_,_,prob=self.network.forward(obs)
            for k,p in zip(legal,prob):records[int(k)]['model_probability']=float(p)
        # Deduplicate pairs in the presentation; preserve selected pair and raw estimate.
        key='model_probability' if self.network is not None else 'value_rate'
        ranked=sorted(records,key=lambda r:r.get(key,0),reverse=True)
        seen=set();ranking=[]
        for row in ranked:
            if row['package'] not in seen:ranking.append(row);seen.add(row['package'])
        self.last_decision=dict(policy=self.name,checkpoint_sha256=self.checkpoint_sha256,command_directive=directive,selected_priority=selected_priority,selected=(chosen['package'],chosen['ground'],chosen['uav'],chosen.get('mode','relay')),operation_remaining_s=f.operation_limit-f.time,estimate_note='ETA is an estimate, not a delivery guarantee',chosen=chosen,ranking=ranking)
        return self.last_decision['selected']
