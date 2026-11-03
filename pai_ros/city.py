"""Seeded delivery-street family. Geometry is independent of orders and labels.
Not an arbitrary urban map generator or SLAM. All roads are connected corridors.
"""
from dataclasses import dataclass,asdict
import hashlib,json,random,math
from .safety import Obstacle
from .airspace import RestrictedArea

@dataclass
class City:
    seed:int
    family:str
    lanes:list
    sources:list
    goals:list
    transfers:list
    home_x:float
    obstacles:list
    restrictions:list
    def manifest(self):
        d=asdict(self)
        d['geometry_sha256']=hashlib.sha256(json.dumps({k:v for k,v in d.items() if k not in ('seed','family')},sort_keys=True).encode()).hexdigest()
        return d
    def apply(self,fleet):
        if len(self.lanes)!=len(fleet.jobs):raise ValueError('city/order count mismatch')
        fleet.city=self;fleet.city_manifest=self.manifest();fleet.homes={}
        for i,a in enumerate(fleet.agents.values()):
            idx=int(a.name.rsplit('_',1)[1]);lane=self.lanes[idx%len(self.lanes)]
            # Extra agents get additional apron parking rows.
            y=lane+(1 if a.kind=='uav' else 0)+int(idx/len(self.lanes))*2
            a.pos[:]=[self.home_x,y,.25 if a.kind=='ground' else .20]
            fleet.homes[a.name]=[self.home_x,y,.25 if a.kind=='ground' else (.055 if fleet.motion.name=='gazebo_diffdrive_rotor_feedback' else .20)]
        for i,j in enumerate(fleet.jobs.values()):
            j.source[:]=self.sources[i];j.pos[:]=j.source
            j.delivery_goals={c:list(g) for c,g in zip(('red','green','blue'),self.goals[i])}
            j.transfer=list(self.transfers[i])
        fleet.restricted_areas=list(self.restrictions);fleet.motion.set_restrictions(fleet.restricted_areas)
        fleet.motion.set_obstacles(self.obstacles)
        fleet.metrics['city']=dict(seed=self.seed,family=self.family,geometry_sha256=fleet.city_manifest['geometry_sha256'],map_access='provided')

def generate_city(seed,count=6,family='train',layout='lanes'):
    if layout=='blocks':
        from .urban_city import generate_urban
        return generate_urban(seed,count,family)
    if layout!='lanes':raise ValueError('invalid city layout')
    if count<1 or family not in ('train','shift'):raise ValueError('invalid city arguments')
    rng=random.Random(seed);shift=family=='shift'
    # Held-out family has longer streets and more road furniture than training.
    spacing=rng.uniform(4.1,5.0);y=rng.uniform(-1.5,1.5)
    lanes=[];sources=[];goals=[];transfers=[];obstacles=[];areas=[]
    home=5.0 if not shift else 5.6
    for i in range(count):
        if i:y+=spacing+rng.uniform(-.2,.2)
        lanes.append(round(y,3));sx=rng.uniform(2.6,3.5);tx=rng.uniform(-.15,.05)
        sources.append([round(sx,3),round(y,3),.08]);transfers.append([round(tx,3),round(y,3),.08])
        gx=rng.uniform(-3.2,-2.6)-(rng.uniform(.6,1.0) if shift else 0)
        gs=[[round(gx-k*.65,3),round(y+rng.uniform(-.12,.12),3),.08] for k in range(3)];goals.append(gs)
        for k,g in enumerate(gs):
            w=rng.uniform(.38,.52);h=rng.uniform(1.8,2.7)
            obstacles.append(Obstacle(f'city_building_{i}_{k}',(g[0]-w,y+1.35,0),(g[0]+w,y+2.2,h)))
        for k in range(2 if shift else 1):
            x=rng.uniform(1.45,1.65) if k==0 else rng.uniform(-2.2,-1.0)
            dy=rng.uniform(-.15,.15)
            obstacles.append(Obstacle(f'city_barrier_{i}_{k}',(x-.17,y+dy-.18,0),(x+.17,y+dy+.18,rng.uniform(.5,.8))))
        ax=rng.uniform(-1.7,-1.2);offset=rng.uniform(-.15,.15)
        areas.append(RestrictedArea(f'city_airspace_{i}',ax-.35,ax+.35,y+offset-.32,y+offset+.32))
    return City(seed,family,lanes,sources,goals,transfers,home,obstacles,areas)

class LocalObstacleMap:
    """Ideal range discovery of static boxes using simulator truth within a radius.
    This is a limited sensor fixture, not camera perception, LiDAR raycasting or SLAM.
    Dispatch receives discovered geometry only. The safety shield uses full truth.
    """
    def __init__(self,fleet,radius=1.8):
        if radius<=0:raise ValueError('radius must be positive')
        self.truth=list(fleet.motion.obstacles);self.known={};self.radius=radius
        fleet.observed_obstacles=[]
        fleet.metrics['city']['map_access']='local_range_fixture_with_privileged_safety_shield'
    def tick(self,fleet):
        for o in self.truth:
            if o.name in self.known:continue
            if any(math.sqrt(sum(max(lo-p,0,p-hi)**2 for p,lo,hi in zip(a.pos,o.low,o.high)))<=self.radius for a in fleet.agents.values()):
                self.known[o.name]=o
                fleet.events.append(dict(time=round(fleet.time,3),event='obstacle_discovered',obstacle=o.name))
        fleet.observed_obstacles=list(self.known.values())
        fleet.metrics['discovered_obstacles']=len(self.known)

def scenery(fleet):
    """Geometry aligned with the generated map; static collisions added by SDF builder."""
    from .neighborhood import box,cylinder,tree
    if getattr(fleet.city,'layout','lanes')=='blocks':
        from .urban_city import scenery as urban_scenery
        return urban_scenery(fleet)
    c=fleet.city;y0=min(c.lanes)-2;y1=max(c.lanes)+2
    out='<geom type="plane" size="0 0 .1" material="lawn"/>'
    warehouse=c.home_x+(2.5 if hasattr(fleet,'meeting') else 1.5)
    out+=box((warehouse,(y0+y1)/2,1.1),(.7,(y1-y0)/2,1.1),'.58 .65 .69 1')
    out+=box((warehouse,(y0+y1)/2,2.24),(.8,(y1-y0)/2+.1,.04),'.2 .28 .34 1')
    out+=box((c.home_x,(y0+y1)/2,.01),(.7,(y1-y0)/2,.01),material='concrete')
    if hasattr(fleet,'meeting'):
        for a in fleet.agents.values():
            if a.kind=='uav':out+=cylinder((a.pos[0],a.pos[1],.02),.42,.012,'.2 .45 .7 1')
    front=warehouse-.71
    for lane in c.lanes:
        out+=box((front,lane,.75),(.012,.52,.68),'.27 .34 .38 1')
        for h in (.2,.38,.56,.74,.92,1.10,1.28):out+=box((front-.015,lane,h),(.006,.51,.008),'.49 .55 .57 1')
        out+=box((front-.02,lane,1.61),(.01,.55,.08),'.16 .43 .49 1')
        out+=box((warehouse,lane,2.30),(.46,.38,.02),'.10 .25 .33 1')
    for i,j in enumerate(fleet.jobs.values()):
        left=min(g[0] for g in c.goals[i])-.5;right=c.home_x+.4;y=c.lanes[i]
        out+=box(((left+right)/2,y,.006),((right-left)/2,.55,.006),material='road')
        for dy in (-.7,.7):out+=box(((left+right)/2,y+dy,.015),((right-left)/2,.12,.015),material='concrete')
        for x in range(math.floor(left),math.ceil(right)):out+=box((x,y,.018),(.13,.018,.004),'.9 .87 .69 1')
        for k,g in enumerate(c.goals[i]):out+=cylinder((g[0],g[1],.022),.23,.012,('.85 .2 .2 1','.2 .7 .3 1','.2 .4 .85 1')[k])
        out+=cylinder((j.transfer[0],y,.022),.3,.012,'.2 .6 .65 1')
        out+=tree(c.home_x+.65,y+1.25)+tree(left-.4,y+1.1)
    for i,o in enumerate(c.obstacles):
        if 'building' not in o.name:continue # barriers get their own models/collisions
        mid=[(a+b)/2 for a,b in zip(o.low,o.high)];half=[(b-a)/2 for a,b in zip(o.low,o.high)]
        out+=box(mid,half,('.72 .65 .55 1','.60 .71 .67 1','.74 .60 .56 1')[i%3])
        out+=box((mid[0],mid[1],o.high[2]+.04),(half[0]+.05,half[1]+.05,.04),'.25 .3 .34 1')
        for dx in (-.17,.17):out+=box((mid[0]+dx,o.low[1]-.015,1.05),(.10,.02,.18),material='glass')
    for a in c.restrictions:
        out+=box(((a.xmin+a.xmax)/2,(a.ymin+a.ymax)/2,.025),((a.xmax-a.xmin)/2,(a.ymax-a.ymin)/2,.01),'.8 .25 .18 .55')
    return out

def configure_orders(fleet,seed):
    rng=random.Random(seed)
    for i,j in enumerate(fleet.jobs.values()):
        j.priority=rng.randint(1,3);j.deadline=95+i*14+rng.randint(0,65);j.release=0 if i<2 else rng.uniform(10,45)
