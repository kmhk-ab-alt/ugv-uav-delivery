"""Compact mixed-use town with shared intersections and mountain delivery sites.
The town footprint is independent of order count. New parcels reuse loading bays;
extra orders never add another private street. Dimensions are simulation metres.
"""
from dataclasses import dataclass
import random
from .city import City
from .safety import Obstacle
from .airspace import RestrictedArea
from .street_network import StreetNetwork

@dataclass
class UrbanCity(City):
    roads:list
    buildings:list
    places:list
    meeting_sites:list
    pickup_sites:list
    delivery_sites:list
    layout:str='blocks'
    def pickup_index(self,point):
        return min(range(len(self.pickup_sites)),key=lambda i:sum((point[k]-self.pickup_sites[i][k])**2 for k in (0,1)))
    def pickup_available(self,fleet,job):
        index=self.pickup_index(job.source)
        perception=getattr(fleet,'site_perception',None)
        if perception is not None and not perception.available('pickup_'+'AB'[index]):return False
        return not any(j.name!=job.name and j.phase in ('preflight','approach','observe','ground_pickup')
                       and self.pickup_index(j.source)==index for j in fleet.jobs.values())
    def apply(self,fleet):
        slots=max(sum(a.kind==k for a in fleet.agents.values()) for k in ('ground','uav'))
        for i in range(4,slots):
            road=dict(a=[19.,-4.+2*i],b=[21.,-4.+2*i],width=1.8)
            if road not in self.roads:self.roads.append(road)
        for road in self.roads:
            if road['a']==[19.,-4.] and road['b'][0]==19.:road['b'][1]=max(20.,-4.+2*(slots-1)+2)
        super().apply(fleet)
        for a in fleet.agents.values():
            i=int(a.name.rsplit('_',1)[1]);y=-4+2.*i+(.8 if a.kind=='uav' else 0.)
            a.pos[:]=[21.,y,.25 if a.kind=='ground' else .20]
            fleet.homes[a.name]=[21.,y,.25 if a.kind=='ground' else (.055 if fleet.motion.name=='gazebo_diffdrive_rotor_feedback' else .20)]
        fleet.motion.street_network=StreetNetwork(self.roads)
        from .urban_airspace import UrbanAirspace
        fleet.restricted_areas=UrbanAirspace(self.restrictions,fleet.motion.street_network.nodes,margin=.9 if fleet.motion.name=='gazebo_diffdrive_rotor_feedback' else .42)
        fleet.motion.air_network=fleet.restricted_areas
        fleet.motion.set_restrictions(fleet.restricted_areas)
        for i,j in enumerate(fleet.jobs.values()):
            index=(3 if i%2==0 else 0)+(i//2)%3
            site=self.delivery_sites[index]
            j.numbered_destination=site['id'];j.address='配送先 '+site['id'][1:]
            j.recipient='山の受領所' if i%2==0 else '市街地の受領所'
            j.delivery_goals={j.numbered_destination:self.goals[i][0].copy()}
        fleet.metrics['city']['layout']='blocks'

def generate_urban(seed,count=12,family='train'):
    if count<1 or family not in ('train','shift'):raise ValueError('invalid urban arguments')
    rng=random.Random(seed);roads=[]
    # Shared north/south streets and cross streets, connected across the whole town.
    for xs in ((1.,7.,13.,19.),):
        for x in xs:roads.append(dict(a=[x,-4.],b=[x,20.],width=2.4))
        for y in (-4.,4.,12.,20.):roads.append(dict(a=[min(xs),y],b=[max(xs),y],width=2.4))
    for y in (-4.,-2.,0.,2.):roads.append(dict(a=[19.,y],b=[21.,y],width=1.8))
    buildings=[];obstacles=[];areas=[]
    def building(name,x,y,hx,hy,height,style):
        buildings.append(dict(name=name,x=x,y=y,hx=hx,hy=hy,height=height,style=style))
        obstacles.append(Obstacle('city_building_'+name,(x-hx,y-hy,0),(x+hx,y+hy,height)))
        areas.append(RestrictedArea('roof_'+name,x-hx-.05,x+hx+.05,y-hy-.05,y+hy+.05))
    styles=('shop','apartment','office','house','cafe')
    for bank,centers in [('east',(4.,10.,16.))]:
        for col,x in enumerate(centers):
            for row,y in enumerate((0.,8.,16.)):
                if bank=='east' and col==1 and row==1:continue # civic green, not an empty duplicate block
                style=styles[(col+2*row+(0 if bank=='east' else 1))%len(styles)]
                height=rng.uniform(1.4,2.8)
                building(f'{bank}_{col}_{row}_a',x,y-1.05,1.35,.85,height,style)
                building(f'{bank}_{col}_{row}_b',x+rng.uniform(-.18,.18),y+1.15,1.15,.70,rng.uniform(1.1,2.3),'house' if row%2 else 'shop')
    building('hub',26.,-.2,1.35,2.7,2.1,'hub')
    for y in (8.,16.):
        building('collection_'+str(int(y)),25.,y,1.5,1.1,1.6,'hub')
        roads.append(dict(a=[19.,y],b=[22.,y],width=3.6))
    places=[dict(name='物流拠点',position=[26.,-.2,2.9]),dict(name='商店街',position=[7.,3.,2.6]),dict(name='中央公園',position=[10.,8.,1.6]),dict(name='住宅街',position=[16.,16.,3.2]),dict(name='山間の配送地区',position=[-8.,8.,3.0]),dict(name='集荷センター A',position=[25.,8.,2.5]),dict(name='集荷センター B',position=[25.,16.,2.5]),dict(name='合流地点 A',position=[1.,4.,1.2]),dict(name='合流地点 B',position=[1.,12.,1.2])]
    # Elevated delivery terrace has no road connection for a wheeled UGV.
    obstacles.append(Obstacle('city_building_mountain_terrace',(-14.,-3.,0),(-3.,21.,1.2)))
    for y in (1.,9.,17.):
        obstacles.append(Obstacle('city_building_mountain_hut_'+str(y),(-13.82,y-.82,1.2),(-12.18,y+.82,2.32)))
    pickup_sites=[[21.55,8.,.08],[21.55,16.,.08]]
    meeting_sites=[[1.,4.,.08],[1.,12.,.08]]
    delivery_sites=[dict(id=f'D{i+1:02d}',position=p) for i,p in enumerate(
        [[7.,1.,.08],[13.,9.,.08],[7.,17.,.08],[-10.8,1.,1.28],[-10.8,9.,1.28],[-10.8,17.,1.28]])]
    sources=[];goals=[];transfers=[]
    for i in range(count):
        center=(i//2)%2;slot=sum((k//2)%2==center for k in range(i));slots=sum((k//2)%2==center for k in range(count))
        # Individual parcel slots share one facility; orders do not add facilities.
        y=pickup_sites[center][1]+.32*(slot-(slots-1)/2)
        sources.append([21.55,round(y,3),.08]);transfers.append(meeting_sites[center].copy())
        site_index=(3 if i%2==0 else 0)+(i//2)%3
        point=delivery_sites[site_index]['position'].copy()
        slots=(count+5-i%6)//6
        point[1]+=.32*((i//6)-(slots-1)/2)
        goals.append([point])
    return UrbanCity(seed,family,[p[1] for p in sources],sources,goals,transfers,21.,obstacles,areas,roads,buildings,places,meeting_sites,pickup_sites,delivery_sites)

def road_surface_rectangles(roads, padding=0.):
    """Tile the road union without overlapping faces at intersections."""
    rects=[]
    for r in roads:
        a,b=r['a'],r['b'];w=r['width']/2+padding
        rects.append((min(a[0],b[0])-w,max(a[0],b[0])+w,
                      min(a[1],b[1])-w,max(a[1],b[1])+w))
    xs=sorted({x for r in rects for x in r[:2]})
    for left,right in zip(xs,xs[1:]):
        intervals=sorted((r[2],r[3]) for r in rects if r[0]<right and r[1]>left)
        merged=[]
        for low,high in intervals:
            if merged and low<=merged[-1][1]:merged[-1][1]=max(high,merged[-1][1])
            else:merged.append([low,high])
        for low,high in merged:yield left,right,low,high


def scenery(fleet):
    from .neighborhood import box,cylinder,tree
    c=fleet.city;s='<geom type="plane" size="0 0 .1" pos="0 0 -.04" material="lawn"/>'
    parking_end=max(3.,max(a.pos[1] for a in fleet.agents.values())+1)
    s+=box((22,(parking_end-5)/2,.01),(3.2,(parking_end+5)/2,.01),'.62 .65 .62 1')
    s+=box((7,8,-.02),(23,17,.015),'.43 .51 .39 1')
    # Each layer is a non-overlapping union, including shared junctions.
    for padding,z,hz,color in ((.23,.012,.012,'.69 .70 .66 1'),(0.,.024,.006,'.19 .23 .26 1')):
        for left,right,low,high in road_surface_rectangles(c.roads,padding):
            s+=box(((left+right)/2,(low+high)/2,z),((right-left)/2,(high-low)/2,hz),color)
    for r in c.roads:
        a,b=r['a'],r['b'];h=a[1]==b[1];length=abs(b[0]-a[0]) if h else abs(b[1]-a[1])
        for k in range(int(length/1.3)):
            p=[a[0]+(b[0]-a[0])*(k+.5)/(length/1.3),a[1]+(b[1]-a[1])*(k+.5)/(length/1.3)]
            # Do not draw lane markings through junctions.
            if (h and any(abs(p[0]-x)<1.4 for x in (1,7,13,19,-16,-10,-4))) or (not h and any(abs(p[1]-y)<1.4 for y in (-4,4,12,20))):continue
            s+=box((*p,.035),(.22,.025,.003) if h else (.025,.22,.003),'.90 .85 .64 1')
    # Crosswalks at selected junctions; connected streets remain unobstructed.
    for x,y in ((7,4),(13,12),(-10,4),(-4,12),(19,-4)):
        for k in range(6):s+=box((x-1+k*.38,y+1.38,.04),(.105,.30,.004),'.92 .92 .84 1')
    palette={'shop':'.72 .49 .34 1','apartment':'.69 .71 .65 1','office':'.49 .62 .66 1','house':'.81 .73 .58 1','cafe':'.60 .68 .53 1','hub':'.55 .63 .66 1'}
    for b in c.buildings:
        x,y,hx,hy,h=b['x'],b['y'],b['hx'],b['hy'],b['height'];style=b['style']
        s+=box((x,y,h/2),(hx,hy,h/2),palette[style])+box((x,y,h+.045),(hx+.08,hy+.08,.045),'.22 .28 .30 1')
        # Facades, entrances and shop canopies vary by use.
        for z in (.65,1.4,2.1):
            if z>h-.15:continue
            for dx in (-.65,0,.65):s+=box((x+dx,y-hy-.018,z),(.19,.018,.20),'.18 .36 .43 1')
        s+=box((x+.48,y-hy-.025,.38),(.22,.02,.38),'.24 .28 .29 1')
        if style in ('shop','cafe'):
            s+=box((x,y-hy-.25,1.12),(hx,.28,.055),'.19 .49 .47 1' if style=='cafe' else '.72 .30 .24 1')
            s+=box((x,y-hy-.035,1.36),(hx*.8,.02,.12),'.91 .85 .66 1')
        if style=='house':s+=box((x,y,h+.15),(hx*.65,hy*.6,.10),'.47 .27 .21 1')
        if style=='hub':
            for dx in (-.65,.65):s+=box((x+dx,y-hy-.02,.7),(.45,.02,.65),'.25 .32 .36 1')
        if style in ('office','apartment'):s+=box((x-.4,y+.2,h+.15),(.30,.25,.10),'.42 .49 .49 1')
    # Stepped forested mountainside, with a flat collision-backed receiving terrace.
    s+=box((-8.5,9,.18),(6.,13.2,.18),'.35 .42 .30 1')
    s+=box((-8.5,9,.52),(6.,12.7,.16),'.39 .45 .32 1')
    s+=box((-8.5,9,.94),(5.5,12.,.26),'.45 .50 .35 1')
    for y in (1.,9.,17.):
        s+=box((-13.,y,1.7),(.65,.65,.5),'.54 .34 .20 1')
        s+=box((-13.,y,2.26),(.82,.82,.06),'.27 .31 .29 1')
        s+=box((-12.34,y,1.65),(.015,.22,.35),'.25 .20 .16 1')
        s+=box((-11.65,y,1.205),(.65,.30,.005),'.65 .62 .52 1')
        for x in (-15.,-2.5):
            # Trees stand beside the receiving terrace, clear of delivery bays.
            s+=tree(x,y+2)
    # One neutral receiving pad per numbered destination, independent of importance.
    for site in c.delivery_sites:
        goal=site['position']
        s+=cylinder((goal[0],goal[1],goal[2]-.04),.30,.005,'.78 .80 .78 1')
    # Park paths, fountain, playground and benches are in a block, outside roads.
    s+=box((10,8,.038),(1.45,2.45,.012),'.32 .49 .30 1')+box((10,8,.06),(.25,2.45,.012),'.73 .68 .54 1')
    s+=cylinder((10,8,.16),.55,.10,'.62 .65 .61 1')+cylinder((10,8,.27),.42,.015,'.20 .55 .64 1')
    for x,y in ((9.1,6.3),(10.9,9.6),(9.1,9.8),(10.9,6.2)):
        s+=tree(x,y)+box((x,y-.6,.25),(.4,.12,.035),'.44 .29 .18 1')
    for x in (-16.,30.):
        for y in range(-3,23,4):s+=tree(x,y)
    for x,y,_ in c.pickup_sites:
        s+=box((x,y,.034),(.8,1.05,.004),'.21 .49 .56 1')
        for dy in (-1.,1.):s+=box((x,y+dy,.042),(.8,.025,.003),'.95 .87 .59 1')
        s+=box((23.4,y,1.35),(.45,1.1,.07),'.16 .42 .54 1')
    for x,y,_ in c.meeting_sites:
        s+=cylinder((x,y,.035),.65,.005,'.76 .58 .27 1')
    for j in fleet.jobs.values():
        x,y,_=j.source;s+=box((x,y,.035),(.30,.37,.008),'.70 .57 .27 1')
    for a in fleet.agents.values():
        if a.kind=='uav':s+=cylinder((a.pos[0],a.pos[1],.025),.45,.01,'.22 .48 .63 1')
    return s
