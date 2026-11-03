"""Procedural neighborhood assets; all positions keep delivery corridors clear."""
from html import escape

ASSETS = '''
<texture name="sky" type="skybox" builtin="gradient" rgb1=".22 .43 .65" rgb2=".85 .91 .95" width="512" height="3072"/>
<texture name="asphalt" type="2d" builtin="checker" rgb1=".16 .19 .22" rgb2=".18 .21 .24" width="256" height="256"/>
<material name="road" texture="asphalt" texrepeat="12 3" texuniform="true" reflectance=".05"/>
<texture name="grass" type="2d" builtin="checker" rgb1=".25 .38 .25" rgb2=".28 .42 .28" width="256" height="256"/>
<material name="lawn" texture="grass" texrepeat="30 30" texuniform="true"/>
<material name="concrete" rgba=".70 .72 .70 1"/>
<material name="cardboard" rgba=".66 .45 .26 1"/>
<material name="glass" rgba=".18 .38 .48 1" reflectance=".3"/>
<material name="metal" rgba=".33 .39 .43 1" shininess=".6"/>
'''

def box(pos, size, color=None, material=None, **attrs):
    opts = dict(type='box', pos=' '.join(map(str,pos)), size=' '.join(map(str,size)), contype='0', conaffinity='0')
    if color: opts['rgba'] = color
    if material: opts['material'] = material
    opts.update(attrs)
    return '<geom ' + ' '.join(f'{k}="{escape(str(v))}"' for k,v in opts.items()) + '/>'

def cylinder(pos, radius, height, color):
    return f'<geom type="cylinder" pos="{pos[0]} {pos[1]} {pos[2]}" size="{radius} {height}" rgba="{color}" contype="0" conaffinity="0"/>'

def tree(x,y):
    return (cylinder((x,y,.48),.055,.48,'.30 .22 .14 1')+
            f'<geom type="ellipsoid" pos="{x} {y} 1.15" size=".32 .32 .6" rgba=".13 .32 .22 1" contype="0" conaffinity="0"/>'+
            f'<geom type="ellipsoid" pos="{x+.14} {y} 1.38" size=".25 .25 .38" rgba=".22 .43 .25 1" contype="0" conaffinity="0"/>')

def building(x,y,style,index):
    palettes = [(' .80 .72 .60 1','.24 .31 .34 1'),('.64 .72 .70 1','.27 .36 .32 1'),('.78 .65 .58 1','.35 .22 .20 1')]
    wall, roof = palettes[index%3]
    out=box((x,y,.85),(.65,.62,.85),wall)
    out+=box((x,y,1.76),(.74,.7,.08),roof)
    # Front faces the road at smaller y; windows, recessed doorway, entry steps.
    out+=box((x,y-.635,.5),(.15,.014,.5),'.23 .25 .26 1')
    out+=box((x,y-.78,.04),(.25,.2,.04),material='concrete')
    for dx in (-.4,.4):
        out+=box((x+dx,y-.642,1.05),(.15,.018,.25),'.88 .89 .84 1')
        out+=box((x+dx,y-.665,1.05),(.12,.01,.22),material='glass')
        out+=box((x+dx,y-.678,1.05),(.008,.006,.22),'.75 .78 .75 1')
    if style=='clinic':
        out+=box((x,y-.68,1.48),(.10,.018,.035),'.86 .20 .20 1')
        out+=box((x,y-.685,1.48),(.035,.018,.10),'.86 .20 .20 1')
    elif style=='shop':
        out+=box((x,y-.85,1.36),(.66,.28,.045),'.20 .48 .33 1')
        for dx in (-.5,-.25,0,.25,.5):
            out+=box((x+dx,y-1.09,1.29),(.07,.025,.09),'.88 .84 .65 1')
    else:
        out+=box((x,y,1.9),(.55,.56,.06),roof,euler='0 0 0')
        out+=box((x+.4,y+.3,2.02),(.09,.09,.22),'.36 .33 .30 1')
    return out

def environment(fleet):
    from .fleet import LANE_SPACING, COLORS
    count=len(fleet.jobs)
    center=(count-1)*LANE_SPACING/2
    span=max(2.,count*LANE_SPACING/2)
    out='<geom type="plane" size="0 0 .1" material="lawn"/>'
    # Distribution center behind the loading apron, outside robot travel.
    out+=box((6.2,center,1.15),(.85,span,1.15),'.60 .65 .68 1')
    out+=box((6.2,center,2.34),(1.,span+.12,.06),'.20 .27 .31 1')
    out+=box((4.4,center,.005),(1.,span,.005),'.57 .60 .60 1')
    out+=box((5.32,center,1.94),(.018,span,.18),'.13 .30 .39 1')
    for i,j in enumerate(fleet.jobs.values()):
        y=j.source[1]
        # Loading bay shutter + striped bay apron.
        out+=box((5.33,y,.82),(.02,.6,.8),'.32 .39 .43 1')
        for z in (.2,.45,.7,.95,1.2,1.45):
            out+=box((5.30,y,z),(.014,.57,.008),'.49 .55 .57 1')
        for x in (3.,3.5,4.,4.5):
            out+=box((x,y-.48,.018),(.1,.03,.008),'.92 .70 .22 1')
        out+=box((-.7,y,.005),(4.05,.43,.005),material='road')
        out+=box((-.5,y+.55,.025),(4.8,.12,.025),material='concrete')
        out+=box((-.5,y-.55,.025),(4.8,.12,.025),material='concrete')
        for x in (-4,-3,-2,-1,0,1,2):
            out+=box((x,y,.013),(.16,.015,.004),'.88 .86 .70 1')
        # Separate street-side receiving facilities, one of each per address row.
        for k,x in enumerate((-3.,-3.8,-4.6)):
            # Color pads remain in the travel corridor; buildings are behind the sidewalk.
            color=('.85 .2 .2 1','.2 .7 .3 1','.2 .4 .85 1')[k]
            out+=cylinder((x,y,.025),.26,.014,color)
        # One address block per shipment; style corresponds to its routing label.
        style=('clinic','shop','home')[list(COLORS).index(j.color)]
        out+=building(-3.-list(COLORS).index(j.color)*.8,y+1.7,style,i)
        # Transfer port, bollards, illuminated pad and solar shelter.
        out+=cylinder((0,y,.022),.31,.012,'.21 .52 .56 1')
        for dy in (-.22,.22):
            out+=box((0,y+dy,.039),(.18,.015,.006),'.85 .94 .88 1')
        for x in (-.38,.38):
            out+=cylinder((x,y+.73,.14),.035,.14,'.92 .67 .20 1')
        out+=tree(1.4,y+1.1)+tree(-1.5,y+1.25)
        # Streetlight, bench and recycling bins.
        out+=cylinder((2.,y+.8,.9),.025,.9,'.24 .28 .3 1')
        out+=box((1.90,y+.8,1.82),(.15,.08,.025),'.95 .89 .65 1')
        out+=box((.8,y+.95,.25),(.32,.12,.035),'.40 .27 .17 1')
        for dx in (-.22,.22): out+=box((.8+dx,y+.95,.12),(.025,.1,.12),material='metal')
        for x,c in ((2.8,'.25 .43 .36 1'),(3.1,'.27 .38 .52 1')):
            out+=box((x,y+1.1,.20),(.1,.1,.2),c)
            out+=box((x,y+1.1,.41),(.11,.11,.015),'.21 .24 .25 1')
        # Parked service car, safely off the delivery route.
        out+=box((-.5,y+1.35,.22),(.43,.22,.13),'.62 .32 .22 1')
        out+=box((-.53,y+1.35,.42),(.23,.19,.09),material='glass')
        for dx in (-.28,.28):
            for dy in (-.22,.22):
                out+=f'<geom type="sphere" pos="{-.5+dx} {y+1.35+dy} .13" size=".1" rgba=".09 .1 .11 1" contype="0" conaffinity="0"/>'
    # A parked loading trolley deliberately blocks one delivery lane.
    # It provides a visible static-obstacle avoidance case, away from the inspection parcel.
    if fleet.jobs:
        y=next(iter(fleet.jobs.values())).source[1]
        out+=box((1.55,y,.20),(.25,.20,.16),'.32 .37 .40 1')
        out+=box((1.55,y,.48),(.20,.16,.12),'.66 .46 .27 1')
        for dx in (-.18,.18):
            for dy in (-.17,.17):
                out+=f'<geom type="sphere" pos="{1.55+dx} {y+dy} .07" size=".055" rgba=".08 .09 .10 1" contype="0" conaffinity="0"/>'
    # Deliberately restricted rectangles. These local coordinates are our demo additions.
    for area in fleet.restricted_areas:
        x=(area.xmin+area.xmax)/2;y=(area.ymin+area.ymax)/2
        out+=box((x,y,.025),((area.xmax-area.xmin)/2,(area.ymax-area.ymin)/2,.008),'.84 .35 .16 .60')
        for px in (area.xmin,area.xmax):
            out+=box((px,y,.055),(.018,(area.ymax-area.ymin)/2,.02),'.96 .65 .18 1')
            for py in (area.ymin,area.ymax):
                out+=cylinder((px,py,.20),.025,.20,'.88 .39 .15 1')
        for py in (area.ymin,area.ymax):
            out+=box((x,py,.055),((area.xmax-area.xmin)/2,.018,.02),'.96 .65 .18 1')
    # Individual marked landing spots at the depot.
    for agent in fleet.agents.values():
        if agent.kind!='uav': continue
        hx,hy,_=fleet.home(agent)
        out+=cylinder((hx,hy,.025),.30,.01,'.18 .27 .30 1')
        out+=box((hx,hy,.039),(.17,.018,.004),'.91 .93 .88 1')
        for dy in (-.12,.12):
            out+=box((hx,hy+dy,.039),(.018,.12,.004),'.91 .93 .88 1')
    return out

def parcel_geoms(job):
    from .fleet import COLORS
    from .fleet import PRIORITY_COLORS
    panel=PRIORITY_COLORS[job.priority] if job.numbered_destination else COLORS[job.color]
    route=' '.join(map(str,(*panel,1)))
    # Rich shapes retain a clear, saturated TOP routing panel for visual recognition.
    kind=job.package_type
    if kind=='medical_case':
        out=box((0,0,0),(.11,.085,.08),'.77 .81 .82 1',mass='.6')
        out+=box((0,0,.065),(.115,.09,.015),'.91 .92 .88 1')
        for x in (-.08,.08): out+=box((x,-.088,0),(.014,.008,.022),material='metal')
    elif kind=='grocery_crate':
        out=box((0,0,0),(.12,.095,.08),'.39 .30 .20 1',mass='.8')
        for z in (-.05,-.015,.02,.055):
            out+=box((0,-.098,z),(.12,.008,.009),'.67 .50 .28 1')
            out+=box((0,.098,z),(.12,.008,.009),'.67 .50 .28 1')
        out+=box((0,0,.075),(.124,.10,.005),'.74 .64 .40 1')
    else:
        out=box((0,0,0),(.11,.09,.08),material='cardboard',mass='.4')
        out+=box((0,0,.081),(.018,.09,.002),'.81 .69 .45 1')
        out+=box((0,-.091,0),(.018,.002,.08),'.81 .69 .45 1')
        for x in (-.109,.109): out+=box((x,0,0),(.002,.09,.08),'.56 .37 .22 1')
    out+=box((0,0,.085),(.078,.062,.002),route)
    # White waybill and barcode on the front leave the routing panel unobstructed.
    out+=box((.035,-.101,.015),(.035,.003,.035),'.93 .93 .87 1')
    for i in range(7):
        out+=box((.007+i*.008,-.105,.015),(.0015,.001,.023),'.12 .14 .15 1')
    return out
