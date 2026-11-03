"""Visual-only industrial detailing. No masses, collisions or control parameters."""
import random
from .neighborhood import box,cylinder,tree

def ground_details():
    s=box((-.01,0,.045),(.21,.185,.065),'.02 .85 .95 1')
    s+=box((.04,0,.118),(.17,.17,.014),'.08 .12 .16 1')
    s+=box((.40,0,.17),(.18,.145,.018),'.38 .43 .47 1')
    for y in (-.15,.15):s+=box((.4,y,.20),(.18,.009,.035),'.74 .78 .80 1')
    s+=cylinder((-.12,0,.17),.065,.035,'.08 .12 .15 1')+cylinder((-.12,0,.20),.058,.006,'.17 .55 .61 1')
    s+=box((.245,0,-.025),(.015,.19,.025),'.10 .13 .16 1')
    for y in (-.13,.13):
        s+=box((.253,y,.038),(.006,.036,.014),'.78 .95 1 1')
        s+=box((-.246,y,.035),(.004,.028,.012),'.95 .19 .12 1')
    for i in range(5):s+=box((-.19+i*.035,-.194,.037),(.009,.003,.032),'.26 .33 .38 1')
    s+=box((.04,-.196,.07),(.075,.004,.014),'.10 .54 .65 1')
    return s

def drone_details():
    s=box((0,0,.035),(.12,.075,.025),'.16 .30 .38 1')
    s+=box((-.025,0,.073),(.075,.055,.012),'1 .35 .03 1')
    for x in (-.06,-.03,0,.03):s+=box((x,0,.087),(.004,.048,.002),'.21 .27 .30 1')
    s+=box((.135,0,-.008),(.018,.035,.024),'.07 .10 .13 1')
    s+=cylinder((.151,0,-.005),.016,.01,'.14 .42 .53 1')
    for y in (-.10,.10):s+=box((0,y,-.038),(.13,.014,.007),'.28 .33 .36 1')
    for y,c in ((-.19,'.95 .16 .11 1'),(.19,'.15 .90 .55 1')):s+=box((.19,y,.035),(.012,.012,.012),c)
    return s

def backdrop(fleet):
    if getattr(getattr(fleet,'city',None),'layout','lanes')=='blocks':return ''
    ys=[j.source[1] for j in fleet.jobs.values()];mid=(min(ys)+max(ys))/2;span=max(8,max(ys)-min(ys)+8)
    rng=random.Random(6301);s=''
    # Scenery stays beyond the mission envelope, and never changes the planner map.
    for _ in range(65):
        x=rng.uniform(-10,-7.5);y=rng.uniform(min(ys)-5,max(ys)+5)
        s+=tree(x,y)
    for j in fleet.jobs.values():
        sx,y,_=j.source
        # Inspection/loading apron: strips, bollards and electric charging cabinets.
        for dx in (-.43,.43):
            s+=box((sx+dx,y-.45,.012),(.025,.38,.006),'.92 .77 .35 1')
        s+=box((sx+.7,y+1.15,.38),(.12,.1,.38),'.21 .30 .35 1')
        s+=box((sx+.7,y+1.045,.50),(.07,.005,.08),'.15 .64 .73 1')
        s+=cylinder((sx+.9,y+.75,.85),.025,.85,'.33 .39 .42 1')
        s+=box((sx+.80,y+.75,1.72),(.14,.07,.018),'.95 .92 .75 1')
        for dy in (-.72,.72):
            s+=cylinder((sx-.45,y+dy,.14),.035,.14,'.92 .69 .18 1')
    return s

def closure_visual(obstacle):
    half=(obstacle.high[1]-obstacle.low[1])/2
    if obstacle.high[0]-obstacle.low[0]>1:
        return box((0,0,-.31),((obstacle.high[0]-obstacle.low[0])/2,half,.012),'.12 .43 .57 1')
    s=box((0,0,-.31),(.15,half,.012),'.10 .42 .55 1')
    for side in (-.18,.18):
        s+=box((side,0,-.29),(.025,half,.04),'.43 .46 .43 1')
        s+=box((side,0,.18),(.025,half,.035),'.80 .67 .25 1')
        for i in range(int(half*2)+1):s+=cylinder((side,-half+i,.0),.022,.18,'.32 .37 .39 1')
    return s
