"""Map-aware route estimates averaged over all possible receiving bays.
Never reads a parcel's true label. Ideal-map estimates, not a learned navigator.
"""
import math
from .safety import SafeKinematicMotion
from .airspace import plan_outbound,CRUISE

def length(points):return sum(math.dist(a,b) for a,b in zip(points,points[1:]))

def estimate(fleet,job,mode):
    obstacles=getattr(fleet,'observed_obstacles',fleet.motion.obstacles)
    # Include discovered map, live closures and areas; changed maps invalidate cached costs.
    key=(job.name,mode,tuple((o.name,o.low,o.high) for o in obstacles),tuple((a.name,a.xmin,a.xmax,a.ymin,a.ymax) for a in fleet.restricted_areas))
    cache=getattr(fleet,'eta_cache',{});fleet.eta_cache=cache
    if key in cache:return cache[key]
    planner=SafeKinematicMotion();planner.set_obstacles(obstacles)
    src=[job.source[0],job.source[1],.25];transfer=[job.transfer[0],job.transfer[1],.25]
    def ground(a,b):
        if planner.static_clear(a,b,'ground'):return math.dist(a,b)
        path=planner.plan(a,b,'ground')
        return length([a]+path) if path else 200.
    gd=[];ud=[]
    for goal in job.delivery_goals.values():
        end=[goal[0],goal[1],.25]
        if mode=='scout':gd.append(ground(src,end));ud.append(0.)
        else:
            start=transfer if mode=='relay' else src
            try:route=plan_outbound(start,goal,fleet.restricted_areas,src[1]);flight=length(route)
            except ValueError:flight=200.
            if mode=='relay':gd.append(ground(src,transfer));ud.append(flight+4.)
            else:
                gd.append(sum(ground([a[0],a[1],.25],[b[0],b[1],.25]) for a,b in zip(route,route[1:])) if flight<200 else 200.)
                ud.append(flight)
    g=sum(gd)/len(gd);u=sum(ud)/len(ud);speed=.5 if fleet.motion.name=='gazebo_diffdrive_rotor_feedback' else .9
    eta=(g/speed+u/1.3+12 if mode=='relay' else g/speed+5 if mode=='scout' else max(g/speed,u/1.3)+8)
    cache[key]=(eta,g+u)
    return cache[key]
