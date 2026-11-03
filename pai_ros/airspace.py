"""Demo route planning and warning volumes inspired by PODIUM Odense Scenario 2.
Geometry, heights, routes and thresholds are our scaled simulation parameters.
"""
from dataclasses import dataclass
import heapq
import math

SCENARIO_ID='podium_odense_s2_adapted'
SCENARIO_URL='https://www.sesarju.eu/sites/default/files/documents/projects/783230_D4_1_PODIUM_Odense_demo_report%20%281_0%29.pdf'
CRUISE=3.2
RETURN_ALTITUDE=4.2

@dataclass(frozen=True)
class RestrictedArea:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    ceiling: float=8.

    def contains(self, pos, margin=0.):
        x,y,z=pos
        return (self.xmin-margin <= x <= self.xmax+margin and
                self.ymin-margin <= y <= self.ymax+margin and 0 <= z <= self.ceiling)

    def intersects(self, start, end, margin=0.):
        # Slab intersection of line segment and inflated axis-aligned volume.
        low, high=0.,1.
        for a,b,lo,hi in zip(start,end,
                (self.xmin-margin,self.ymin-margin,0),
                (self.xmax+margin,self.ymax+margin,self.ceiling)):
            delta=b-a
            if abs(delta)<1e-12:
                if a<lo or a>hi: return False
            else:
                t0,t1=(lo-a)/delta,(hi-a)/delta
                low,high=max(low,min(t0,t1)),min(high,max(t0,t1))
                if low>high: return False
        return True


def areas_for_jobs(jobs):
    return [RestrictedArea('restricted_'+j.name,-2.2,-.7,j.source[1]-.4,j.source[1]+.4)
            for j in jobs]


def plan_outbound(start, target, areas, lane_y):
    """Bounded 2D A* at cruise altitude, plus exact endpoints.

    Neighbors and endpoint links are checked as segments, not just as points.
    No permission or regulatory validity is implied by a feasible route.
    """
    if hasattr(areas,'plan'):return areas.plan(start,target,lane_y)
    step=.2
    def point(cell): return [cell[0]*step,lane_y+cell[1]*step,CRUISE]
    origin=(round(start[0]/step),round((start[1]-lane_y)/step))
    goal=(round(target[0]/step),round((target[1]-lane_y)/step))
    def clear(a,b): return not any(r.intersects(a,b,.42) for r in areas)
    begin=[start[0],start[1],CRUISE]
    finish=[target[0],target[1],CRUISE]
    def endpoint_cell(position,rounded):
        if not clear(position,position):raise ValueError('route endpoints intersect restricted airspace')
        candidates=[(rounded[0]+dx,rounded[1]+dy) for dx in range(-2,3) for dy in range(-2,3)]
        legal=[c for c in candidates if clear(position,point(c))]
        if not legal:raise ValueError('route endpoint cannot connect to grid')
        return min(legal,key=lambda c:(math.dist(position,point(c)),c))
    origin=endpoint_cell(begin,origin);goal=endpoint_cell(finish,goal)
    frontier=[(0.,origin)]
    previous={origin:None}
    costs={origin:0.}
    while frontier:
        _,cell=heapq.heappop(frontier)
        if cell==goal: break
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nxt=(cell[0]+dx,cell[1]+dy)
            if not (min(origin[0],goal[0])-12<=nxt[0]<=max(origin[0],goal[0])+12 and min(origin[1],goal[1])-16<=nxt[1]<=max(origin[1],goal[1])+16): continue
            if not clear(point(cell),point(nxt)): continue
            score=costs[cell]+1
            if score<costs.get(nxt,float('inf')):
                costs[nxt]=score;previous[nxt]=cell
                priority=score+abs(nxt[0]-goal[0])+abs(nxt[1]-goal[1])
                heapq.heappush(frontier,(priority,nxt))
    if goal not in previous: raise ValueError('no corridor available')
    cells=[];cell=goal
    while cell is not None:
        cells.append(cell);cell=previous[cell]
    cells.reverse()
    raw=[begin]+[point(c) for c in cells]+[finish]
    # Remove redundant straight segments without cutting corners across the area.
    route=[]
    for p in raw:
        if route and math.dist(route[-1],p)<1e-8: continue
        if len(route)>=2:
            a,b=route[-2],route[-1]
            v=[b[i]-a[i] for i in range(3)];w=[p[i]-b[i] for i in range(3)]
            cross=[v[1]*w[2]-v[2]*w[1],v[2]*w[0]-v[0]*w[2],v[0]*w[1]-v[1]*w[0]]
            if sum(c*c for c in cross)<1e-12 and clear(a,p): route.pop()
        route.append(p)
    return route
