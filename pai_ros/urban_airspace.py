"""Cached visibility routing through the town's open intersections and river.
Roof keep-out volumes remain hard constraints. This is geometric routing, not a
learned controller or permission for real flight.
"""
import heapq,math
from .airspace import CRUISE
class UrbanAirspace(list):
    def __init__(self,areas,nodes,margin=.42):super().__init__(areas);self.nodes=nodes;self.margin=margin;self.cached={}
    def plan(self,start,target,lane_y=0,z=CRUISE):
        points=[[x,y,z] for x,y in self.nodes]
        clear=lambda a,b:not any(r.intersects(a,b,self.margin) for r in self)
        begin=[start[0],start[1],z];end=[target[0],target[1],z]
        if not clear(begin,begin) or not clear(end,end):raise ValueError('urban flight endpoint intersects a roof keep-out')
        key=(z,tuple((r.xmin,r.xmax,r.ymin,r.ymax,r.ceiling) for r in self))
        if key not in self.cached:
            graph=[[] for _ in points]
            for i,a in enumerate(points):
                for j in range(i):
                    if clear(a,points[j]):
                        d=math.dist(a,points[j]);graph[i].append((j,d));graph[j].append((i,d))
            self.cached={key:graph}
        graph=[list(g) for g in self.cached[key]];n=len(points);points += [begin,end];graph += [[],[]]
        for i in (n,n+1):
            for j in range(i):
                if clear(points[i],points[j]):
                    d=math.dist(points[i],points[j]);graph[i].append((j,d));graph[j].append((i,d))
        queue=[(0.,n)];costs={n:0.};parents={n:None}
        while queue:
            c,i=heapq.heappop(queue)
            if c>costs[i]:continue
            if i==n+1:break
            for j,d in graph[i]:
                if c+d<costs.get(j,math.inf):costs[j]=c+d;parents[j]=i;heapq.heappush(queue,(c+d,j))
        if n+1 not in parents:raise ValueError('no urban flight corridor')
        route=[];i=n+1
        while i is not None:route.append(points[i]);i=parents[i]
        return list(reversed(route))
