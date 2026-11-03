"""Shared paved-road graph. Ground routing follows connected streets, not lawns.
Road corridors constrain robot centers with a body-clearance margin. Buildings and
other obstacles are still checked by the existing safety controller.
"""
import heapq,math

class StreetNetwork:
    def __init__(self,roads):
        self.roads=roads;self.cache={};nodes=set()
        for r in roads:nodes.update((tuple(r['a']),tuple(r['b'])))
        for h in roads:
            if h['a'][1]!=h['b'][1]:continue
            for v in roads:
                if v['a'][0]!=v['b'][0]:continue
                x,y=v['a'][0],h['a'][1]
                if min(h['a'][0],h['b'][0])<=x<=max(h['a'][0],h['b'][0]) and min(v['a'][1],v['b'][1])<=y<=max(v['a'][1],v['b'][1]):nodes.add((x,y))
        self.nodes=sorted(nodes)
    def covers(self,a,b):
        intervals=[]
        for r in self.roads:
            pad=r['width']/2-.36
            lo=[min(r['a'][i],r['b'][i])-pad for i in (0,1)];hi=[max(r['a'][i],r['b'][i])+pad for i in (0,1)]
            near,far=0.,1.
            for i in (0,1):
                d=b[i]-a[i]
                if abs(d)<1e-10:
                    if a[i]<lo[i] or a[i]>hi[i]:near,far=1.,0.;break
                else:
                    p,q=(lo[i]-a[i])/d,(hi[i]-a[i])/d
                    near=max(near,min(p,q));far=min(far,max(p,q))
            if near<=far:intervals.append((near,far))
        end=0.
        for lo,hi in sorted(intervals):
            if lo>end+1e-8:return False
            end=max(end,hi)
            if end>=1-1e-8:return True
        return False
    def route(self,start,target,clear,key):
        if not self.covers(start,start) or not self.covers(target,target):return None
        z=start[2];points=[[x,y,z] for x,y in self.nodes]
        if key not in self.cache:
            edges=[[] for _ in points]
            for i,a in enumerate(points):
                for j in range(i):
                    if clear(a,points[j]):
                        d=math.dist(a,points[j]);edges[i].append((j,d));edges[j].append((i,d))
            self.cache={key:edges}
        edges=[list(e) for e in self.cache[key]];n=len(points);points+=[list(start),list(target)];edges+=[[],[]]
        for i in (n,n+1):
            for j in range(i):
                if clear(points[i],points[j]):
                    d=math.dist(points[i],points[j]);edges[i].append((j,d));edges[j].append((i,d))
        costs={n:0.};parents={n:None};queue=[(0.,n)]
        while queue:
            cost,i=heapq.heappop(queue)
            if cost>costs[i]:continue
            if i==n+1:break
            for j,d in edges[i]:
                if cost+d<costs.get(j,math.inf):costs[j]=cost+d;parents[j]=i;heapq.heappush(queue,(cost+d,j))
        if n+1 not in parents:return None
        path=[];i=n+1
        while i!=n:path.append(points[i]);i=parents[i]
        return list(reversed(path))
