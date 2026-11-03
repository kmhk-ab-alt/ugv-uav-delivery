"""Loopback-only live 3D presentation of the ROS fleet state and authored SDF.
The viewer never commands vehicles or supplies synthetic mission trajectories.
"""
import argparse,json,math,threading,time
import xml.etree.ElementTree as ET
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer


def scene_document(path):
    world=ET.parse(path).getroot().find('world');models=[]
    def numbers(text,default):return [float(v) for v in text.split()] if text else default
    for model in world.findall('model'):
        links=[]
        for link in model.findall('link'):
            objects=[]
            for visual in link.findall('visual'):
                geo=visual.find('geometry');shape=list(geo)[0]
                if shape.tag=='box':scale=numbers(shape.findtext('size'),[1,1,1]);kind='box'
                elif shape.tag=='plane':scale=numbers(shape.findtext('size'),[100,100])+[1];kind='plane'
                elif shape.tag=='cylinder':r=float(shape.findtext('radius'));scale=[r,r,float(shape.findtext('length'))];kind='cylinder'
                elif shape.tag=='sphere':r=float(shape.findtext('radius'));scale=[r,r,r];kind='sphere'
                elif shape.tag=='ellipsoid':scale=numbers(shape.findtext('radii'),[1,1,1]);kind='sphere'
                elif shape.tag=='mesh':
                    from urllib.parse import urlparse,unquote
                    mesh_path=Path(unquote(urlparse(shape.findtext('uri')).path))
                    vertices=[];triangles=[]
                    for line in mesh_path.read_text().splitlines():
                        fields=line.split()
                        if fields and fields[0]=='v':vertices.append([float(v) for v in fields[1:4]])
                        elif fields and fields[0]=='f':
                            face=[int(v.split('/')[0])-1 for v in fields[1:]]
                            for i in range(1,len(face)-1):triangles.extend(vertices[k] for k in (face[0],face[i],face[i+1]))
                    objects.append(dict(kind='mesh',vertices=triangles,scale=[1,1,1],pose=numbers(visual.findtext('pose'),[0]*6),color=numbers(visual.findtext('material/diffuse'),[.3,.4,.3,1])))
                    continue
                else:continue
                objects.append(dict(kind=kind,scale=scale,pose=numbers(visual.findtext('pose'),[0]*6),color=numbers(visual.findtext('material/diffuse'),[.5,.5,.5,1])))
            if objects:links.append(dict(name=link.attrib['name'],pose=numbers(link.findtext('pose'),[0]*6),objects=objects))
        if links:models.append(dict(name=model.attrib['name'],pose=numbers(model.findtext('pose'),[0]*6),links=links,dynamic=model.findtext('static')!='true'))
    positions=[m['pose'] for m in models if m['name'].startswith('package_')]
    center=sum(p[1] for p in positions)/len(positions) if positions else 4
    urban=world.find("model[@name='urban_layout']") is not None
    names=['物流拠点','商店街','中央公園','住宅街','山間の配送地区','集荷センター A','集荷センター B','合流地点 A','合流地点 B']
    places=[dict(name=names[int(m.attrib['name'].split('_')[-1])],position=numbers(m.findtext('pose'),[0]*6)[:3]) for m in world.findall('model') if m.attrib['name'].startswith('poi_')]
    return dict(models=models,center_y=center,mission_count=len(positions),layout='blocks' if urban else 'lanes',places=places,delivery_sites=[dict(id=m.attrib['name'].removeprefix('delivery_site_'),position=numbers(m.findtext('pose'),[0]*6)[:3]) for m in world.findall('model') if m.attrib['name'].startswith('delivery_site_')],pickup_sites=[[21.55,8,.08],[21.55,16,.08]] if urban else [],meeting_sites=[[1,4,.08],[1,12,.08]] if urban else [],bounds=[-17,29,-7,max(28,max((m['pose'][1]+3 for m in models if m['name'].startswith(('ground_','uav_'))),default=23))] if urban else None)

class LiveData:
    def __init__(self,preview=False):self.lock=threading.Lock();self.state=None;self.events=[];self.received=0.;self.preview=preview
    def update_state(self,state):
        with self.lock:self.state=state;self.received=time.monotonic()
    def update_event(self,event):
        with self.lock:self.events=(self.events+[event])[-24:]
    def snapshot(self):
        with self.lock:return dict(state=self.state,events=self.events,preview=self.preview,age_s=time.monotonic()-self.received if self.received else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--ready-file',required=True);p.add_argument('--preview',action='store_true');p.add_argument('--assets')
    args=p.parse_args();data=LiveData(args.preview);doc=scene_document(args.scene)
    if args.assets:assets=Path(args.assets)
    else:
        from ament_index_python.packages import get_package_share_directory
        assets=Path(get_package_share_directory('pai_ros'))/'showcase'
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(assets),**kw)
        def log_message(self,*a):pass
        def do_GET(self):
            path=self.path.split('?',1)[0]
            if path in ('/api/scene','/api/live','/api/benchmark'):
                payload=doc if path=='/api/scene' else data.snapshot()
                if path=='/api/benchmark':
                    benchmark=assets.parent/('checkpoints/budget_blocks/benchmark.json' if doc['layout']=='blocks' else 'checkpoints/budget/benchmark.json')
                    payload=json.loads(benchmark.read_text()) if benchmark.exists() else None
                body=json.dumps(payload).encode()
                self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            else:super().do_GET()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    # Live mode subscribes to the launch's isolated ROS domain; preview has no fabricated state.
    if not args.preview:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        rclpy.init();node=Node('pai_showcase')
        node.create_subscription(String,'/pai/fleet/state',lambda m:data.update_state(json.loads(m.data)),10)
        node.create_subscription(String,'/pai/fleet/events',lambda m:data.update_event(json.loads(m.data)),100)
        threading.Thread(target=lambda:rclpy.spin(node),daemon=True).start()
    url=f'http://127.0.0.1:{server.server_port}/'
    Path(args.ready_file).write_text(json.dumps(dict(url=url,preview=args.preview)))
    print('Live presentation: '+url,flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
