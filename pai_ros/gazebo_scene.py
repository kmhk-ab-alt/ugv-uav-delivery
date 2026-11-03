"""Portable SDF scene. Pose-driven fleet visualization, not flight dynamics.
Reuses authored neighborhood geometry without importing MuJoCo.
"""
import math
import xml.etree.ElementTree as ET
from .neighborhood import environment, parcel_geoms, box, cylinder

MATERIALS = {'road': '.17 .20 .23 1', 'lawn': '.27 .40 .27 1',
             'concrete': '.70 .72 .70 1', 'cardboard': '.66 .45 .26 1',
             'glass': '.18 .38 .48 1', 'metal': '.33 .39 .43 1'}

def visuals(fragment):
    out=[]
    for i,g in enumerate(ET.fromstring('<root>'+fragment+'</root>')):
        a=g.attrib; kind=a['type']; size=list(map(float,a['size'].split()))
        pos=list(map(float,a.get('pos','0 0 0').split())); rpy=a.get('euler','0 0 0')
        if kind=='box': geo='<box><size>'+' '.join(str(x*2) for x in size)+'</size></box>'
        elif kind=='plane': geo='<plane><normal>0 0 1</normal><size>100 100</size></plane>'
        elif kind=='sphere': geo=f'<sphere><radius>{size[0]}</radius></sphere>'
        elif kind=='ellipsoid': geo='<ellipsoid><radii>'+' '.join(map(str,size))+'</radii></ellipsoid>'
        elif kind in ('cylinder','capsule'):
            length=2*size[1] if len(size)>1 else .1
            if 'fromto' in a:
                v=list(map(float,a['fromto'].split())); d=[v[j+3]-v[j] for j in range(3)]
                length=math.sqrt(sum(x*x for x in d)); pos=[(v[j+3]+v[j])/2 for j in range(3)]
                rpy=f'0 {math.acos(d[2]/length)} {math.atan2(d[1],d[0])}'
            geo=f'<cylinder><radius>{size[0]}</radius><length>{length}</length></cylinder>'
        else: raise ValueError('unsupported shape '+kind)
        color=a.get('rgba',MATERIALS.get(a.get('material'),'.5 .5 .5 1'))
        pose=' '.join(map(str,pos))+' '+rpy
        out.append(f'<visual name="v{i}"><pose>{pose}</pose><geometry>{geo}</geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>')
    return ''.join(out)

def model(name,pos,fragment,sensor=''):
    return f'<model name="{name}"><static>true</static><pose>{" ".join(map(str,pos))} 0 0 0</pose><link name="body">{visuals(fragment)}{sensor}</link></model>'

def build_world(fleet,render=True,physical=False,disruptions=False,real_time_factor=1.):
    if not math.isfinite(real_time_factor) or not 0<real_time_factor<=2:raise ValueError('real_time_factor must be in (0,2]')
    city=getattr(fleet,'city',None)
    urban=getattr(city,'layout','lanes')=='blocks'
    center=sum(city.lanes)/len(city.lanes) if city else (len(fleet.jobs)-1)*2
    from .city import scenery
    from .presentation_assets import backdrop,closure_visual
    models=[model('neighborhood',[0,0,0],(scenery(fleet) if city else environment(fleet))+backdrop(fleet))]
    try:
        from ament_index_python.packages import get_package_share_directory
        from pathlib import Path
        ridge=Path(get_package_share_directory('pai_ros'))/'showcase/ridge.obj'
    except (ImportError,LookupError):
        from pathlib import Path
        ridge=Path(__file__).resolve().parent.parent/'showcase/ridge.obj'
    if urban:
        models.append(model('urban_layout',[0,0,0],''))
        for i,place in enumerate(city.places):models.append(model('poi_'+str(i),place['position'],''))
        for site in city.delivery_sites:models.append(model('delivery_site_'+site['id'],site['position'],''))
    if urban:
        from .site_perception import sites_for,road_obstacle
        from .physical_models import collision_box
        for name,p in sites_for(city).items():
            pos=[p[0],p[1],-3.]
            o=road_obstacle(name,p);hx=(o.high[0]-o.low[0])/2;hy=(o.high[1]-o.low[1])/2
            block=model('site_blocker_'+name,pos,box((0,0,0),(hx,hy,.4),'.9 .49 .13 1')+box((0,0,.42),(hx+.02,hy+.02,.025),'.20 .23 .25 1'))
            if physical:block=block.replace('<link name="body">','<link name="body">'+collision_box(f'{2*hx} {2*hy} .8'))
            models.append(block)
            if render:
                camera=f'<sensor name="site_camera" type="camera"><pose>0 0 7 0 1.57079632679 0</pose><always_on>true</always_on><update_rate>10</update_rate><topic>/pai_gz/site_{name}/image</topic><camera><horizontal_fov>.85</horizontal_fov><image><width>192</width><height>192</height><format>R8G8B8</format></image><clip><near>.02</near><far>20</far></clip></camera></sensor>'
                models.append(model('site_camera_'+name,p,'',camera))
    if ridge.exists() and not urban:
        models.append(f'<model name="background_ridge"><static>true</static><pose>{"-18 0 0 0 0 0" if urban else "0 0 0 0 0 0"}</pose><link name="ridge"><visual name="terrain"><geometry><mesh><uri>{ridge.as_uri()}</uri></mesh></geometry><material><ambient>.25 .34 .28 1</ambient><diffuse>.25 .34 .28 1</diffuse></material></visual></link></model>')
    if city:
        from .physical_models import collision_box
        for o in city.obstacles:
            pos=[(a+b)/2 for a,b in zip(o.low,o.high)];size=[b-a for a,b in zip(o.low,o.high)]
            visual='' if 'building' in o.name else box((0,0,0),[v/2 for v in size],'.94 .56 .15 1')
            m=model(o.name,pos,visual)
            if physical:m=m.replace('<link name="body">','<link name="body">'+collision_box(' '.join(map(str,size))))
            models.append(m)
    if physical:
        from .physical_models import ground_model,uav_model,floor_model,collision_box
        models.append(floor_model())
        from .fleet import COLORS
        for i,j in ([] if city else enumerate(fleet.jobs.values())):
            x=-3.-list(COLORS).index(j.color)*.8;y=j.source[1]+1.7
            m=model(f'building_collision_{i}',[x,y,1.1],'')
            models.append(m.replace('<link name="body">','<link name="body">'+collision_box('1.48 1.4 2.2')))
        y=next(iter(fleet.jobs.values())).source[1]
        if not city:
            m=model('trolley_collision',[1.55,y,.3],'')
            models.append(m.replace('<link name="body">','<link name="body">'+collision_box('.5 .4 .6')))
    if hasattr(fleet,'meeting'):
        from .physical_models import collision_box
        for o in fleet.rendezvous_obstacles:
            pos=[(a+b)/2 for a,b in zip(o.low,o.high)];size=[b-a for a,b in zip(o.low,o.high)]
            m=model(o.name,pos,closure_visual(o))
            if physical:m=m.replace('<link name="body">','<link name="body">'+collision_box(' '.join(map(str,size))))
            models.append(m)
        for j in fleet.jobs.values():models.append(model('meeting_'+j.name,[j.transfer[0],j.transfer[1],-1.],cylinder((0,0,0),.34,.012,'.95 .8 .15 1')))
    for a in fleet.agents.values():
        if physical:
            models.append(ground_model(a) if a.kind=='ground' else uav_model(a))
            continue
        if a.kind=='ground':
            g=box((0,0,0),(.26,.22,.10),'.02 .85 .95 1')+box((0,0,.18),(.18,.18,.025),'.55 .55 .6 1')
            for x in (-.18,.18):
                for y in (-.22,.22): g+=f'<geom type="sphere" pos="{x} {y} -.12" size=".09" rgba=".08 .08 .08 1"/>'
        else:
            g=box((0,0,0),(.14,.10,.045),'1 .16 .04 1')
            for x in (-.21,.21):
                for y in (-.21,.21):
                    g+=f'<geom type="capsule" fromto="0 0 0 {x} {y} .02" size=".025" rgba=".3 .35 .39 1"/>'
                    g+=cylinder((x,y,.02),.12,.01,'.25 .5 .8 1')
        models.append(model(a.name,a.pos,g))
    for j in fleet.jobs.values():
        # Fixed depot inspection station: the camera stays in the world when cargo moves.
        camera=f'''<sensor name="inspection" type="camera"><pose>0 0 .85 0 1.57079632679 0</pose><always_on>true</always_on><update_rate>5</update_rate><topic>/pai_gz/{j.name}/image</topic><camera><horizontal_fov>.6</horizontal_fov><image><width>192</width><height>192</height><format>R8G8B8</format></image><clip><near>.02</near><far>20</far></clip></camera></sensor>''' if render else ''
        if camera:models.append(model('inspection_'+j.name,j.source,'',camera))
        parcel=model(j.name,j.pos,parcel_geoms(j))
        if physical:
            from .physical_models import inertial,odometry,collision_box
            parcel=parcel.replace('<static>true</static>','<static>false</static>').replace('<link name="body">','<link name="body">'+inertial(j.weight_kg,.004,.004,.004)+collision_box('.22 .17 .16'))
            parcel=parcel.replace('</model>',odometry(j.name,frequency=10)+'</model>')
        models.append(parcel)
    if disruptions:
        y=next(iter(fleet.jobs.values())).source[1]
        barrier=model('temporary_roadwork',[.75,y,-2.],box((0,0,0),(.3,.35,.35),'.95 .55 .1 1'))
        if physical:barrier=barrier.replace('<link name="body">','<link name="body">'+collision_box('.6 .7 .7'))
        models.append(barrier)
        models.append(model('temporary_airspace',[-2.6,y-1.,-2.],box((0,0,0),(.2,.4,.015),'.95 .2 .1 .7')))
    plugins=''.join(f'<plugin filename="gz-sim-{filename}-system" name="gz::sim::systems::{name}"/>' for filename,name in [('physics','Physics'),('user-commands','UserCommands'),('scene-broadcaster','SceneBroadcaster')])
    if physical:plugins+='<plugin filename="pai-cargo-grip" name="pai::CargoGrip"/>'
    if render: plugins+='<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>'
    return f'''<?xml version="1.0"?><sdf version="1.9"><world name="pai_delivery">
<physics name="default" type="ignored"><max_step_size>{.002 if physical else .02}</max_step_size><real_time_factor>{real_time_factor}</real_time_factor></physics>{plugins}
<scene><ambient>.5 .5 .5 1</ambient><background>.65 .79 .9 1</background><shadows>true</shadows></scene>
<light name="sun" type="directional"><pose>0 0 10 0 0 0</pose><diffuse>.9 .85 .78 1</diffuse><direction>-.3 .2 -1</direction><cast_shadows>true</cast_shadows></light>
<gui fullscreen="false"><plugin filename="MinimalScene" name="3D View"><gz-gui><title>PAI Delivery</title><property type="bool" key="showTitleBar">false</property><property type="string" key="state">docked</property></gz-gui><engine>ogre2</engine><scene>scene</scene><ambient_light>.5 .5 .5</ambient_light><background_color>.65 .79 .9</background_color><camera_pose>14 {center-16} 16 0 .6 2.2</camera_pose></plugin><plugin filename="GzSceneManager" name="Scene Manager"/><plugin filename="InteractiveViewControl" name="Interactive view control"/><plugin filename="WorldControl" name="World control"><start_paused>false</start_paused><use_event>true</use_event></plugin><plugin filename="WorldStats" name="World stats"/></gui>
{''.join(models)}</world></sdf>'''
