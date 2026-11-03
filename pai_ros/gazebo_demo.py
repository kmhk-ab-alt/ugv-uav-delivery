"""Launch Gazebo, ROS 2, VLM perception and interchangeable dispatch policies.
--physical enables wheel/rotor dynamics and compliant payload grips.
--stub is an explicit network-free synthetic-image fixture.
"""
import argparse
import math
import sys
import uuid
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from .fleet import Fleet
from .scheduling import parse_priorities
from .gazebo_scene import build_world

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-perception',choices=('off','fixture','vlm'),default='off')
    p.add_argument('--site-events',choices=('on','off'),default='on')
    p.add_argument('--speed-scale',type=float,default=1.,help='multiply both cruise speeds; precise handling stays slow')
    p.add_argument('--ground-speed',type=float,default=2.4,help='UGV cruise speed in m/s')
    p.add_argument('--uav-speed',type=float,default=5.2,help='UAV cruise speed in m/s; precision descent remains slower')
    p.add_argument('--ground-count',type=int,default=2);p.add_argument('--uav-count',type=int,default=2)
    p.add_argument('--package-count',type=int,default=3);p.add_argument('--mode',choices=('relay','scout','joint'),default='relay')
    p.add_argument('--perception',choices=('vlm','pixels'),default='vlm');p.add_argument('--dispatch-policy',choices=('priority','deadline','random','learned'),default='priority');p.add_argument('--checkpoint',default='');p.add_argument('--adaptive-modes',action='store_true');p.add_argument('--disruptions',action='store_true');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--meeting-policy',choices=('off','fixed','eta','learned'));p.add_argument('--meeting-checkpoint',default='')
    p.add_argument('--city-layout',choices=('lanes','blocks'),default='lanes')
    p.add_argument('--city-seed',type=int,default=-1);p.add_argument('--city-family',choices=('train','shift'),default='train');p.add_argument('--local-map',action='store_true');p.add_argument('--randomize-orders',action='store_true')
    p.add_argument('--physical',action='store_true');p.add_argument('--priorities',default='');p.add_argument('--headless',action='store_true');p.add_argument('--stub',action='store_true')
    p.add_argument('--lockstep',action='store_true',help='advance physics only after measured feedback is consumed')
    p.add_argument('--real-time-factor',type=float,default=1.,help='Gazebo simulation seconds per wall second')
    p.add_argument('--operation-limit',type=float,default=0.,help='simulated service cutoff; active jobs then finish and return')
    p.add_argument('--budget-policy',choices=('learned','untrained','value','priority','fifo','random'),default='learned');p.add_argument('--budget-checkpoint',default='')
    p.add_argument('--showcase',action='store_true');p.add_argument('--showcase-hold',type=float,default=15.)
    p.add_argument('--model');p.add_argument('--base-url');p.add_argument('--timeout',type=float,default=600);p.add_argument('--record-dir',default='')
    args=p.parse_args()
    if not math.isfinite(args.speed_scale) or args.speed_scale<=0:p.error('--speed-scale must be finite and positive')
    args.ground_speed*=args.speed_scale;args.uav_speed*=args.speed_scale
    # The bundled Cyclone DDS avoids observed Fast DDS SHM/discovery stalls.
    # This applies to this launch only and preserves an explicit user choice.
    if sys.platform=='darwin':os.environ.setdefault('RMW_IMPLEMENTATION','rmw_cyclonedds_cpp')
    if not math.isfinite(args.real_time_factor) or not 0<args.real_time_factor<=2:p.error('--real-time-factor must be in (0,2]')
    if args.scene_perception!='off':
        if args.city_layout!='blocks':p.error('--scene-perception requires --city-layout blocks')
        if args.physical and not args.lockstep:p.error('--scene-perception with physics requires --lockstep')
        if args.scene_perception=='vlm':
            from .cargo_vlm import api_key_for,DEFAULT_BASE_URL
            api_key_for(args.base_url or DEFAULT_BASE_URL)
    if args.lockstep and not args.physical:p.error('--lockstep requires --physical')
    if args.city_layout=='blocks' and args.city_seed<0:p.error('--city-layout blocks requires --city-seed')
    if args.meeting_policy is None:args.meeting_policy='off' if args.adaptive_modes or args.mode!='relay' else 'eta'
    if args.meeting_policy!='off' and (args.adaptive_modes or args.mode!='relay'):p.error('Meeting selection requires --mode relay without --adaptive-modes')
    if args.meeting_policy!='off' and args.local_map:p.error('Meeting selection currently requires a provided map; remove --local-map')
    if args.meeting_policy=='learned' and not args.meeting_checkpoint:
        from ament_index_python.packages import get_package_share_directory
        args.meeting_checkpoint=str(Path(get_package_share_directory('pai_ros'))/'checkpoints/rendezvous/best.npz')
    if not math.isfinite(args.showcase_hold) or args.showcase_hold<0:p.error('--showcase-hold must be nonnegative')
    if args.timeout<=0 or not math.isfinite(args.timeout):p.error('--timeout must be finite and positive')
    if not math.isfinite(args.operation_limit) or args.operation_limit<0:p.error('--operation-limit must be nonnegative and finite')
    if args.operation_limit:
        if args.mode!='relay' or args.adaptive_modes:p.error('service budget requires relay without adaptive modes')
        if args.budget_policy=='learned' and not args.budget_checkpoint:
            from ament_index_python.packages import get_package_share_directory
            args.budget_checkpoint=str(Path(get_package_share_directory('pai_ros'))/('checkpoints/site_dispatch/best.npz' if args.scene_perception!='off' else 'checkpoints/budget_blocks/best.npz' if args.city_layout=='blocks' else 'checkpoints/budget/best.npz'))
    if not args.stub and args.perception=='vlm':
        from .cargo_vlm import api_key_for,DEFAULT_BASE_URL
        api_key_for(args.base_url or DEFAULT_BASE_URL)
    if args.dispatch_policy=='learned' and not args.checkpoint:
        from ament_index_python.packages import get_package_share_directory
        args.checkpoint=str(Path(get_package_share_directory('pai_ros'))/('checkpoints/city_generalization/diverse_cities.npz' if args.city_seed>=0 else 'checkpoints/native_dispatch_v3/best.npz'))
    fleet=Fleet(args.ground_count,args.uav_count,args.package_count,args.mode,seed=args.seed,randomize=args.randomize_orders,priorities=parse_priorities(args.priorities,args.package_count))
    fleet.configure_speeds(args.ground_speed,args.uav_speed)
    if args.city_seed>=0:
        from .city import generate_city
        generate_city(args.city_seed,args.package_count,args.city_family,args.city_layout).apply(fleet)
        if args.randomize_orders:
            from .city import configure_orders
            configure_orders(fleet,args.seed)
        print(f"Generated city seed={args.city_seed} geometry={fleet.city_manifest['geometry_sha256']} map={'local fixture with truth safety shield' if args.local_map else 'provided'}",flush=True)
    if args.operation_limit:
        from .service_budget import configure as budget_configure,orders
        budget_configure(fleet,args.operation_limit);orders(fleet,args.seed)
    if args.priorities:
        for j,priority in zip(fleet.jobs.values(),parse_priorities(args.priorities,args.package_count)):j.priority=priority
    if args.meeting_policy!='off':
        from .rendezvous import configure
        configure(fleet,args.meeting_policy,args.meeting_checkpoint)
    if args.local_map and args.city_seed<0:p.error('--local-map requires --city-seed')
    from gz.transport13 import Node
    from gz.msgs10.world_control_pb2 import WorldControl
    from gz.msgs10.boolean_pb2 import Boolean
    os.environ.setdefault('GZ_IP','127.0.0.1')
    # Isolate each launch, including from unrelated Gazebo instances.
    os.environ['GZ_PARTITION']=f'pai_delivery_{os.getpid()}'
    os.environ['ROS_DOMAIN_ID']=str(20+os.getpid()%60)
    log_root=Path(os.environ.get('ROS_HOME',str(Path.cwd()/'runtime/ros'))).parent/'gazebo_runs'
    run_dir=log_root/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    gui_log=run_dir/'gui.log'
    print(f'Gazebo logs: {run_dir}',flush=True)
    import json
    (run_dir/'run.json').write_text(json.dumps(dict(city=getattr(fleet,'city_manifest',None),arguments=vars(args),ros_domain_id=os.environ['ROS_DOMAIN_ID'],gazebo_partition=os.environ['GZ_PARTITION']),indent=2))
    children=[]
    def spawn(command,log_path=None,child_env=None):
        if log_path is None:
            child=subprocess.Popen(command,start_new_session=True,env=child_env)
        else:
            with log_path.open('w') as stream:
                child=subprocess.Popen(command,start_new_session=True,stdout=stream,stderr=subprocess.STDOUT,env=child_env)
        children.append(child)
        return child
    with tempfile.TemporaryDirectory(prefix='pai_gazebo_') as directory:
        world=Path(directory)/'delivery.sdf';world.write_text(build_world(fleet,render=not args.stub or args.scene_perception=='vlm',physical=args.physical,disruptions=args.disruptions,real_time_factor=args.real_time_factor))
        (run_dir/'delivery.sdf').write_text(world.read_text())
        try:
            server_flags=['--render-engine-server','ogre2']
            if sys.platform=='darwin':server_flags+=['--render-engine-server-api-backend','metal']
            server_env=None
            if sys.platform=='darwin' and not args.stub:
                # Avoid loading the unused GL3Plus subsystem on a Metal-only server.
                metal_plugins=run_dir/'metal_plugins';metal_plugins.mkdir()
                original=Path(os.environ['OGRE2_RESOURCE_PATH'])
                for library in ('RenderSystem_Metal.dylib','Plugin_ParticleFX.dylib'):
                    (metal_plugins/library).symlink_to(original/library)
                server_env=dict(os.environ,OGRE2_RESOURCE_PATH=str(metal_plugins))
            server=spawn(['gz','sim','-s',str(world),'-v','2']+server_flags,child_env=server_env)
            node=Node(); ready=False
            for _ in range(30):
                if server.poll() is not None:raise RuntimeError('Gazebo server exited during startup')
                ok,res=node.request('/world/pai_delivery/control',WorldControl(pause=True),WorldControl,Boolean,1000)
                if ok and res.data:ready=True;break
            if not ready:raise RuntimeError('Gazebo server did not become ready')
            bridge_args=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock', '/world/pai_delivery/set_pose@ros_gz_interfaces/srv/SetEntityPose']
            if args.physical:
                bridge_args += ['/pai/cargo/command@std_msgs/msg/String]gz.msgs.StringMsg']
                bridge_args += [f'/model/{name}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry' for name in fleet.jobs]
                for name,a in fleet.agents.items():
                    bridge_args += [f'/model/{name}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry']
                    topic=f'/model/{name}/cmd_vel' if a.kind=='ground' else f'/{name}/cmd_vel'
                    bridge_args += [f'{topic}@geometry_msgs/msg/Twist]gz.msgs.Twist']
                    if a.kind=='uav':bridge_args += [f'/{name}/enable@std_msgs/msg/Bool]gz.msgs.Boolean']
            if not args.stub:
                bridge_args += [f'/pai_gz/{j}/image@sensor_msgs/msg/Image[gz.msgs.Image' for j in fleet.jobs]
            if args.scene_perception=='vlm':
                from .site_perception import sites_for
                bridge_args += [f'/pai_gz/site_{name}/image@sensor_msgs/msg/Image[gz.msgs.Image' for name in sites_for(fleet.city)]
            bridge=spawn(['ros2','run','ros_gz_bridge','parameter_bridge']+bridge_args)
            gui_flags=['--render-engine-gui','ogre2']
            if sys.platform=='darwin':gui_flags+=['--render-engine-gui-api-backend','metal']
            gui=None if args.headless or args.showcase else spawn(['gz','sim','-g','-v','3']+gui_flags,gui_log)
            from ament_index_python.packages import get_package_share_directory
            config=str(Path(get_package_share_directory('pai_ros'))/'config/fleet.yaml')
            params=['--ros-args','-r','__ns:=/pai','--params-file',config]
            settings=dict(speed_scale=1.,scene_perception=args.scene_perception,site_events=str(args.site_events=='on').lower(),ground_speed=args.ground_speed,uav_speed=args.uav_speed,lockstep=str(args.lockstep).lower(),city_layout=args.city_layout,operation_limit=args.operation_limit,budget_policy=args.budget_policy,budget_checkpoint=args.budget_checkpoint,meeting_policy=args.meeting_policy,meeting_checkpoint=args.meeting_checkpoint,randomize=str(args.randomize_orders).lower(),city_seed=args.city_seed,city_family=args.city_family,local_map=str(args.local_map).lower(),pixel_test=str(args.perception=='pixels').lower(),dispatch_policy=args.dispatch_policy,checkpoint=args.checkpoint,adaptive_modes=str(args.adaptive_modes).lower(),disruptions=str(args.disruptions).lower(),seed=args.seed,physical=str(args.physical).lower(),backend='gazebo',headless='true',stub=str(args.stub).lower(),ground_count=args.ground_count,
                          uav_count=args.uav_count,package_count=args.package_count,mode=args.mode,mission_timeout=float(args.timeout))
            if args.model:settings['model']=args.model
            if args.base_url:settings['base_url']=args.base_url
            if args.priorities:settings['priorities']=args.priorities
            if args.record_dir:settings['record_dir']=args.record_dir
            for k,v in settings.items():
                if v!='':params+=['-p',f'{k}:={v}']
            presentation=None
            if args.showcase:
                ready_file=run_dir/'showcase.json'
                presentation=spawn([sys.executable,'-m','pai_ros.showcase_server','--scene',str(world),'--ready-file',str(ready_file)])
                for _ in range(600):
                    if ready_file.exists():break
                    if presentation.poll() is not None:raise RuntimeError('Presentation server failed')
                    time.sleep(.1)
                if not ready_file.exists():raise RuntimeError('Presentation server startup timeout')
                url=json.loads(ready_file.read_text())['url'];print('Presentation: '+url,flush=True)
                if not args.headless:
                    import webbrowser
                    webbrowser.open(url)
            planner=spawn(['ros2','run','pai_ros','cargo_planner']+params)
            simulation=spawn(['ros2','run','pai_ros','fleet_sim']+params)
            ok,res=node.request('/world/pai_delivery/control',WorldControl(pause=bool(args.lockstep)),WorldControl,Boolean,2000)
            if not ok or not res.data:raise RuntimeError('Gazebo could not start simulation')
            deadline=time.monotonic()+args.timeout+30
            while simulation.poll() is None:
                if time.monotonic()>deadline:raise RuntimeError('Demo timeout')
                if presentation is not None and presentation.poll() is not None:raise RuntimeError('Presentation server stopped')
                if server.poll() is not None or bridge.poll() is not None or planner.poll() is not None:raise RuntimeError('Gazebo server or ROS bridge stopped')
                if gui is not None and gui.poll() is not None:
                    tail='\n'.join(gui_log.read_text(errors='replace').splitlines()[-25:])
                    print(tail,file=sys.stderr)
                    raise RuntimeError(f'Gazebo GUI exited (status={gui.returncode}). Log: {gui_log}')
                time.sleep(.1)
            if simulation.returncode:raise RuntimeError('ROS fleet demo failed')
            if args.showcase and args.showcase_hold:
                print(f'Presentation complete. Holding the final view for {args.showcase_hold:g} seconds.',flush=True)
                time.sleep(args.showcase_hold)
        except KeyboardInterrupt:pass
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    try:os.killpg(child.pid,signal.SIGINT)
                    except ProcessLookupError:pass
            for child in reversed(children):
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()

if __name__=='__main__':main()
