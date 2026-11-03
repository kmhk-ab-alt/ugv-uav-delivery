"""Config file supplies defaults; explicitly supplied launch args override it."""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler, OpaqueFunction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

TYPES = dict(speed_scale=float,scene_perception=str,site_events=bool,ground_speed=float,uav_speed=float,lockstep=bool,city_layout=str,operation_limit=float,budget_policy=str,budget_checkpoint=str,meeting_policy=str,meeting_checkpoint=str,city_seed=int,city_family=str,local_map=bool,physical=bool,dispatch_policy=str,checkpoint=str,adaptive_modes=bool,disruptions=bool,pixel_test=bool,backend=str, ground_count=int, uav_count=int, package_count=int, mode=str, headless=bool,
             stub=bool, priorities=str, seed=int, randomize=bool, record_dir=str, model=str, base_url=str, mission_timeout=float)

def start(context):
    overrides = {}
    for name, kind in TYPES.items():
        value = LaunchConfiguration(name).perform(context)
        if value == '':
            continue
        if kind is bool:
            if value.lower() not in ('true', 'false'):
                raise ValueError(name + ' must be true or false')
            overrides[name] = value.lower() == 'true'
        else:
            overrides[name] = kind(value)
    params = [LaunchConfiguration('config').perform(context), overrides]
    sim = Node(package='pai_ros', executable='fleet_sim', namespace='pai', output='screen', parameters=params)
    planner = Node(package='pai_ros', executable='cargo_planner', namespace='pai', output='screen', parameters=params)
    stop_sim = RegisterEventHandler(OnProcessExit(target_action=sim,
        on_exit=[EmitEvent(event=Shutdown(reason='fleet simulation stopped'))]))
    stop_planner = RegisterEventHandler(OnProcessExit(target_action=planner,
        on_exit=[EmitEvent(event=Shutdown(reason='cargo planner stopped'))]))
    return [stop_sim, stop_planner, planner, sim]

def generate_launch_description():
    config = os.path.join(get_package_share_directory('pai_ros'), 'config', 'fleet.yaml')
    return LaunchDescription([DeclareLaunchArgument('config', default_value=config)] +
        [DeclareLaunchArgument(name, default_value='') for name in TYPES] + [OpaqueFunction(function=start)])
