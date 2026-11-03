import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    defaults = {'menagerie_dir': os.environ.get('PAI_MENAGERIE_DIR', ''),
                'stub': 'false', 'headless': 'true', 'rounds': '8',
                'model': 'qwen2.5vl:3b', 'base_url': 'http://localhost:11434/v1',
                'decision_timeout': '45.0'}
    args = [DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()]
    val = lambda k, t: ParameterValue(LaunchConfiguration(k), value_type=t)
    sim = Node(package='pai_ros', executable='sim', namespace='pai', output='screen',
               parameters=[{k: val(k, t) for k, t in [('menagerie_dir', str),
                   ('stub', bool), ('headless', bool), ('rounds', int), ('decision_timeout', float)]}])
    planner = Node(package='pai_ros', executable='planner', namespace='pai', output='screen',
                   parameters=[{k: val(k, t) for k, t in [('stub', bool), ('model', str), ('base_url', str)]}])
    stop = RegisterEventHandler(OnProcessExit(target_action=sim,
             on_exit=[EmitEvent(event=Shutdown(reason='simulation finished'))]))
    return LaunchDescription(args + [stop, planner, sim])
