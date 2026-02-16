from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    localize_config = os.path.join(
        get_package_share_directory('particle_filter'),
        'config',
        'params.yaml'
    )

    # Declare the argument (so you can swap config files if needed)
    localize_la = DeclareLaunchArgument(
        'localize_config',
        default_value=localize_config,
        description='Localization configs')

    # ONLY start the particle filter node
    pf_node = Node(
        package='particle_filter',
        executable='particle_filter', # Ensure this matches setup.py
        name='particle_filter',
        parameters=[LaunchConfiguration('localize_config')],
        output='screen'
    )

    return LaunchDescription([
        localize_la,
        pf_node
    ])