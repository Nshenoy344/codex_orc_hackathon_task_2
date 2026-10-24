from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default = os.path.join(get_package_share_directory('franka_realsense_handeye'),
                           'config', 'calibration.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default),
        Node(package='franka_realsense_handeye', executable='calibrate',
             name='handeye_calibrator', output='screen',
             parameters=[LaunchConfiguration('params_file')])])
