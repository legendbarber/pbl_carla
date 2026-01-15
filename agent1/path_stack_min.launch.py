from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Local planner (v2)
        Node(
            package="your_pkg_name",
            executable="ros2_local_planner_v2.py",
            name="local_path_avoid_v2",
            output="screen",
            parameters=[{
                "global_path_csv": "global_path.csv",
                "n_candidates": 9,
                "max_offset": 3.0,
                "L": 25.0,
            }],
        ),
        # Controller (ODOM-based pure pursuit)
        Node(
            package="your_pkg_name",
            executable="ros2_pure_pursuit_from_odom.py",
            name="pure_pursuit_from_odom",
            output="screen",
            parameters=[{
                "lookahead": 6.0,
                "base_speed": 6.0,
            }],
        ),
    ])
