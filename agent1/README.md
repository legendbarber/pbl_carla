# Stage 1 Oracle Stack (CARLA + ROS2 Humble)
'''
python3 ros2_native.py -f hero_path_stack.json
python3 ros2_lidar_clustering.py
python3 ros2_odom_path_maker.py
python3 ros2_local_planner_v2.py --ros-args -p global_path_csv:=global_path.csv -p n_candidates:=9
python3 ros2_pure_pursuit_from_odom.py
'''