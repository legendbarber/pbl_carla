cd ~
cd carla
./CarlaUE4.sh --ros2


python3 sensor_setup2.py -f sensor2.json
python3 path_follower_pure_pursuit.py --csv global_path.csv --ros-args -p use_sim_time:=true
python3 carla_scenario_manager.py --delay 10 --ahead 25 --duration 12




python3 sensor_setup2.py -f sensor2.json

python3 global_path_loader.py --ros-args -p path_file:=global_path.csv -p publish_hz:=1.0 -p decimate:=1

python3 gnss_localizer_fixed_origin.py --ros-args -p path_file:=global_path.csv -p base_frame:=base_link -p broadcast_tf:=true

python3 global_guidance.py --ros-args -p lookahead_m:=12.0 -p search_window:=200

python3 global_speed_profile.py
python3 lane_follow_controller_stageE.py