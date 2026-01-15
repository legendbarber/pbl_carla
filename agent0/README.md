# CARLA 실행 (Terminal 1)
bash carla/CarlaUE4.sh --ros2 -RenderOffScreen

# 맵/날씨 변경 (Terminal 2)
python3 carla_setup.py

# 시각화 프로그램 rviz 실행 (Terminal 3)
rviz2

# 자율 주행 차량 스폰 (Terminal 4) 
python3 koo.py -f lincoln.json

# 스폰 차량 수동 제어 (Terminal 5)
python3 teleop_key.py

# 테스트용 차량/보행자 생성 (Terminal 6) 
python3 obstacles.py

# Lidar-Camera 데이터 처리 (Terminal 7) 
python3 cfusion.py

# Lidar 기반 장애물 감지 정보 전달 (Terminal 8)
python3 lidar_clustering.py

# 센서 데이터 기반 장애물 회피 지역 경로 생성 (Terminal 9) 
python3 local_planner.py

# 전역/지역 경로 수신 후 차량 제어 (Terminal 10) 
python3 controller.py