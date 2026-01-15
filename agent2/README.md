# Stage 1 Oracle Stack (CARLA + ROS2 Humble)

목표: **“주행이 되게끔만”** — 시맨틱 세그(도로) + 시맨틱 LiDAR(장애물)만으로 차로 유지/정지까지 돌아가는 최소 스택.

## 구성
- `config/stage1_oracle_sensors.json` : 차량/월드/센서 정의(ON/OFF 가능)
- `scripts/stage1_bridge.py` : CARLA 연결/스폰/센서 -> ROS2 publish, `/carla/hero/cmd_vel` 구독 후 제어, `/clock` publish
- `scripts/stage1_oracle_autonomy.py` : 시맨틱 세그 + 시맨틱 LiDAR로 조향/속도 계산 후 `/carla/hero/cmd_vel` publish

## 실행
터미널 1:
```bash
# CARLA 서버 실행 (별도 터미널/프로세스)
./CarlaUE4.sh -quality-level=Epic
```

터미널 2:
```bash
source /opt/ros/humble/setup.bash
cd stage1_oracle_stack
python3 scripts/stage1_bridge.py --config config/stage1_oracle_sensors.json
```

터미널 3:
```bash
source /opt/ros/humble/setup.bash
cd stage1_oracle_stack
python3 scripts/stage1_oracle_autonomy.py
```

## 토픽
- semantic seg raw: `/carla/hero/sem_front/image_raw`
- semantic lidar: `/carla/hero/sem_lidar/points`
- control: `/carla/hero/cmd_vel`
