#!/usr/bin/env python3
import carla
import time
import sys

# 설정
TOWN = "Town01"  # Town01~10, Town11, Town12, Town13, Town15
WEATHER = "ClearNoon"  # ClearNoon, CloudyNoon, WetCloudyNoon, SoftRainNoon, HardRainNoon, ClearSunset
HOST, PORT = "127.0.0.1", 2000

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)
    
    # 맵 변경
    print(f"Loading {TOWN}...")
    client.load_world(TOWN)
    
    # 맵 로딩 대기
    while True:
        try:
            if client.get_world().get_map().name.endswith(TOWN):
                break
        except RuntimeError:
            pass
        time.sleep(5)
    
    world = client.get_world()
    print(f"Success to load {TOWN}")
    
    # 날씨 변경
    print(f"Setting weather: {WEATHER}")
    if hasattr(carla.WeatherParameters, WEATHER):
        wp = getattr(carla.WeatherParameters, WEATHER)
        world.set_weather(wp)
        print(f"Success to set weather: {WEATHER}")
    else:
        print(f"Warning: '{WEATHER}' not found. Using default weather.")
        world.set_weather(carla.WeatherParameters.Default)
    
    # 스펙테이터 카메라 위치 설정 (위에서 내려다보기)
    spawns = world.get_map().get_spawn_points()
    if spawns:
        sp = spawns[0]
        loc = sp.location
        loc.z += 60.0
        rot = carla.Rotation(pitch=-90)
        world.get_spectator().set_transform(carla.Transform(loc, rot))
        print("Spectator camera positioned")
    
    print("\n=== Setup Complete ===")
    print(f"Map: {TOWN}")
    print(f"Weather: {WEATHER}")
    print("======================")

if __name__ == "__main__":
    main()