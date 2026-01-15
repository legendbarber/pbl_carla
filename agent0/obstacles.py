#!/usr/bin/env python3
import glob
import os
import sys
import time
import random
import logging
import math

# ==============================================================================
# CARLA 모듈 경로 설정
# ==============================================================================
try:
    sys.path.append(glob.glob('../../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import VehicleLightState as vls
from carla.command import SpawnActor, SetAutopilot, FutureActor, DestroyActor

def main():
    # ==============================================================================
    # [설정] 고정 장애물 삭제 & 랜덤 교통량 설정
    # ==============================================================================
    
    # 1. 고정 장애물 리스트를 모두 비웁니다 (삭제)
    FIXED_VEHICLES = {} 
    FIXED_CONES = {} 
    
    # 2. 목표 총 개수 (전부 움직이는 객체로 채움)
    TOTAL_VEHICLES_TARGET = 50   
    TOTAL_WALKERS_TARGET = 50    
    TM_PORT = 8005    
    # ==============================================================================

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    synchronous_master = False

    vehicles_list = []
    walkers_list = []
    all_id = []
    
    try:
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        
        # Traffic Manager 설정
        traffic_manager = client.get_trafficmanager(TM_PORT)
        traffic_manager.set_global_distance_to_leading_vehicle(5.0)
        traffic_manager.global_percentage_speed_difference(50.0)
        
        settings = world.get_settings()
        if settings.synchronous_mode:
            traffic_manager.set_synchronous_mode(True)
            synchronous_master = True

        # ------------------------------------------------------------------
        # 1. 고정 장애물 스폰 (리스트가 비어있으므로 실행되지 않음)
        # ------------------------------------------------------------------
        print("--- No Fixed Obstacles (Skipped) ---")
        used_indices = []

        def apply_offset(transform, f_off, r_off):
            yaw = math.radians(transform.rotation.yaw)
            fx, fy = math.cos(yaw), math.sin(yaw)
            rx, ry = math.sin(yaw), -math.cos(yaw)
            transform.location.x += (fx * f_off) + (rx * r_off)
            transform.location.y += (fy * f_off) + (ry * r_off)
            return transform

        def spawn_fixed(targets, filter_pattern, type_name):
            if not targets: return
            blueprints = bp_lib.filter(filter_pattern)
            if type_name == "Vehicle":
                blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

            for idx, (f_off, r_off) in targets.items():
                if idx >= len(spawn_points): continue
                used_indices.append(idx)
                
                tf = carla.Transform(spawn_points[idx].location, spawn_points[idx].rotation)
                tf = apply_offset(tf, f_off, r_off)
                bp = random.choice(blueprints)
                if bp.has_attribute('color'):
                    bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))

                actor = world.try_spawn_actor(bp, tf)
                if actor:
                    actor.set_simulate_physics(True)
                    if type_name == "Vehicle":
                        actor.set_autopilot(False, TM_PORT)
                        actor.apply_control(carla.VehicleControl(hand_brake=True))
                    vehicles_list.append(actor.id)
                    print(f"  [{type_name}] Spawned at Index {idx}")

        spawn_fixed(FIXED_VEHICLES, 'vehicle.*', "Vehicle")
        spawn_fixed(FIXED_CONES, 'static.prop.constructioncone', "Cone")

        # ------------------------------------------------------------------
        # 2. 랜덤 차량 50대 스폰 (Moving Vehicles)
        # ------------------------------------------------------------------
        # 고정 차량이 0대이므로, 50대 전부 랜덤으로 스폰됩니다.
        n_random_vehicles = TOTAL_VEHICLES_TARGET
        print(f"--- Spawning {n_random_vehicles} Random Vehicles ---")
        
        available_spawn_points = [sp for i, sp in enumerate(spawn_points) if i != 0]
        random.shuffle(available_spawn_points)

        vehicle_bps = bp_lib.filter('vehicle.*')
        vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]

        batch = []
        for i, transform in enumerate(available_spawn_points):
            if i >= n_random_vehicles: break
            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
            
            bp.set_attribute('role_name', 'autopilot')
            
            # Autopilot ON으로 스폰
            batch.append(SpawnActor(bp, transform)
                .then(SetAutopilot(FutureActor, True, TM_PORT)))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if not response.error:
                vehicles_list.append(response.actor_id)

        # [안전 주행 설정] - 모든 차량이 움직여야 함
        print("Applying Safe-Driving settings...")
        all_vehicles = world.get_actors(vehicles_list)
        for actor in all_vehicles:
            if not actor.attributes.get('role_name') == 'autopilot': continue
            try:
                # 차선 변경 금지 & 안전 거리 확보 & 물리 엔진 ON
                traffic_manager.auto_lane_change(actor, False)
                traffic_manager.random_left_lanechange_percentage(actor, 0.0)
                traffic_manager.random_right_lanechange_percentage(actor, 0.0)
                traffic_manager.distance_to_leading_vehicle(actor, 5.0)
                traffic_manager.vehicle_percentage_speed_difference(actor, 50.0)
                traffic_manager.ignore_lights_percentage(actor, 0.0)
                traffic_manager.ignore_signs_percentage(actor, 0.0)
                try: traffic_manager.keep_right_rule_percentage(actor, 0.0)
                except: pass
                
                # 움직여야 하므로 물리 엔진 켬
                actor.set_simulate_physics(True)
            except: pass

        # ------------------------------------------------------------------
        # 3. 보행자 50명 스폰 (Walkers)
        # ------------------------------------------------------------------
        print(f"--- Spawning {TOTAL_WALKERS_TARGET} Walkers ---")
        
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        controller_bp = bp_lib.find('controller.ai.walker')
        
        spawned_walkers = 0
        max_retries = TOTAL_WALKERS_TARGET * 2
        retry_count = 0

        while spawned_walkers < TOTAL_WALKERS_TARGET and retry_count < max_retries:
            loc = world.get_random_location_from_navigation()
            if loc:
                spawn_transform = carla.Transform(loc)
                walker_bp = random.choice(walker_bps)
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                
                walker = world.try_spawn_actor(walker_bp, spawn_transform)
                if walker:
                    controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
                    if controller:
                        walkers_list.append({"id": walker.id, "con": controller.id})
                        spawned_walkers += 1
                    else:
                        walker.destroy()
                
            retry_count += 1

        for item in walkers_list:
            all_id.append(item["con"])
            all_id.append(item["id"])

        all_actors = world.get_actors(all_id)
        
        if not args.asynch and synchronous_master:
            world.tick()
        else:
            world.wait_for_tick()

        # 걷기 시작
        world.set_pedestrians_cross_factor(0.0)
        for i in range(0, len(all_id), 2):
            all_actors[i].start()
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            all_actors[i].set_max_speed(1.4)

        print(f"Done. (Vehicles: {len(vehicles_list)}, Walkers: {len(walkers_list)})")
        print("Press Ctrl+C to exit and destroy actors.")

        # ------------------------------------------------------------------
        # 4. 루프 유지
        # ------------------------------------------------------------------
        while True:
            if not args.asynch and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

    except KeyboardInterrupt:
        print('\nCancelled by user. Destroying actors...')
    finally:
        if synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        client.apply_batch([DestroyActor(x) for x in vehicles_list])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()
        client.apply_batch([DestroyActor(x) for x in all_id])
        
        time.sleep(0.5)
        print('Cleanup done.')

if __name__ == '__main__':
    class Args:
        asynch = False
    args = Args()
    
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
