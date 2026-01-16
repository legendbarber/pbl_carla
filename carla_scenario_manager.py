#!/usr/bin/env python3
"""A simple CARLA scenario script:
- waits a few seconds
- spawns a stopped vehicle ~25m ahead of ego (role_name='hero')
- keeps it for some seconds, then destroys it

This is meant to validate:
1) path following works
2) LiDAR-based safety stop triggers
"""
import argparse
import time
import carla


def get_ego(world, role_name="hero"):
    for a in world.get_actors().filter("vehicle.*"):
        if a.attributes.get("role_name") == role_name:
            return a
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--delay", type=float, default=10.0, help="Seconds before spawning obstacle")
    parser.add_argument("--ahead", type=float, default=25.0, help="Distance ahead of ego (m)")
    parser.add_argument("--duration", type=float, default=12.0, help="How long the obstacle stays (s)")
    args, _unknown = parser.parse_known_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    mp = world.get_map()
    bps = world.get_blueprint_library()

    ego = None
    t0 = time.time()
    while ego is None and time.time() - t0 < 30.0:
        ego = get_ego(world)
        time.sleep(0.2)

    if ego is None:
        raise RuntimeError("Ego vehicle (role_name='hero') not found. Start your bridge first.")

    print(f"[Scenario] Ego found: id={ego.id}")
    print(f"[Scenario] Waiting {args.delay:.1f}s...")
    time.sleep(args.delay)

    wp = mp.get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    nxt = wp.next(args.ahead)
    if not nxt:
        raise RuntimeError("Could not find a waypoint ahead.")
    tr = nxt[0].transform

    obstacle_bp = bps.filter("vehicle.tesla.model3")[0]
    obstacle_bp.set_attribute("role_name", "obstacle")
    obstacle = world.try_spawn_actor(obstacle_bp, tr)
    if obstacle is None:
        raise RuntimeError("Failed to spawn obstacle vehicle.")

    # Fully stop it
    obstacle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    print(f"[Scenario] Spawned obstacle vehicle id={obstacle.id} at ~{args.ahead:.1f}m ahead. Holding {args.duration:.1f}s...")
    time.sleep(args.duration)

    if obstacle.is_alive:
        obstacle.destroy()
    print("[Scenario] Obstacle destroyed. Scenario done.")


if __name__ == "__main__":
    main()
