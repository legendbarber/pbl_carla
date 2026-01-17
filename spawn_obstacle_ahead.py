#!/usr/bin/env python3
import argparse
import random
import signal
import sys
import time

import carla


def find_ego_vehicle(world: carla.World, role_name: str) -> carla.Vehicle:
    for v in world.get_actors().filter("vehicle.*"):
        if v.attributes.get("role_name") == role_name:
            return v
    return None


def list_blueprints(world: carla.World, pattern: str):
    lib = world.get_blueprint_library()
    bps = lib.filter(pattern)
    for bp in bps:
        print(bp.id)
    print(f"\nTotal: {len(bps)}")


def pick_blueprint(world: carla.World, bp_id: str | None, category: str, seed: int | None):
    lib = world.get_blueprint_library()
    rng = random.Random(seed)

    if bp_id:
        return lib.find(bp_id)

    if category == "vehicle":
        cands = lib.filter("vehicle.*")
    else:
        cands = lib.filter("static.prop.*")

    if not cands:
        raise RuntimeError(f"No blueprints for category={category}")
    return rng.choice(cands)


def try_spawn(world: carla.World, bp: carla.ActorBlueprint, tf: carla.Transform, attempts: int = 12):
    # 작은 좌우/높이 지터로 try_spawn 반복
    base = tf
    for i in range(attempts):
        lateral = (i % 5 - 2) * 0.5   # -1.0, -0.5, 0, 0.5, 1.0
        z = 0.15 + 0.05 * (i // 5)    # 0.15, 0.20, 0.25...
        t = carla.Transform(
            carla.Location(
                x=base.location.x,
                y=base.location.y + lateral,
                z=base.location.z + z,
            ),
            base.rotation,
        )
        actor = world.try_spawn_actor(bp, t)
        if actor is not None:
            return actor
    return None


def park_vehicle(actor: carla.Actor):
    # 차량이면 “서있는 장애물”로 만들기
    if isinstance(actor, carla.Vehicle):
        actor.set_autopilot(False)
        actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--role-name", default="base_link")
    ap.add_argument("--category", choices=["vehicle", "prop"], default="vehicle")
    ap.add_argument("--bp", default=None, help="Blueprint id (optional).")
    ap.add_argument("--distance", type=float, default=18.0, help="meters ahead of ego")
    ap.add_argument("--lateral", type=float, default=0.0, help="meters to the right(+)/left(-) in ego frame")
    ap.add_argument("--yaw-offset", type=float, default=0.0, help="deg added to ego yaw for obstacle rotation")
    ap.add_argument("--z-offset", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=1, help="number of obstacles")
    ap.add_argument("--spacing", type=float, default=4.0, help="extra forward spacing between obstacles")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--wait", type=float, default=0.0, help="seconds; 0 means wait until Ctrl+C")
    ap.add_argument("--list", default=None, help="List blueprints by pattern, then exit (e.g. 'vehicle.*' or 'static.prop.*').")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()

    if args.list:
        list_blueprints(world, args.list)
        return

    ego = find_ego_vehicle(world, args.role_name)
    if ego is None:
        print(f"[ERROR] Ego vehicle with role_name='{args.role_name}' not found.")
        print("        Existing vehicles role_name:")
        for v in world.get_actors().filter('vehicle.*'):
            rn = v.attributes.get('role_name')
            if rn:
                print(" ", rn, v.type_id)
        sys.exit(1)

    bp = pick_blueprint(world, args.bp, args.category, args.seed)

    spawned = []
    stop = False

    def _sigint(_sig, _frm):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    t0 = time.time()
    for i in range(args.n):
        ego_tf = ego.get_transform()
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()

        d = args.distance + i * args.spacing
        spawn_loc = ego_tf.location + fwd * d + right * args.lateral + carla.Location(z=args.z_offset)
        spawn_rot = carla.Rotation(
            pitch=ego_tf.rotation.pitch,
            yaw=ego_tf.rotation.yaw + args.yaw_offset,
            roll=ego_tf.rotation.roll,
        )
        spawn_tf = carla.Transform(spawn_loc, spawn_rot)

        actor = try_spawn(world, bp, spawn_tf)
        if actor is None:
            print(f"[WARN] Failed to spawn obstacle {i+1}/{args.n} with bp='{bp.id}'.")
            continue

        park_vehicle(actor)
        spawned.append(actor)
        print(f"[OK] Spawned {actor.type_id} id={actor.id} at ({spawn_loc.x:.2f}, {spawn_loc.y:.2f}, {spawn_loc.z:.2f})")

    if not spawned:
        print("[ERROR] No obstacles spawned.")
        return

    # CARLA가 synchronous tick이면, 다음 tick에서 보이기 시작할 수 있음 (네 sensor_setup3가 tick 돌리고 있으면 OK)
    while not stop:
        if args.wait > 0.0 and (time.time() - t0) >= args.wait:
            break
        time.sleep(0.1)

    print("\n[INFO] Cleaning up...")
    for a in spawned:
        try:
            a.destroy()
        except Exception:
            pass
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
