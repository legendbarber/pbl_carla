# CARLA semantic tags (v0.9.14+ table in CARLA docs)
# Reference: CARLA "Sensors reference" -> "Semantic segmentation camera" tag table.

from dataclasses import dataclass

@dataclass(frozen=True)
class Tags:
    UNLABELED: int = 0
    ROADS: int = 1
    SIDEWALKS: int = 2
    BUILDING: int = 3
    WALL: int = 4
    FENCE: int = 5
    POLE: int = 6
    TRAFFIC_LIGHT: int = 7
    TRAFFIC_SIGN: int = 8
    VEGETATION: int = 9
    TERRAIN: int = 10
    SKY: int = 11
    PEDESTRIAN: int = 12
    RIDER: int = 13
    CAR: int = 14
    TRUCK: int = 15
    BUS: int = 16
    TRAIN: int = 17
    MOTORCYCLE: int = 18
    BICYCLE: int = 19
    STATIC: int = 20
    DYNAMIC: int = 21
    OTHER: int = 22
    WATER: int = 23
    ROADLINE: int = 24
    GROUND: int = 25
    BRIDGE: int = 26
    RAILTRACK: int = 27
    GUARDRAIL: int = 28

OBSTACLE_TAGS = {
    Tags.PEDESTRIAN,
    Tags.RIDER,
    Tags.CAR,
    Tags.TRUCK,
    Tags.BUS,
    Tags.TRAIN,
    Tags.MOTORCYCLE,
    Tags.BICYCLE,
    Tags.DYNAMIC,
}
