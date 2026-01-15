#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

class GlobalPathFromOdom(Node):
    def __init__(self):
        super().__init__("global_path_from_odom")

        self.sub = self.create_subscription(
            Odometry, "/carla/hero/odometry", self.odom_cb, 10
        )
        self.pub = self.create_publisher(Path, "/carla/path/global", 10)

        self.declare_parameter("min_dist", 0.5)              # [m]
        self.declare_parameter("frame_id", "map")            # RViz 고정 프레임
        self.declare_parameter("csv_out", "global_path.csv") # 로컬플래너 입력

        self.min_dist = float(self.get_parameter("min_dist").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.csv_out  = str(self.get_parameter("csv_out").value)

        self.path_xy = []  # [(x,y)]
        self.timer = self.create_timer(0.2, self.publish_path)

    def odom_cb(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        if not self.path_xy:
            self.path_xy.append((x, y))
            return

        lx, ly = self.path_xy[-1]
        if math.hypot(x - lx, y - ly) >= self.min_dist:
            self.path_xy.append((x, y))

    def publish_path(self):
        if not self.path_xy:
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.frame_id
        for (x, y) in self.path_xy:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0
            path.poses.append(ps)
        self.pub.publish(path)

    def save_csv(self):
        if not self.path_xy:
            self.get_logger().warn("No path to save.")
            return
        try:
            with open(self.csv_out, "w") as f:
                f.write("# x[m],y[m]\n")
                for x, y in self.path_xy:
                    f.write(f"{x},{y}\n")
            self.get_logger().info(f"Saved global path to {self.csv_out}")
        except Exception as e:
            self.get_logger().error(f"Failed to save path: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = GlobalPathFromOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.save_csv()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
