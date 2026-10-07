"""Log commanded vs measured speed, change vx_max on a schedule, plot on Ctrl+C.

Usage: python3 speed_logger.py   (start it, send a long straight goal, Ctrl+C after the robot stops)
"""
import csv
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterEvent, Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (seconds after the robot starts moving, vx_max to set)
SCHEDULE = [(3.0, 0.12), (8.0, 0.45)]
MOVING_THRESHOLD = 0.05


class SpeedLogger(Node):
    def __init__(self):
        super().__init__("speed_logger")
        self.t0 = self.get_clock().now()
        self.cmd, self.odom, self.vx_max = [], [], []
        self.t_moving = None
        self.pending = list(SCHEDULE)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_subscription(Odometry, "/odom", lambda m: self.odom.append((self.t(), m.twist.twist.linear.x)), 10)
        self.create_subscription(ParameterEvent, "/parameter_events", self.on_param, 10)
        self.set_client = self.create_client(SetParameters, "/controller_server/set_parameters")
        self.create_timer(0.1, self.run_schedule)

    def t(self):
        return (self.get_clock().now() - self.t0).nanoseconds * 1e-9

    def on_cmd(self, msg):
        self.cmd.append((self.t(), msg.linear.x))
        if self.t_moving is None and abs(msg.linear.x) > MOVING_THRESHOLD:
            self.t_moving = self.t()
            self.get_logger().info("Robot moving, schedule started")

    def run_schedule(self):
        if self.t_moving is None or not self.pending:
            return
        delay, value = self.pending[0]
        if self.t() - self.t_moving >= delay:
            self.pending.pop(0)
            self.set_vx_max(value)

    def set_vx_max(self, value):
        if not self.set_client.service_is_ready():
            self.get_logger().error("/controller_server/set_parameters not available")
            return
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name="FollowPath.vx_max",
            value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value))]
        future = self.set_client.call_async(req)
        future.add_done_callback(lambda f: self.on_set_done(f, value))

    def on_set_done(self, future, value):
        res = future.result().results[0]
        if res.successful:
            self.get_logger().info(f"Set vx_max = {value}: accepted")
        else:
            self.get_logger().error(f"Set vx_max = {value}: REJECTED ({res.reason})")

    def on_param(self, msg):
        if msg.node != "/controller_server":
            return
        for p in msg.changed_parameters:
            if p.name == "FollowPath.vx_max":
                self.vx_max.append((self.t(), p.value.double_value))

    def save(self):
        with open("speed_log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["topic", "t", "value"])
            for name, data in (("cmd_vel", self.cmd), ("odom", self.odom), ("vx_max", self.vx_max)):
                w.writerows((name, t, v) for t, v in data)
        plt.figure(figsize=(10, 4))
        for data, label in ((self.cmd, "/cmd_vel"), (self.odom, "/odom")):
            if data:
                plt.plot(*zip(*data), label=label)
        for t, v in self.vx_max:
            plt.axvline(t, color="k", ls="--", lw=0.8)
            plt.text(t, plt.ylim()[1], f" vx_max={v}", va="top", fontsize=8)
        plt.xlabel("t [s]"); plt.ylabel("linear.x [m/s]"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig("speed_log.png", dpi=120)
        print(f"Saved speed_log.csv, speed_log.png  (cmd={len(self.cmd)}, odom={len(self.odom)}, vx_max events={len(self.vx_max)})")


def main():
    rclpy.init()
    node = SpeedLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.save()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()