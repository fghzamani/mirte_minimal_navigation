#!/usr/bin/env python3
"""Publish the state of the running NavigateToPose goal as RViz markers.

Nav2 already publishes everything needed on the action's own topics, but
neither of them is renderable in RViz:

  /navigate_to_pose/_action/status    action_msgs/GoalStatusArray
  /navigate_to_pose/_action/feedback  nav2_msgs/NavigateToPose_FeedbackMessage

This node folds the two into one text marker that floats above the robot
(ACCEPTED / NAVIGATING / GOAL REACHED / CANCELED / FAILED, plus distance
remaining, ETA, elapsed time and recovery count) and one marker at the goal
pose coloured by the same state. Add a MarkerArray display on
/navigation_status in RViz and the outcome of every goal is visible there,
whoever sent the goal -- RViz, a script, or the tuner.

The goal pose is taken from the last pose of the planned path on /plan,
because the action goal itself travels over a service and never appears on
a topic.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.time import Time
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from action_msgs.msg import GoalStatus, GoalStatusArray
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


# label and RGB shown for each action_msgs/GoalStatus value
STATUS_STYLE = {
    GoalStatus.STATUS_UNKNOWN: ('NAV: IDLE', (0.7, 0.7, 0.7)),
    GoalStatus.STATUS_ACCEPTED: ('NAV: GOAL ACCEPTED', (0.4, 0.7, 1.0)),
    GoalStatus.STATUS_EXECUTING: ('NAV: NAVIGATING', (1.0, 0.85, 0.1)),
    GoalStatus.STATUS_CANCELING: ('NAV: CANCELING...', (1.0, 0.55, 0.0)),
    GoalStatus.STATUS_SUCCEEDED: ('NAV: GOAL REACHED', (0.1, 0.9, 0.2)),
    GoalStatus.STATUS_CANCELED: ('NAV: GOAL CANCELED', (1.0, 0.55, 0.0)),
    GoalStatus.STATUS_ABORTED: ('NAV: GOAL FAILED (aborted)', (1.0, 0.15, 0.15)),
}

TERMINAL_STATUSES = (
    GoalStatus.STATUS_SUCCEEDED,
    GoalStatus.STATUS_CANCELED,
    GoalStatus.STATUS_ABORTED,
)

ACTIVE_STATUSES = (
    GoalStatus.STATUS_ACCEPTED,
    GoalStatus.STATUS_EXECUTING,
    GoalStatus.STATUS_CANCELING,
)

# Matches the QoS the action server uses for its status topic; a plain
# default subscription would drop the latched last-known status.
ACTION_STATUS_QOS = QoSProfile(
    depth=1,
    history=QoSHistoryPolicy.KEEP_LAST,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class NavStatusMonitor(Node):

    def __init__(self):
        super().__init__('nav_status_monitor')

        self.declare_parameter('action_name', 'navigate_to_pose')
        self.declare_parameter('plan_topic', '/plan')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('marker_topic', '/navigation_status')
        # how long a finished goal keeps showing its result before going IDLE
        self.declare_parameter('result_hold_time', 8.0)
        self.declare_parameter('text_height', 0.7)
        self.declare_parameter('text_size', 0.16)

        action = self.get_parameter('action_name').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.global_frame = self.get_parameter('global_frame').value
        self.result_hold_time = float(self.get_parameter('result_hold_time').value)
        self.text_height = float(self.get_parameter('text_height').value)
        self.text_size = float(self.get_parameter('text_size').value)

        self.status = GoalStatus.STATUS_UNKNOWN
        self.goal_id = None
        self.feedback = None
        self.goal_pose = None
        self.result_stamp = None
        self.last_logged = None

        self.markers = self.create_publisher(
            MarkerArray, self.get_parameter('marker_topic').value, 1)

        self.create_subscription(
            GoalStatusArray, f'/{action}/_action/status',
            self.on_status, ACTION_STATUS_QOS)
        self.create_subscription(
            NavigateToPose.Impl.FeedbackMessage, f'/{action}/_action/feedback',
            self.on_feedback, 10)
        self.create_subscription(
            Path, self.get_parameter('plan_topic').value, self.on_plan, 1)

        self.create_timer(0.2, self.publish_markers)
        self.get_logger().info(
            f'Watching /{action} -- add a MarkerArray display on '
            f"{self.get_parameter('marker_topic').value} in RViz")

    # ---------------------------------------------------------------- inputs

    def on_status(self, msg):
        """Track the most recently issued goal in the status array."""
        newest = None
        newest_key = None
        for status in msg.status_list:
            stamp = Time.from_msg(status.goal_info.stamp).nanoseconds * 1e-9
            # an active goal always outranks a finished one, so a stale
            # SUCCEEDED entry cannot mask the goal that is running now
            key = (1 if status.status in ACTIVE_STATUSES else 0, stamp)
            if newest_key is None or key > newest_key:
                newest, newest_key = status, key

        if newest is None:
            return

        goal_id = bytes(newest.goal_info.goal_id.uuid)
        if goal_id != self.goal_id:
            # a new goal: drop the previous goal's feedback and result.
            # goal_pose is deliberately kept -- the plan for this goal may
            # already have arrived (planner_server publishes /plan before the
            # status update reaches us), and clearing it here would delete the
            # goal marker until the next replan cycle.
            self.goal_id = goal_id
            self.feedback = None
            self.result_stamp = None
            self.last_logged = None

        previous = self.status
        self.status = newest.status

        if self.status != previous:
            if self.status in TERMINAL_STATUSES:
                self.result_stamp = self.get_clock().now()
            self.log_transition()

    def on_feedback(self, msg):
        self.feedback = msg.feedback

    def on_plan(self, msg):
        if msg.poses:
            self.goal_pose = msg.poses[-1]

    def log_transition(self):
        if self.status == self.last_logged:
            return
        self.last_logged = self.status
        label = STATUS_STYLE.get(self.status, (f'NAV: status {self.status}', None))[0]
        detail = ''
        if self.status in TERMINAL_STATUSES and self.feedback is not None:
            detail = (f' after {self.elapsed():.1f} s, '
                      f'{self.feedback.number_of_recoveries} recoveries')
        if self.status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error(label + detail)
        elif self.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn(label + detail)
        else:
            self.get_logger().info(label + detail)

    # ---------------------------------------------------------------- output

    def elapsed(self):
        if self.feedback is None:
            return 0.0
        return Duration.from_msg(self.feedback.navigation_time).nanoseconds * 1e-9

    def eta(self):
        if self.feedback is None:
            return 0.0
        return Duration.from_msg(
            self.feedback.estimated_time_remaining).nanoseconds * 1e-9

    def effective_status(self):
        """STATUS_UNKNOWN once a finished goal's result has been shown long enough."""
        if self.status in TERMINAL_STATUSES and self.result_stamp is not None:
            age = (self.get_clock().now() - self.result_stamp).nanoseconds * 1e-9
            if age > self.result_hold_time:
                return GoalStatus.STATUS_UNKNOWN
        return self.status

    def status_text(self, status):
        label = STATUS_STYLE.get(status, (f'NAV: status {status}', None))[0]
        if status in ACTIVE_STATUSES and self.feedback is not None:
            return (f'{label}\n'
                    f'{self.feedback.distance_remaining:.2f} m left  |  '
                    f'ETA {self.eta():.0f} s\n'
                    f't {self.elapsed():.1f} s  |  '
                    f'{self.feedback.number_of_recoveries} recoveries')
        if status in TERMINAL_STATUSES and self.feedback is not None:
            return (f'{label}\n'
                    f'{self.elapsed():.1f} s  |  '
                    f'{self.feedback.number_of_recoveries} recoveries')
        return label

    def publish_markers(self):
        status = self.effective_status()
        rgb = STATUS_STYLE.get(status, (None, (1.0, 1.0, 1.0)))[1]
        now = self.get_clock().now().to_msg()

        text = Marker()
        text.header.frame_id = self.robot_frame
        text.header.stamp = now
        text.ns = 'nav_status'
        text.id = 0
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.z = self.text_height
        text.pose.orientation.w = 1.0
        text.scale.z = self.text_size
        text.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=1.0)
        text.text = self.status_text(status)

        array = MarkerArray(markers=[text])

        if self.goal_pose is not None and status != GoalStatus.STATUS_UNKNOWN:
            goal = Marker()
            goal.header.frame_id = self.global_frame
            goal.header.stamp = now
            goal.ns = 'nav_goal'
            goal.id = 0
            goal.type = Marker.CYLINDER
            goal.action = Marker.ADD
            goal.pose = self.goal_pose.pose
            goal.pose.position.z = 0.02
            goal.scale.x = 0.3
            goal.scale.y = 0.3
            goal.scale.z = 0.04
            goal.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=0.8)
            array.markers.append(goal)
        else:
            array.markers.append(self.delete_marker('nav_goal', 0))

        self.markers.publish(array)

    def delete_marker(self, ns, marker_id):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.ns = ns
        marker.id = marker_id
        marker.action = Marker.DELETE
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = NavStatusMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
