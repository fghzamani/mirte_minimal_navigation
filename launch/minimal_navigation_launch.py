from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.actions import LogInfo

def generate_launch_description():
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('mirte_navigation'),
                'params',
                'minimal_nav2_params.yaml'
            ])
        ),

        LogInfo(msg='\033[1mNote: mirte_navigation requires proper PID gains, especially an integrator term, in /opt/ros/humble/share/mirte_base_control/config/mirte_base_control.yaml\033[0m'),

        # TF: base_link → base_footprint
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
            output='screen',
        ),

        # TF: base_link → base_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_frame'],
            output='screen',
        ),

        # Relay /cmd_vel → /mirte_base_controller/cmd_vel
        Node(
            package='topic_tools',
            executable='relay',
            arguments=['/cmd_vel', '/mirte_base_controller/cmd_vel'],
            output='screen',
        ),

        # Relay /mirte_base_controller/odom → /odom
        Node(
            package='topic_tools',
            executable='relay',
            arguments=['/mirte_base_controller/odom', '/odom'],
            output='screen',
        ),

        # Map server
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[params_file],
        ),

        # AMCL
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[params_file],
        ),

        # Planner server
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params_file],
        ),

        # Controller server
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[params_file],
        ),

        # BT Navigator
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params_file],
            arguments=['--ros-args', '--log-level', 'bt_navigator:=debug'],
        ),

        # Recovery behaviours. Without this node the behaviour tree's Spin,
        # BackUp, DriveOnHeading and Wait calls have NO action server, so every
        # recovery fails instantly and the goal is aborted the moment the
        # footprint touches a lethal cell -- even when the robot has not
        # actually collided and could back off and retry. The "N recoveries"
        # reported by the status monitor were BT retries against nothing.
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[params_file],
        ),

        # Navigation status/feedback as RViz markers on /navigation_status
        Node(
            package='mirte_navigation',
            executable='nav_status_monitor',
            name='nav_status_monitor',
            output='screen',
        ),

        # Lifecycle manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'autostart': True,
                # Default is 4.0s. A transient DDS hiccup that delays one bond
                # heartbeat makes the manager declare a perfectly healthy node
                # DOWN and tear down the entire stack mid-navigation:
                #   CRITICAL FAILURE: SERVER map_server IS DOWN after not
                #   receiving a heartbeat for 4000 ms. Shutting down related nodes.
                # The robot then stops for good, and the nav logs show only the
                # downstream damage (stale map->odom, timed-out service calls),
                # never the cause. Be generous: a genuinely crashed node is
                # still caught, just 20s later.
                'bond_timeout': 20.0,
                'node_names': [
                    'map_server',
                    'amcl',
                    'planner_server',
                    'controller_server',
                    'behavior_server',
                    'bt_navigator'
                ]
            }]
        ),
    ])


