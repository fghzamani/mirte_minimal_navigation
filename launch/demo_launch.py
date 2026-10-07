"""Full Mirte demo stack: nav2 + risk node + online causal tuner.

The run this sets up:
  1. the robot starts at the initial pose from demo_params.yaml (default 0,0)
     with the arm in the CARRY pose, and the costmap footprint set to the
     measured carry polygon so the geometry matches the physical robot;
  2. the risk node publishes the 9-element risk state at 10 Hz;
  3. the tuner observes that risk and adapts the configuration (arm/footprint,
     speed limit, inflation, critic weights) to reach the goal without
     collision. It runs in envelope_only mode, so decisions come from the
     physics feasibility envelope A(R) -- no causal model required.

Every run records itself. One directory per run, under <ws>/logs/:

  logs/run_<timestamp>/
    bag/              rosbag2 (zstd): scan, tf, costmaps, plan, cmd_vel,
                      /risk_state and /tuner_decision -- the risk and the
                      config applied on that tick, paired, so a run can be
                      replayed and the models retrained offline
    node_logs/        one log file per node (ROS_LOG_DIR is pointed here), so
                      amcl / planner / controller / tuner are separate files
                      instead of interleaved on one terminal
    config/           snapshot of demo_params.yaml, the tuner config and the
                      map actually used -- parameters change between runs, so
                      a bag without them cannot be interpreted later
    run_info.txt      timestamp and what was recorded
    console.log       the full interleaved console stream (only when launched
                      via scripts/run_demo.sh, which tees it here)

Usage:
  ros2 launch mirte_navigation demo_launch.py
  ros2 launch mirte_navigation demo_launch.py record:=false     # no bag
  ros2 launch mirte_navigation demo_launch.py arm_to_carry:=false
  ros2 launch mirte_navigation demo_launch.py tuner:=false      # nav + risk only

scripts/run_demo.sh wraps this, adds the arm and DDS pre-flight checks, and
captures the console stream into the same run directory.
"""
import datetime
import os
import shutil

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            LogInfo, SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Arm joint order: shoulder_pan, shoulder_lift, elbow, wrist.
# Carry pose measured 2026-09-29 (see ARM_CONFIGS in online_tuner_node.py).
CARRY_JOINTS = "[1.15, -0.030, -1.27, 0.0]"   # achievable; see ARM_CONFIGS
ARM_TOPIC = "/mirte_master_arm_controller/joint_trajectory"

# Workspace root. Resolved the same way as the arm script below: with
# --symlink-install __file__ is the source tree and walking up four levels
# lands on the repo root; a plain install lands on install/, so fall back to an
# absolute path. NOT '~' -- HOME in the container is /root while the workspace
# is mounted at /home/forough/phd_projects.
_WS = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if not os.path.isdir(os.path.join(_WS, 'src')):
    _WS = '/home/forough/phd_projects/mirte_demo'

# Topics worth keeping. Enough to replay a run and to retrain the risk/causal
# models offline: the raw sensing, the full TF chain, what each costmap actually
# contained, what was planned and commanded, and the risk/decision pair.
BAG_TOPICS = [
    '/scan',
    '/scan_filtered',                    # what the costmaps mark from (self-filtered)
    '/odom',
    '/tf',
    '/tf_static',                        # transient_local; a late subscriber still gets it
    '/joint_states',                     # verifies which arm pose was really held
    '/map',                              # latched; one message is enough
    '/local_costmap/costmap_raw',        # what the risk node actually reads
    '/global_costmap/costmap',
    '/local_costmap/published_footprint',   # nav2's truth: what the costmap used
    '/global_costmap/published_footprint',
    '/tuner/target_footprint',            # the tuner's intent, for comparison
    '/amcl_pose',
    '/particle_cloud',
    '/risk_state',
    '/risk_state_diagnostics',
    '/tuner_decision',                   # the config applied on that tick, + the
                                         # r_min/r_width it was derived from
    '/rosout',                           # EVERY node's log messages, so the bag
                                         # carries the tuner's TUNER TICK lines,
                                         # planner/controller errors and warnings
                                         # timestamped alongside the data. node_logs/
                                         # has the same text, but only the bag keeps
                                         # it aligned with /scan, /tf and /cmd_vel.
    '/speed_limit',
    '/plan',
    '/cmd_vel',
    '/navigation_status',
    '/behavior_tree_log',
    '/navigate_to_pose/_action/status',
    '/navigate_to_pose/_action/feedback',
]


def _rel_demo_params():
    """Workspace-relative path to demo_params.yaml, found by glob.

    Hardcoding the package directory name broke everything when the directory
    was renamed; glob over src/*/params/ survives it.
    """
    import glob as _g
    hits = sorted(_g.glob(os.path.join(_WS, 'src', '*', 'params', 'demo_params.yaml')))
    return os.path.relpath(hits[0], _WS) if hits else \
        'src/mirte_minimal_navigation/params/demo_params.yaml'


def _prepare_run_dir():
    """Create this run's directory and snapshot the configs into it.

    Returns (run_dir, notes). Runs once, when the launch description is built,
    so every artefact of one run shares a single timestamped directory.

    run_demo.sh exports MIRTE_RUN_NAME so its console.log lands in the same
    directory as the bag; a bare `ros2 launch` just gets a timestamp.
    """
    run_name = os.environ.get('MIRTE_RUN_NAME') or \
        datetime.datetime.now().strftime('run_%Y%m%dT%H%M%S')
    run_dir = os.path.join(_WS, 'logs', run_name)
    cfg_dir = os.path.join(run_dir, 'config')
    os.makedirs(cfg_dir, exist_ok=True)

    notes = []
    # A bag is uninterpretable later without the parameters that produced it --
    # footprint, inflation and the envelope constants all change between runs.
    for rel in (_rel_demo_params(),
                'src/online_causal_tuner/config/default_tuner_config.yaml',
                'src/online_causal_tuner/models/envelope_constants.json'):
        src = os.path.join(_WS, rel)
        if os.path.exists(src):
            shutil.copy2(src, cfg_dir)
            notes.append(os.path.basename(src))

    # The map too: resolve the yaml_filename the params actually point at.
    params = os.path.join(_WS, _rel_demo_params())
    try:
        with open(params) as fh:
            for line in fh:
                line = line.split('#')[0].strip()
                if line.startswith('yaml_filename:'):
                    mp = line.split(':', 1)[1].strip().strip('"\'')
                    if not os.path.isabs(mp):
                        mp = os.path.join(_WS, mp)
                    if os.path.exists(mp):
                        shutil.copy2(mp, cfg_dir)
                        notes.append(os.path.basename(mp))
                        with open(mp) as mh:
                            for ml in mh:
                                if ml.strip().startswith('image:'):
                                    img = os.path.join(os.path.dirname(mp),
                                                       ml.split(':', 1)[1].strip())
                                    if os.path.exists(img):
                                        shutil.copy2(img, cfg_dir)
                                        notes.append(os.path.basename(img))
                    break
    except OSError:
        pass

    with open(os.path.join(run_dir, 'run_info.txt'), 'w') as fh:
        fh.write('run        : %s\n' % run_name)
        fh.write('started    : %s\n' % datetime.datetime.now().isoformat())
        fh.write('workspace  : %s\n' % _WS)
        fh.write('config kept: %s\n' % ', '.join(notes))
        fh.write('bag topics : %d\n' % len(BAG_TOPICS))
        fh.write('\nnode_logs/ holds one file per node (ROS_LOG_DIR).\n')
        fh.write('console.log exists only if launched via scripts/run_demo.sh.\n')
    return run_dir, notes


def generate_launch_description():
    nav_params = PathJoinSubstitution([
        FindPackageShare('mirte_navigation'), 'params', 'demo_params.yaml'])
    tuner_cfg = os.path.join(
        get_package_share_directory('online_causal_tuner'), 'config',
        'default_tuner_config.yaml')

    # Every demo run must start with the arm in the CARRY pose, so this is
    # verified rather than fire-and-forget. The trajectory controller reports
    # "Goal successfully reached!" whether or not the servos moved, so a bare
    # `ros2 topic pub --once` cannot tell a successful move from a stalled
    # shoulder_pan -- and the run would then begin from the wrong footprint
    # with nothing in the log saying so. set_arm_pose.sh commands the pose,
    # checks /joint_states against it, retries, and fails loudly.
    # The tuner additionally holds carry (startup_carry_hold) until the first
    # goal arrives, so the starting condition survives even if this is skipped.
    arm_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))),
        'scripts', 'set_arm_pose.sh')
    if not os.path.exists(arm_script):
        # Installed share layout (non-symlink install) walks up to install/, not
        # to the repo root. Fall back to an absolute path -- NOT '~', because
        # HOME inside the container is /root while the workspace is mounted at
        # /home/forough/phd_projects, so expanduser resolves to the wrong tree.
        arm_script = '/home/forough/phd_projects/mirte_demo/scripts/set_arm_pose.sh'
    arm_cmd = f"'{arm_script}' carry 4"

    # Locate the robot by matching /scan against the map, and hand the result to
    # AMCL. `set_initial_pose: true` pins AMCL to {0,0,0}; when the robot is not
    # standing there, its footprint lands inside mapped walls and EVERY goal is
    # refused with "Starting point in lethal space!" or "no valid path found" --
    # an error that names the planner and never the pose. Measured 2026-10-02:
    # AMCL was told (0, 0, 0 deg) while the robot was actually at
    # (0.42, 1.35, 142 deg); scan-to-map match 11% vs 95% at the true pose.
    # Placing the robot by hand to within a few degrees is not realistic, so this
    # removes the requirement.
    pose_script = os.path.join(os.path.dirname(arm_script), 'set_initial_pose_from_scan.py')

    run_dir, _cfg_kept = _prepare_run_dir()
    bag_dir = os.path.join(run_dir, 'bag')
    node_log_dir = os.path.join(run_dir, 'node_logs')

    return LaunchDescription([
        # Per-node log files land in this run's directory instead of the shared
        # ~/.ros/log, so "the log of the demo launch" is one directory per run
        # with amcl / planner_server / controller_server / online_causal_tuner
        # as separate files. Launch's own launch.log still goes to the default
        # location, because launch reads ROS_LOG_DIR before this action runs.
        SetEnvironmentVariable('ROS_LOG_DIR', node_log_dir),

        DeclareLaunchArgument('params_file', default_value=nav_params,
                              description='Nav2 parameters (carry footprint + omni)'),
        DeclareLaunchArgument('record', default_value='true',
                              description='Record a rosbag of this run into logs/<run>/bag'),
        DeclareLaunchArgument('auto_initial_pose', default_value='true',
                              description='Locate the robot by scan-matching the map and '
                                          'publish it to /initialpose once AMCL is up'),
        DeclareLaunchArgument('arm_to_carry', default_value='true',
                              description='Move the arm to the carry pose on startup'),
        DeclareLaunchArgument('tuner', default_value='true',
                              description='Run the online causal tuner'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Open RViz with the Mirte config'),

        LogInfo(msg='\033[1mDemo stack: nav2 (omni, carry footprint) + risk node + '
                    'tuner (envelope_only)\033[0m'),
        LogInfo(msg='Run directory: ' + run_dir),

        # Recording starts before the tuner exists, so --include-unpublished-topics
        # is required: without it rosbag2 silently drops every name in the list
        # that has no publisher yet (/tuner_decision, /plan, /speed_limit ...)
        # and the bag comes back missing exactly the topics worth recording.
        # zstd file compression keeps scan+costmap bags manageable.
        TimerAction(period=4.0, condition=IfCondition(LaunchConfiguration('record')),
                    actions=[
            LogInfo(msg='Recording bag -> ' + bag_dir),
            ExecuteProcess(
                cmd=['ros2', 'bag', 'record',
                     '-o', bag_dir,
                     '--include-unpublished-topics',
                     '--compression-mode', 'file',
                     '--compression-format', 'zstd',
                     '--max-cache-size', '50000000'] + BAG_TOPICS,
                output='log',
            ),
        ]),

        # 0b. Fix the initial pose once AMCL is active (it activates ~t=5s) and
        #     before the tuner starts reading risk at t=12s.
        TimerAction(period=7.0,
                    condition=IfCondition(LaunchConfiguration('auto_initial_pose')),
                    actions=[
            LogInfo(msg='Locating robot by scan-matching the map -> /initialpose'),
            ExecuteProcess(cmd=['python3', pose_script], output='screen'),
        ]),

        # 1. Nav2 (also starts nav_status_monitor)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('mirte_navigation'), 'launch',
                'minimal_navigation_launch.py'])),
            launch_arguments={'params_file': LaunchConfiguration('params_file')}.items(),
        ),

        # 1b. Robot self-filter: /scan -> /scan_filtered for the costmaps only
        #     (AMCL and the risk node keep the raw /scan). See the params file.
        Node(
            package='laser_filters',
            executable='scan_to_scan_filter_chain',
            name='scan_to_scan_filter_chain',
            output='log',
            parameters=[PathJoinSubstitution([
                FindPackageShare('mirte_navigation'), 'params',
                'scan_self_filter.yaml'])],
            remappings=[('scan', '/scan'), ('scan_filtered', '/scan_filtered')],
        ),

        # 2. Put the arm in the carry pose once the controllers are up, so the
        #    physical footprint matches the carry polygon in the costmap.
        TimerAction(
            period=8.0,
            condition=IfCondition(LaunchConfiguration('arm_to_carry')),
            actions=[
                LogInfo(msg='Moving arm to CARRY pose (footprint matches costmap)'),
                ExecuteProcess(cmd=['bash', '-lc', arm_cmd], output='screen'),
            ],
        ),

        # 2b. RViz. LIBGL_ALWAYS_SOFTWARE works around the container's
        #     "libGL error: failed to load driver: iris" on this machine.
        #     Needs `xhost +local:docker` on the laptop host once per session.
        TimerAction(period=3.0, condition=IfCondition(LaunchConfiguration('rviz')),
                    actions=[Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='log',
            arguments=['-d', PathJoinSubstitution([
                FindPackageShare('mirte_navigation'), 'mirte_rviz.rviz'])],
            additional_env={'LIBGL_ALWAYS_SOFTWARE': '1'},
        )]),

        # 3. Risk node. Topic overrides are required: the defaults target
        #    another robot (/scan_raw, /mobile_base_controller/odom).
        TimerAction(period=5.0, actions=[Node(
            package='rct_collector',
            executable='risk_state_node',
            name='risk_state_node',
            output='screen',
            parameters=[{
                'scan_topic': '/scan',
                'odom_topic': '/odom',
                'pose_topic': '/amcl_pose',
                'costmap_topic': '/local_costmap/costmap_raw',
                'path_topic': '/plan',
            }],
        )]),

        # 4. Tuner. Starts last so /risk_state already exists; envelope_only
        #    comes from the config, so no causal model is loaded.
        TimerAction(period=12.0, condition=IfCondition(LaunchConfiguration('tuner')),
                    actions=[
            LogInfo(msg='Starting online causal tuner (envelope_only)'),
            Node(
                package='online_causal_tuner',
                executable='online_tuner_node',
                name='online_causal_tuner',
                output='screen',
                parameters=[tuner_cfg],
            )]),
    ])
