"""路 B：map-less Nav2 跟人栈 + 跟人目标桥。

起：controller_server(内含 local_costmap) + planner_server(内含 global_costmap)
   + behavior_server + bt_navigator + lifecycle_manager(autostart) + person_goal_bridge。

一条命令起全套（默认 bringup:=true run_detector:=true）：
   ros2 launch yahboomcar_astra follow_nav2.launch.py
   即含 bringup(odom+TF) + nav2 + 桥 + objTracker_tpu 检测器。别再单独起 bringup/检测器。
   若 bringup 已单独在跑，加 bringup:=false 避免节点重名。

注意：/scan 的 frame_id 是 `laser_frame`，但 bringup 的 URDF 里没有它（只有 radar_Link），
所以本 launch 自补一个静态 TF base_link→laser_frame（同路C），否则 costmap 会丢掉所有 scan。
检测器发 /Current_point(cx) + /image_raw，桥节点消费 /Current_point。

底盘：controller_server 直接发 /cmd_vel（YB_Car_Node 订阅），不经 velocity_smoother/collision_monitor（spike 从简）。

调参（无需重建镜像，launch 参数透传给桥）：
   ros2 launch yahboomcar_astra follow_nav2.launch.py hfov_deg:=62 bearing_sign:=-1 standoff:=1.2
控制器 A/B 对比（换 nav2 参数文件，其余不变）：
   ros2 launch yahboomcar_astra follow_nav2.launch.py controller:=rpp   # 默认 dwb
检测器 A/B 对比（换检测器可执行文件，其余不变；默认仍是旧检测器，不影响现状）：
   ros2 launch yahboomcar_astra follow_nav2.launch.py detector:=objTracker_reid_tpu   # 默认 objTracker_tpu
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

NAV2_NODES = ["controller_server", "planner_server", "behavior_server", "bt_navigator"]


def generate_launch_description():
    pkg = get_package_share_directory("yahboomcar_astra")
    # controller:=rpp → nav2_follow.yaml；controller:=dwb → nav2_follow_dwb.yaml（仅控制器块不同，供 A/B 对比）。
    # 显式传 params_file 则覆盖此默认。
    default_params = PythonExpression(
        ["'", os.path.join(pkg, "config", "nav2_follow"),
         "' + ('_dwb' if '", LaunchConfiguration("controller"), "' == 'dwb' else '') + '.yaml'"])

    params_file = LaunchConfiguration("params_file")
    hfov_deg = LaunchConfiguration("hfov_deg")
    bearing_sign = LaunchConfiguration("bearing_sign")
    standoff = LaunchConfiguration("standoff")
    gimbal_tilt = LaunchConfiguration("gimbal_tilt")
    enable_gimbal_pan = LaunchConfiguration("enable_gimbal_pan")
    servo_sign = LaunchConfiguration("servo_sign")

    args = [
        DeclareLaunchArgument("controller", default_value="dwb",
                              description="rpp|dwb 控制器 A/B 对比：dwb=重/局部避障(默认)，rpp=轻/纯几何跟线"),
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("hfov_deg", default_value="60.0", description="相机水平视场角(实测)"),
        DeclareLaunchArgument("bearing_sign", default_value="1.0", description="转错方向就设 -1.0"),
        DeclareLaunchArgument("standoff", default_value="1.0", description="停在人前多远(m)"),
        DeclareLaunchArgument("gimbal_tilt", default_value="-30",
                              description="云台俯仰(servo_s2)：负=低头看近处/人身，觉得太高就更负"),
        DeclareLaunchArgument("enable_gimbal_pan", default_value="true",
                              description="云台v3：跟随回正(0)由底盘转向对准、丢失才慢扫搜索(不积分cx→不震荡)；false=死锁正前方"),
        DeclareLaunchArgument("servo_sign", default_value="-1.0",
                              description="云台硬件极性(本车实测-1.0)；正负反了会让云台跑到一侧不回来"),
        DeclareLaunchArgument("bringup", default_value="true",
                              description="是否一并起 yahboomcar_bringup(odom+TF)；已单独起就设 false"),
        DeclareLaunchArgument("run_detector", default_value="true",
                              description="是否一并起检测器(发 /Current_point)"),
        DeclareLaunchArgument("detector", default_value="objTracker_tpu",
                              description="objTracker_tpu(默认，不变)|objTracker_reid_tpu(Re-ID外观锁定，拒误检/多人不跳/遮挡重认)"),
        DeclareLaunchArgument("show_image", default_value="true",
                              description="检测器弹 OpenCV 画面窗口(看黄框)；嫌占 CPU 设 false"),
    ]

    # 一条命令起全套：bringup(odom+TF) → nav2+桥 → 检测器。已单独起 bringup 就传 bringup:=false。
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("yahboomcar_bringup"),
            "launch", "yahboomcar_bringup_launch.py")),
        condition=IfCondition(LaunchConfiguration("bringup")),
    )

    detector = Node(
        package="yahboomcar_astra", executable=LaunchConfiguration("detector"),
        name=LaunchConfiguration("detector"),
        output="screen",
        parameters=[{"show_image": ParameterValue(LaunchConfiguration("show_image"), value_type=bool)}],
        condition=IfCondition(LaunchConfiguration("run_detector")),
    )

    # /scan 用 laser_frame，bringup 的 URDF 没有它 → 自补静态 TF（激光在车几何中心）。
    static_tf = Node(
        package="tf2_ros", executable="static_transform_publisher", name="base_to_laser_follow",
        arguments=["0", "0", "0.11", "0", "0", "0", "base_link", "laser_frame"],
    )

    nav2 = [
        Node(package="nav2_controller", executable="controller_server", name="controller_server",
             output="screen", parameters=[params_file]),
        Node(package="nav2_planner", executable="planner_server", name="planner_server",
             output="screen", parameters=[params_file]),
        Node(package="nav2_behaviors", executable="behavior_server", name="behavior_server",
             output="screen", parameters=[params_file]),
        Node(package="nav2_bt_navigator", executable="bt_navigator", name="bt_navigator",
             output="screen", parameters=[params_file]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_follow", output="screen",
             parameters=[{"autostart": True, "node_names": NAV2_NODES}]),
    ]

    bridge = Node(
        package="yahboomcar_astra", executable="person_goal_bridge", name="person_goal_bridge",
        output="screen",
        parameters=[{
            "hfov_deg": ParameterValue(hfov_deg, value_type=float),
            "bearing_sign": ParameterValue(bearing_sign, value_type=float),
            "standoff": ParameterValue(standoff, value_type=float),
            "gimbal_tilt": ParameterValue(gimbal_tilt, value_type=int),
            "enable_gimbal_pan": ParameterValue(enable_gimbal_pan, value_type=bool),
            "servo_sign": ParameterValue(servo_sign, value_type=float),
        }],
    )

    return LaunchDescription(args + [bringup, static_tf] + nav2 + [bridge, detector])
