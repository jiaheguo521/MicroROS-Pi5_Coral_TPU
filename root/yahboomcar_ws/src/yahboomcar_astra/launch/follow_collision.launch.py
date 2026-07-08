"""路 C：nav2_collision_monitor 当 footprint 感知的安全刹车层。
起三件：
  1. static_transform_publisher  base_link -> laser_frame (激光居中，离地 0.11m)
  2. nav2_collision_monitor       /cmd_vel_raw --(停/减速区)--> /cmd_vel
  3. nav2_lifecycle_manager       自动 configure+activate collision_monitor

跟随节点这样接：
  ros2 run yahboomcar_astra objControl --ros-args -r /cmd_vel:=/cmd_vel_raw -p avoid_enable:=false
  ros2 run yahboomcar_astra objTracker_tpu

collision_monitor 只减速/停车(footprint 感知)，不绕行；只需 base_link->laser_frame 静态 TF
(base_shift_correction=False，故不依赖 odom TF)。

注意：本版 Nav2 的 polygon points 是 double 数组，扁平列出 [x1,y1,x2,y2,...]（顺/逆时针闭合）。
"""
from launch import LaunchDescription
from launch_ros.actions import Node

# 车 0.24x0.16m(半 0.12x0.08)。停止区=footprint+~5cm；减速区≈+25cm。base_link 在车几何中心。
STOP_POINTS = [0.17, 0.13, 0.17, -0.13, -0.17, -0.13, -0.17, 0.13]
SLOW_POINTS = [0.37, 0.33, 0.37, -0.33, -0.37, -0.33, -0.37, 0.33]


def generate_launch_description():
    collision_params = {
        "base_frame_id": "base_link",
        "odom_frame_id": "odom",
        "cmd_vel_in_topic": "cmd_vel_raw",
        "cmd_vel_out_topic": "cmd_vel",
        "state_topic": "collision_monitor_state",
        "transform_tolerance": 0.5,
        "source_timeout": 1.0,
        "base_shift_correction": False,   # 不依赖 odom TF
        "stop_pub_timeout": 2.0,
        "polygons": ["PolygonStop", "PolygonSlow"],
        "PolygonStop.type": "polygon",
        "PolygonStop.points": STOP_POINTS,
        "PolygonStop.action_type": "stop",
        "PolygonStop.min_points": 2,      # 薄障碍(椅子腿)也就几个点；如误触发调大
        "PolygonStop.visualize": True,
        "PolygonStop.polygon_pub_topic": "polygon_stop",
        "PolygonSlow.type": "polygon",
        "PolygonSlow.points": SLOW_POINTS,
        "PolygonSlow.action_type": "slowdown",
        "PolygonSlow.slowdown_ratio": 0.3,
        "PolygonSlow.min_points": 2,
        "PolygonSlow.visualize": True,
        "PolygonSlow.polygon_pub_topic": "polygon_slow",
        "observation_sources": ["scan"],
        "scan.type": "scan",
        "scan.topic": "/scan",
    }
    return LaunchDescription([
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="base_to_laser",
            arguments=["0", "0", "0.11", "0", "0", "0", "base_link", "laser_frame"],
        ),
        Node(
            package="nav2_collision_monitor", executable="collision_monitor",
            name="collision_monitor", output="screen",
            parameters=[collision_params],
        ),
        Node(
            package="nav2_lifecycle_manager", executable="lifecycle_manager",
            name="lifecycle_manager_collision", output="screen",
            parameters=[{"autostart": True, "node_names": ["collision_monitor"]}],
        ),
    ])
