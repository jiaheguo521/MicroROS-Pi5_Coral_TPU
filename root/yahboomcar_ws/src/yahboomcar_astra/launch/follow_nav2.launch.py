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
    enable_gimbal_tilt = LaunchConfiguration("enable_gimbal_tilt")
    tilt_sign = LaunchConfiguration("tilt_sign")
    tilt_kp = LaunchConfiguration("tilt_kp")
    tilt_head_frac = LaunchConfiguration("tilt_head_frac")
    tilt_setpoint_px = LaunchConfiguration("tilt_setpoint_px")
    tilt_smooth = LaunchConfiguration("tilt_smooth")
    pan_max_step_deg = LaunchConfiguration("pan_max_step_deg")
    close_stop_frac = LaunchConfiguration("close_stop_frac")
    close_stop_px = LaunchConfiguration("close_stop_px")
    size_smooth = LaunchConfiguration("size_smooth")
    servo_sign = LaunchConfiguration("servo_sign")

    args = [
        DeclareLaunchArgument("controller", default_value="dwb",
                              description="rpp|dwb 控制器 A/B 对比：dwb=重/局部避障(默认)，rpp=轻/纯几何跟线"),
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("hfov_deg", default_value="60.0", description="相机水平视场角(实测)"),
        DeclareLaunchArgument("bearing_sign", default_value="1.0", description="转错方向就设 -1.0"),
        DeclareLaunchArgument("standoff", default_value="1.5", description="停在人前多远(m)"),
        DeclareLaunchArgument("gimbal_tilt", default_value="-30",
                              description="云台俯仰(servo_s2)：负=低头看近处/人身，觉得太高就更负"),
        DeclareLaunchArgument("enable_gimbal_pan", default_value="true",
                              description="云台v3 pan：指向odom估计人方位由底盘对准、只在曾锁定又丢失才慢扫(冷启动不扫)；false=死锁正前方"),
        DeclareLaunchArgument("enable_gimbal_tilt", default_value="true",
                              description="云台tilt主动跟踪：跟cy把人竖直居中(缓解贴地仰视头顶出画)；false=固定gimbal_tilt"),
        DeclareLaunchArgument("tilt_sign", default_value="1.0",
                              description="tilt硬件极性(上车验)；人越点头/画面越偏就取 -1.0"),
        DeclareLaunchArgument("tilt_kp", default_value="0.05",
                              description="tilt竖直像素误差→俯仰步进增益(deg/px)；抖就调小、跟不上就调大"),
        DeclareLaunchArgument("tilt_head_frac", default_value="0.3",
                              description="tilt跟踪点从框中心上移的比例×框尺寸(≈框顶=头)；头老出画调大、总仰头调小、0=跟框中心"),
        DeclareLaunchArgument("tilt_setpoint_px", default_value="240.0",
                              description="把头部估计点竖直放在画面哪个y(px，中心240)；想让人更靠上就调小"),
        DeclareLaunchArgument("tilt_smooth", default_value="0.6",
                              description="tilt目标EMA平滑(0~1)；上下颤就调大(更平滑)、跟得太肉就调小"),
        DeclareLaunchArgument("pan_max_step_deg", default_value="6.0",
                              description="云台pan每拍最多转(deg，10Hz)；横走跟不上→调大、抖→调小(只影响跟踪，不影响搜索)"),
        DeclareLaunchArgument("enable_lost_search", default_value="false",
                              description="丢失后云台是否扫描搜索；默认false=保持最后指向不动(停着比扫更快重认、不甩镜头)"),
        DeclareLaunchArgument("search_step_deg", default_value="2.0",
                              description="搜索扫描每拍转(deg，10Hz→20°/s)；仅 enable_lost_search:=true 时生效"),
        DeclareLaunchArgument("close_stop_frac", default_value="0.9",
                              description="近距硬停系数×standoff；dist<此值锁死不发目标(治近距横移前冲)，越大越早停"),
        DeclareLaunchArgument("close_stop_px", default_value="320.0",
                              description="检测框最长边EMA≥此(px)=人明显很近→硬停(不依赖雷达，治近距侧向追幽灵前冲)；看日志ema标定，0=禁用"),
        DeclareLaunchArgument("size_smooth", default_value="0.6",
                              description="框尺寸EMA平滑(0~1)；SSD尺寸噪声大，硬停判据抖就调大"),
        DeclareLaunchArgument("servo_sign", default_value="-1.0",
                              description="云台pan硬件极性(本车实测-1.0)；正负反了会让云台跑到一侧不回来"),
        DeclareLaunchArgument("bringup", default_value="true",
                              description="是否一并起 yahboomcar_bringup(odom+TF)；已单独起就设 false"),
        DeclareLaunchArgument("run_detector", default_value="true",
                              description="是否一并起检测器(发 /Current_point)"),
        DeclareLaunchArgument("detector", default_value="objTracker_tpu",
                              description="objTracker_tpu(默认，不变)|objTracker_reid_tpu(Re-ID外观锁定，拒误检/多人不跳/遮挡重认)"),
        DeclareLaunchArgument("reid_model", default_value="reid_youtu",
                              description="Re-ID模型目录(仅objTracker_reid_tpu用)：reid_youtu(base,60ms/4.4Hz,最准)|reid_youtu_p70(15.6ms/18Hz,云台更跟手,顶替率略高)|reid_youtu_p50"),
        DeclareLaunchArgument("show_image", default_value="true",
                              description="检测器弹 OpenCV 画面窗口(看黄框)；嫌占 CPU 设 false"),
        # Re-ID 阈值(属于具体嵌入空间，换 reid_model 必须重标；上车用 debug_sim 看实际相似度定)
        DeclareLaunchArgument("sim_floor", default_value="0.6",
                              description="维持已锁目标的低门槛(confirmed态)；太高会老丢自己、太低旁人会被认成你"),
        DeclareLaunchArgument("ema_min_sim", default_value="0.65",
                              description="只有相似度≥此才写回模板(比sim_floor高一截)，防边界误匹配污染模板"),
        DeclareLaunchArgument("relock_sim_floor", default_value="0.75",
                              description="跟丢后**重捕**的高门槛(必须明显>sim_floor)：防目标不在时旁人被误认"),
        DeclareLaunchArgument("relock_min_hits", default_value="5",
                              description="重捕需连续这么多帧都过高门槛才重新锁上"),
        DeclareLaunchArgument("debug_sim", default_value="false",
                              description="打印每个候选的相似度(调阈值必备)"),
    ]

    # reid_model 目录 → det/emb 两个 co-compile 产物的绝对路径(镜像内 src 路径，节点默认也在此)
    _reid_base = "/root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/models/"
    det_model_path = PythonExpression(
        ["'", _reid_base, "' + '", LaunchConfiguration("reid_model"), "' + '/det_reid_edgetpu.tflite'"])
    emb_model_path = PythonExpression(
        ["'", _reid_base, "' + '", LaunchConfiguration("reid_model"), "' + '/emb_reid_edgetpu.tflite'"])

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
        parameters=[{
            "show_image": ParameterValue(LaunchConfiguration("show_image"), value_type=bool),
            "det_model_path": ParameterValue(det_model_path, value_type=str),
            "emb_model_path": ParameterValue(emb_model_path, value_type=str),
            "sim_floor": ParameterValue(LaunchConfiguration("sim_floor"), value_type=float),
            "ema_min_sim": ParameterValue(LaunchConfiguration("ema_min_sim"), value_type=float),
            "relock_sim_floor": ParameterValue(LaunchConfiguration("relock_sim_floor"), value_type=float),
            "relock_min_hits": ParameterValue(LaunchConfiguration("relock_min_hits"), value_type=int),
            "debug_sim": ParameterValue(LaunchConfiguration("debug_sim"), value_type=bool),
        }],
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
            "enable_gimbal_tilt": ParameterValue(enable_gimbal_tilt, value_type=bool),
            "tilt_sign": ParameterValue(tilt_sign, value_type=float),
            "tilt_kp": ParameterValue(tilt_kp, value_type=float),
            "tilt_head_frac": ParameterValue(tilt_head_frac, value_type=float),
            "tilt_setpoint_px": ParameterValue(tilt_setpoint_px, value_type=float),
            "tilt_smooth": ParameterValue(tilt_smooth, value_type=float),
            "pan_max_step_deg": ParameterValue(pan_max_step_deg, value_type=float),
            "enable_lost_search": ParameterValue(
                LaunchConfiguration("enable_lost_search"), value_type=bool),
            "search_step_deg": ParameterValue(
                LaunchConfiguration("search_step_deg"), value_type=float),
            "close_stop_frac": ParameterValue(close_stop_frac, value_type=float),
            "close_stop_px": ParameterValue(close_stop_px, value_type=float),
            "size_smooth": ParameterValue(size_smooth, value_type=float),
            "servo_sign": ParameterValue(servo_sign, value_type=float),
        }],
    )

    return LaunchDescription(args + [bringup, static_tf] + nav2 + [bridge, detector])
