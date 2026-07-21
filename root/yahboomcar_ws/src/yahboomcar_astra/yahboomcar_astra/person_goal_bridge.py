"""路 B「跟人目标桥」v3：维护"人在 odom 系的位置估计"作为唯一真相源，喂 Nav2。

与 v2 的区别（v2 每帧现算、无记忆，遇遮挡/漂移/误检就抽）：
  * **有记忆**：把每帧测得的人点 EMA 进一个 odom 系估计 `est_odom`；
    测量在 odom 空间做离群门控——椅子挡在中间→该方向雷达返回骤近→测点跳→判遮挡→
    **保持估计不动**，Nav2 继续朝估计走并绕开椅子，而不是把椅子当成人而停车。
  * **抗漂移**：估计做平滑，且相机持续重测→里程计漂移不累积进目标。
  * **解耦云台**：pan 不积分 cx（那会与底盘转向环耦合震荡），改指向 odom 估计的人方位、
    由底盘转向对准；tilt 跟 cy 竖直居中（纯像素伺服、与底盘无耦合）。**只在"曾锁定过又丢失"
    才慢扫搜索，冷启动/未锁不扫**（避免扫描取景与 Re-ID 站稳锁定打架，见 devlog §8.5）。

链路：/Current_point(cx) →(云台角+残余cx)方位 → /scan 扇区最近距离 → 人在激光帧
      → TF 转 odom → EMA+门控进 est_odom → 沿 robot→est 方向后退 standoff 得 odom 目标
      → NavigateToPose（移动超阈值重发抢占）→ Nav2 planner 真绕行。

前提（阶段0探针已核实，见 memory pathb-nav2-probe）：先起 yahboomcar_bringup（提供
/odom + odom→base_footprint→base_link 的 TF）。位移全交给 Nav2；本节点只在**近距区**
（人已在 standoff 内、Nav2 目标已取消）直接发 /cmd_vel 做**原地转向**把人转回正前方，
此时 controller_server 不发速度，不存在两个发布者打架。
Re-ID 后续作为上游节点接入（只改喂给 /Current_point 的 cx，本桥不动）——那是干净插槽。
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from nav2_msgs.action import NavigateToPose
from yahboomcar_msgs.msg import Position

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  注册 PoseStamped 的 do_transform


# ---- 纯几何（与 benchmarks/smoke_person_bridge.py 手工保持一致）----
def bearing_from_cx(cx, width, hfov_deg, sign):
    cx0 = width / 2.0
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return sign * math.atan2(cx0 - cx, fx)


def sector_min_range(scan, center_rad, half_rad):
    """scan 内以 center_rad 为中心 ±half_rad 扇区的有效最小距离；剔 0.0/inf/超量程；绕±π正确。"""
    amin = scan.angle_min
    inc = scan.angle_increment
    rmin = scan.range_min
    rmax = scan.range_max
    best = None
    for i, r in enumerate(scan.ranges):
        ang = amin + inc * i
        delta = math.atan2(math.sin(ang - center_rad), math.cos(ang - center_rad))
        if abs(delta) <= half_rad and math.isfinite(r) and rmin < r < rmax:
            if best is None or r < best:
                best = r
    return best
# ---- end ----


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class PersonGoalBridge(Node):
    def __init__(self):
        super().__init__("person_goal_bridge")
        # --- 参数 ---
        self.declare_parameter("image_width", 640)
        self.declare_parameter("hfov_deg", 60.0)         # ⏳ 车上实测
        self.declare_parameter("bearing_sign", 1.0)       # ⏳ 转错取反
        self.declare_parameter("sector_half_deg", 8.0)    # 雷达扇区半宽
        self.declare_parameter("standoff", 1.5)           # 停在人前多远 (m)
        self.declare_parameter("goal_frame", "odom")      # Nav2 全局帧
        self.declare_parameter("base_frame", "base_link") # 机器人本体帧(取 odom 位姿)
        self.declare_parameter("resend_dist", 0.25)       # goal 移动超此距离才重发 (m)
        self.declare_parameter("resend_period", 1.0)      # 或至少每隔这么久重发一次 (s)
        self.declare_parameter("lost_timeout", 0.8)       # 相机多久没看到人算丢 (s)
        self.declare_parameter("max_goal_dist", 2.5)      # goal 离车最远(m)：绝不发超 costmap 的远目标
        # odom 估计滤波 + 遮挡门控（治 v2 的抖/椅子挡就停/漂移）
        self.declare_parameter("pos_smooth", 0.6)         # est_odom EMA(0~1，越大越平滑)
        self.declare_parameter("max_pos_jump", 0.8)       # 测点在 odom 里单帧跳超此(m)→判遮挡/离群→保持估计
        self.declare_parameter("max_rejects", 5)          # 连续拒这么多次就认账(人真的移位了)
        self.declare_parameter("stop_deadband", 0.15)     # 到 standoff+此(m)内→取消目标停车(滞回)
        # 近距滞回硬停：dist<standoff×close_stop_frac 进 hold_stop(最高优先级不发目标)，
        # 连续 resume_frames 帧确认退到 standoff+deadband 外才解除(防近距误采偏远→前冲)
        self.declare_parameter("close_stop_frac", 0.9)    # 进硬停的近距系数(×standoff)
        self.declare_parameter("resume_frames", 3)        # 解除硬停需连续这么多帧确认人退远(tick 5Hz)
        # 不依赖雷达的近距判据：人越近检测框越大。近距+侧向时云台指向侧方、雷达扇区易采到背景→
        # 距离读偏远→追幽灵前冲；此时框尺寸(px)才是可靠的"近"信号。框≥此值直接硬停，无视雷达距离。
        self.declare_parameter("close_stop_px", 320.0)    # 检测框最长边(EMA,px)≥此=人明显很近→硬停；0=禁用只靠雷达距离
        self.declare_parameter("size_smooth", 0.6)        # 框尺寸 EMA(0~1,越大越平滑)：SSD 尺寸噪声大，先平滑再判硬停
        # 近距原地转向：进硬停区后**不位移**，但底盘原地转把人转回正前方。
        # 否则人绕到侧后方时只有云台在追，很快撞 ±pan_limit 限位、然后彻底看不到人。
        # 与云台读同一个 odom 估计、互不反馈（底盘转过去→人相对车方位→0→云台自然回中），
        # 不会重演 §4.1 那种"两个环通过 cx 互相打架"的耦合振荡。
        self.declare_parameter("enable_close_rotate", True)
        self.declare_parameter("rotate_deadband_deg", 8.0)  # 人在正前方±此角内就不转(防原地抽搐)
        self.declare_parameter("rotate_kp", 1.2)            # 方位误差(rad)→角速度(rad/s)
        self.declare_parameter("rotate_max", 0.6)           # 角速度上限 (rad/s)
        self.declare_parameter("rotate_min", 0.18)          # 角速度下限：太小电机转不动，原地干磨
        # 丢目标后朝"最后看到的位置"继续转多久(s)。人绕到侧后方走出视野时，底盘再转过去常能重新看到。
        # 转到正对该位置仍没看到就提前停（盲转无意义）。0=关闭，丢了就立刻停。
        self.declare_parameter("lost_rotate_time", 2.5)
        self.declare_parameter("gimbal_tilt", -30)        # servo_s2 俯仰(负=低头)
        # 云台 v3：跟随时回正(0)由底盘转向对准；丢失才慢扫搜索。不积分 cx→不与底盘耦合。
        self.declare_parameter("enable_gimbal_pan", True)
        self.declare_parameter("pan_limit_deg", 70.0)     # 云台 pan 机械上限 (±deg)
        self.declare_parameter("pan_max_step_deg", 6.0)   # 每个云台拍最多转多少(跟踪/搜索速度)；横走跟不上→调大
        self.declare_parameter("pan_deadband_deg", 3.0)   # 人在云台视线±此角内就不动(非必要不动)
        # 丢失后是否扫描搜索。默认 False=**保持最后指向不动**(人多半原地附近重现，停着更快重认、不甩镜头)。
        # 扫描速度独立于跟踪速度：pan_max_step 为跟手调大后若复用会让搜索甩飞(实测 60°/s 太快)。
        self.declare_parameter("enable_lost_search", False)
        self.declare_parameter("search_step_deg", 2.0)    # 搜索扫描每拍转多少(deg，10Hz→20°/s)
        # 主动 tilt(俯仰)跟踪：把人竖直居中(缓解贴地仰视时人走远/走近头顶出画)。tilt 与底盘无耦合
        # (底盘无 pitch)，是纯像素伺服的干净单环，不会重演 pan 的震荡。极性 tilt_sign 上车验，同 servo_sign。
        self.declare_parameter("enable_gimbal_tilt", True)
        self.declare_parameter("tilt_setpoint_px", 240.0)  # 想把人的头部估计点竖直放在画面哪个 y(px；640x480 中心=240)
        # 跟"头部估计点"而非框中心：贴地近距人框占满/顶部被切→框中心恒在画面中部、头出画也不动(饱和)。
        # 用检测器已发的框尺寸把跟踪点上移 head_y=cy−frac×size(≈框顶)，框越大上移越多→越使劲上抬。0=退回框中心。
        self.declare_parameter("tilt_head_frac", 0.3)
        self.declare_parameter("tilt_smooth", 0.6)         # head_y 目标 EMA(0~1，越大越平滑)：滤检测框逐帧抖动、防云台上下颤
        self.declare_parameter("tilt_deadband_px", 25.0)   # 头部点在此竖直像素带内就不动(抑颤)
        self.declare_parameter("tilt_kp", 0.05)            # 竖直像素误差→俯仰角步进增益 (deg/px)
        self.declare_parameter("tilt_max_step_deg", 3.0)   # 每个云台拍俯仰最多转多少(限速抑抖)
        self.declare_parameter("tilt_sign", 1.0)           # 硬件极性(上车验，转反取 -1.0)
        self.declare_parameter("gimbal_period", 0.1)      # 云台控制周期 (s)
        # 硬件极性：本车实测需 -1.0（正号会让"指向估计"环变正反馈→云台跑到一侧不回来）。
        self.declare_parameter("servo_sign", -1.0)
        g = self.get_parameter
        self.image_width = g("image_width").value
        self.hfov_deg = g("hfov_deg").value
        self.bearing_sign = g("bearing_sign").value
        self.sector_half = math.radians(g("sector_half_deg").value)
        self.standoff = g("standoff").value
        self.goal_frame = g("goal_frame").value
        self.base_frame = g("base_frame").value
        self.resend_dist = g("resend_dist").value
        self.resend_period = g("resend_period").value
        self.lost_timeout = g("lost_timeout").value
        self.max_goal_dist = g("max_goal_dist").value
        self.pos_smooth = g("pos_smooth").value
        self.max_pos_jump = g("max_pos_jump").value
        self.max_rejects = g("max_rejects").value
        self.stop_deadband = g("stop_deadband").value
        self.close_stop_frac = g("close_stop_frac").value
        self.resume_frames = g("resume_frames").value
        self.close_stop_px = g("close_stop_px").value
        self.size_smooth = g("size_smooth").value
        self.enable_close_rotate = g("enable_close_rotate").value
        self.rotate_deadband = math.radians(g("rotate_deadband_deg").value)
        self.rotate_kp = g("rotate_kp").value
        self.rotate_max = g("rotate_max").value
        self.rotate_min = g("rotate_min").value
        self.lost_rotate_time = g("lost_rotate_time").value
        self.gimbal_tilt = int(g("gimbal_tilt").value)
        self.enable_gimbal_pan = g("enable_gimbal_pan").value
        self.pan_limit = math.radians(g("pan_limit_deg").value)
        self.pan_max_step = math.radians(g("pan_max_step_deg").value)
        self.pan_deadband = math.radians(g("pan_deadband_deg").value)
        self.enable_lost_search = g("enable_lost_search").value
        self.search_step = math.radians(g("search_step_deg").value)
        self.enable_gimbal_tilt = g("enable_gimbal_tilt").value
        self.tilt_setpoint_px = g("tilt_setpoint_px").value
        self.tilt_head_frac = g("tilt_head_frac").value
        self.tilt_smooth = g("tilt_smooth").value
        self.tilt_deadband_px = g("tilt_deadband_px").value
        self.tilt_kp = g("tilt_kp").value
        self.tilt_max_step = g("tilt_max_step_deg").value  # deg（servo_s2 本身就是度，不转 rad）
        self.tilt_sign = g("tilt_sign").value
        self.gimbal_period = g("gimbal_period").value
        self.servo_sign = g("servo_sign").value

        # --- TF ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- IO ---
        self.sub_pos = self.create_subscription(Position, "/Current_point", self.on_position, 1)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.on_scan, 1)
        self.pub_goal_dbg = self.create_publisher(PoseStamped, "/person_goal", 1)  # RViz/echo 调试
        self.pub_s1 = self.create_publisher(Int32, "servo_s1", 1)
        self.pub_s2 = self.create_publisher(Int32, "servo_s2", 1)
        # 仅用于"近距原地转向"：此时 Nav2 目标已取消、controller_server 不再发速度，
        # 由本节点独占 /cmd_vel；回到正常跟随前会清零交还给 Nav2。
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 1)
        self.ac = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # --- 状态 ---
        self.cx = None
        self.cy = None
        self.person_size = None     # 检测框最长边(px)，供 tilt 估计头部点：head_y=cy−frac×size
        self.size_ema = None        # 框尺寸 EMA(平滑噪声)，供近距硬停判据 close_by_box
        self.new_pos = False        # 有新检测帧未消费：tilt 只在此为真时步进(防低检测率下过度积分→摆动)
        self.head_y_ema = None      # tilt 目标 head_y 的 EMA(滤检测框逐帧抖动)
        self.ever_locked = False    # Re-ID 是否曾经锁定过(收到过 /Current_point)：区分冷启动 vs 丢失
        self.last_pos_time = 0.0
        self.scan = None
        self.last_scan_time = 0.0
        self.est_odom = None        # 人在 odom 的位置估计 (px,py)——唯一真相源
        self.reject_count = 0
        self.last_goal_xy = None
        self.last_sent_time = 0.0
        self.cam_yaw = 0.0          # 云台 pan 角(rad, base 系 +左)：跟随时→0，丢失时扫
        self.cam_tilt = float(self.gimbal_tilt)  # 云台 tilt 角(deg, servo_s2)：跟 cy 竖直居中
        self.search_dir = 1.0
        self.active_goal_handle = None
        self.stopped = False        # 是否已取消目标停车（避免重复取消）
        self.rotating = False       # 是否正在发原地转向速度（底盘锁存最后速度，必须显式清零）
        self.last_seen_odom = None  # 最后看到人的 odom 位置，供丢失后继续转向找人
        self.last_seen_time = 0.0
        self.hold_stop = False      # 近距滞回硬停：为真时最高优先级不发目标
        self.far_count = 0          # 连续确认"人已退远"的帧数，达 resume_frames 才解除 hold_stop

        # 云台俯仰置初始位(gimbal_tilt)、pan 初始回中；之后 _gimbal_tick 主动跟踪
        s2 = Int32(); s2.data = self.gimbal_tilt; self.pub_s2.publish(s2)
        s1 = Int32(); s1.data = 0; self.pub_s1.publish(s1)

        self.timer = self.create_timer(0.2, self.tick)                     # 5Hz 估计+决策
        self.gtimer = self.create_timer(self.gimbal_period, self._gimbal_tick)  # 10Hz 云台
        self.get_logger().info(
            "person_goal_bridge v3 up: hfov=%.1f sign=%.0f standoff=%.2f pan=%s tilt_track=%s(rest=%d)"
            % (self.hfov_deg, self.bearing_sign, self.standoff,
               self.enable_gimbal_pan, self.enable_gimbal_tilt, self.gimbal_tilt))

    def on_position(self, msg):
        self.cx = msg.anglex
        self.cy = msg.angley
        self.person_size = msg.distance   # 框最长边(px)，tilt 用它把跟踪点从框中心上移到头部
        # SSD 框尺寸逐帧噪声极大(同位置 ±120px)，做近距硬停判据前先 EMA 平滑，否则 [box-close]
        # 会一帧触发一帧不触发→hold 在边界反复→趁掉值那几帧被 resume 解锁→前冲
        self.size_ema = msg.distance if self.size_ema is None else \
            self.size_smooth * self.size_ema + (1 - self.size_smooth) * msg.distance
        self.new_pos = True       # 标记新检测帧，供 tilt 门控步进（防低检测率下过度积分→摆动）
        self.ever_locked = True   # 收到过检测点=Re-ID 已确认锁定过；之后没点才算"丢失"(而非冷启动)
        self.last_pos_time = time.time()

    def on_scan(self, msg):
        self.scan = msg
        self.last_scan_time = time.time()

    # ---- 云台 v3：pan 指向 odom 估计(与底盘解耦、不震荡)、tilt 跟 cy 竖直居中；
    #      丢失才慢扫，冷启动/未锁不扫(避 §8.5 扫描 vs 站稳锁定冲突) ----
    def _gimbal_tick(self):
        now = time.time()
        detected = self.cx is not None and (now - self.last_pos_time) <= self.lost_timeout
        # --- pan (servo_s1) ---
        if self.enable_gimbal_pan:
            if not detected:
                # 冷启动/从未锁定：锁正前方不扫（扫描会改取景，与 Re-ID"站稳3秒"锁定打架，见 §8.5）。
                # 曾锁定又丢失：默认 **enable_lost_search=False → 保持最后指向不动**——人多半在原地
                # 附近重现，停着比扫更快重认、也不甩镜头。要扫则用**独立的 search_step**，
                # 绝不复用 pan_max_step（那个为跟手调大后会让搜索甩到飞快）。
                if self.ever_locked and self.enable_lost_search:
                    self.cam_yaw += self.search_dir * self.search_step
                    if self.cam_yaw >= self.pan_limit:
                        self.cam_yaw = self.pan_limit; self.search_dir = -1.0
                    elif self.cam_yaw <= -self.pan_limit:
                        self.cam_yaw = -self.pan_limit; self.search_dir = 1.0
            else:
                # 有人：把云台指向"人相对车的方位"，把人锁在视野中央。
                # 底盘转向对准人时该方位自然→0，云台随之回正——不积分 cx，不与底盘耦合。
                desired = self._person_bearing_base()
                if desired is not None:
                    err = math.atan2(math.sin(desired - self.cam_yaw), math.cos(desired - self.cam_yaw))
                    if abs(err) > self.pan_deadband:   # 人已在视线中央就不动（非必要不动）
                        self.cam_yaw += max(-self.pan_max_step, min(self.pan_max_step, err))
                        self.cam_yaw = max(-self.pan_limit, min(self.pan_limit, self.cam_yaw))
            s1 = Int32(); s1.data = int(self.servo_sign * math.degrees(self.cam_yaw))
            self.pub_s1.publish(s1)
        # --- tilt (servo_s2) ---
        if self.enable_gimbal_tilt:
            if detected and self.cy is not None:
                # 只在有"新检测帧"时才走一步：检测器~4.4Hz、云台 10Hz，若每拍都用同一(过期)cy 误差
                # 步进，一次检测间隔会累加 2~3 步→冲过头→上下摆动(objControl 同注释)。pan 用 odom+TF
                # 连续反馈不受此影响，只 tilt 需要门控。像素伺服：把"头部估计点"移到 tilt_setpoint_px
                # (不是框中心，否则近距框占满/切顶时框中心恒在画面中部→头出画也不上抬，饱和)。
                if self.new_pos:
                    raw = self.cy - self.tilt_head_frac * self.person_size if self.person_size else self.cy
                    # EMA 平滑目标：检测框(尤其尺寸/宽高比)逐帧抖→head_y 直接跟会上下颤，先滤噪再伺服
                    self.head_y_ema = raw if self.head_y_ema is None else \
                        self.tilt_smooth * self.head_y_ema + (1 - self.tilt_smooth) * raw
                    err_px = self.tilt_setpoint_px - self.head_y_ema
                    if abs(err_px) > self.tilt_deadband_px:
                        step = self.tilt_sign * self.tilt_kp * err_px
                        step = max(-self.tilt_max_step, min(self.tilt_max_step, step))
                        self.cam_tilt = max(-90.0, min(20.0, self.cam_tilt + step))
                    self.new_pos = False
            else:
                # 开了搜索才回默认俯仰位；否则保持最后俯仰不动（与 pan 一致：丢失后镜头不乱动）
                if self.ever_locked and self.enable_lost_search:
                    self.cam_tilt = float(self.gimbal_tilt)
                self.head_y_ema = None                    # 清 EMA，重认时从头平滑
            s2 = Int32(); s2.data = int(round(self.cam_tilt))
            self.pub_s2.publish(s2)

    def _person_bearing_base(self):
        """人相对车(base 系)的方位角。优先用 odom 估计换算(解耦)；估计未建立时退回用 cx 直读
        (此刻底盘不动，无耦合)——保证刚重认、雷达还没给距离时也把人锁在画面里。"""
        if self.est_odom is not None:
            pose = self._robot_pose()
            if pose is not None:
                a = math.atan2(self.est_odom[1] - pose[1], self.est_odom[0] - pose[0])
                return math.atan2(math.sin(a - pose[2]), math.cos(a - pose[2]))
        if self.cx is not None:
            return self.cam_yaw + bearing_from_cx(self.cx, self.image_width, self.hfov_deg, self.bearing_sign)
        return None

    # ---- odom 估计 ----
    def _measure_person_odom(self):
        """测一次人在 odom 的位置：cx→(云台角+残余)方位→扇区距离→激光帧→TF odom。无有效距离返回 None。"""
        residual = bearing_from_cx(self.cx, self.image_width, self.hfov_deg, self.bearing_sign)
        bearing = self.cam_yaw + residual
        r = sector_min_range(self.scan, bearing, self.sector_half)
        if r is None:
            return None
        ps = PoseStamped()
        ps.header.frame_id = self.scan.header.frame_id   # /scan 自带 frame（laser_frame）
        ps.header.stamp = rclpy.time.Time().to_msg()      # 用最新可用 TF
        ps.pose.position.x = r * math.cos(bearing)
        ps.pose.position.y = r * math.sin(bearing)
        ps.pose.orientation.w = 1.0
        try:
            po = self.tf_buffer.transform(
                ps, self.goal_frame, timeout=rclpy.duration.Duration(seconds=0.2))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException, tf2_ros.TransformException) as e:
            self.get_logger().warn("TF %s->%s 失败: %s" % (ps.header.frame_id, self.goal_frame, e),
                                   throttle_duration_sec=2.0)
            return None
        return (po.pose.position.x, po.pose.position.y)

    def _update_estimate(self, z):
        """EMA 更新 est_odom；odom 空间离群门控：突跳(遮挡/误检)→保持估计，连续多次才认账。"""
        if self.est_odom is None:
            self.est_odom = z
            self.reject_count = 0
            return
        d = math.hypot(z[0] - self.est_odom[0], z[1] - self.est_odom[1])
        if d > self.max_pos_jump and self.reject_count < self.max_rejects:
            self.reject_count += 1
            return   # 遮挡/离群：保持估计不动（Nav2 继续朝估计走，绕开遮挡物）
        self.reject_count = 0
        a = self.pos_smooth
        self.est_odom = (a * self.est_odom[0] + (1 - a) * z[0],
                         a * self.est_odom[1] + (1 - a) * z[1])

    def _robot_pose(self):
        """机器人在 odom 的位姿 (x, y, yaw)。"""
        try:
            t = self.tf_buffer.lookup_transform(
                self.goal_frame, self.base_frame, rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException, tf2_ros.TransformException) as e:
            self.get_logger().warn("TF %s->%s 失败: %s" % (self.goal_frame, self.base_frame, e),
                                   throttle_duration_sec=2.0)
            return None
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.transform.translation.x, t.transform.translation.y, yaw)

    def _store_handle(self, future):
        try:
            self.active_goal_handle = future.result()
        except Exception:
            self.active_goal_handle = None

    def _stop(self):
        """取消当前 Nav2 目标让车停（到 standoff / 丢目标时用）。只取消一次。"""
        if self.stopped:
            return
        if self.active_goal_handle is not None:
            self.active_goal_handle.cancel_goal_async()
            self.active_goal_handle = None
        self.last_goal_xy = None
        self.stopped = True

    def _halt_rotation(self):
        """清零原地转向速度。**底盘会锁存最后一条速度**，不显式发 0 车会一直转下去
        （objControl.py 退出兜底同理）。交还 /cmd_vel 给 Nav2 前、丢目标时都必须调。"""
        if self.rotating:
            self.pub_cmd.publish(Twist())
            self.rotating = False

    def _rotate_to_bearing(self, b):
        """把 base 系方位误差 b(rad, +左) 转成原地转向速度发出；已在死区内则明确停住。"""
        if abs(b) <= self.rotate_deadband:
            self._halt_rotation()
            return
        w = max(-self.rotate_max, min(self.rotate_max, self.rotate_kp * b))
        if abs(w) < self.rotate_min:  # 太小电机转不动，只会原地干磨
            w = math.copysign(self.rotate_min, w)
        tw = Twist()
        tw.angular.z = w              # +z=左转；人在左(bearing>0)就左转。转反了把 rotate_kp 取负
        self.pub_cmd.publish(tw)      # linear.x 恒为 0——只转不走
        self.rotating = True

    def _rotate_toward_person(self):
        """近距：不位移，只原地转底盘把人转回正前方。
        人绕到侧后方时若只有云台追，很快撞 ±pan_limit 限位、之后彻底看不到人；底盘转过去后
        人相对车的方位自然→0，云台随之回中，限位问题消失。方位取自与云台同一个 odom 估计，
        两环互不反馈（不是拿 cx 互相喂），所以不会重演 §4.1 的耦合振荡。"""
        if not self.enable_close_rotate:
            self._halt_rotation()
            return
        b = self._person_bearing_base()
        if b is None:                 # 估计还没建立：宁可不动
            self._halt_rotation()
            return
        self._rotate_to_bearing(b)

    def _rotate_toward_last_seen(self, now):
        """丢目标后朝"最后看到的位置"再转一段：人绕到侧后方走出视野时，底盘继续转过去往往
        就重新看到了（否则车停在原地、人再也进不了画面）。

        用**最后看到的 odom 位置**而不是"当时的 base 方位角"——车一转，base 系的角度立刻过期；
        odom 是世界系固定的，能随车转动持续换算出正确方位，转到正对它就自然收敛。
        超过 lost_rotate_time 或已经转到正对它仍没看到，就停（再转下去只是盲转）。"""
        if (not self.enable_close_rotate or self.lost_rotate_time <= 0
                or self.last_seen_odom is None
                or now - self.last_seen_time > self.lost_rotate_time):
            self._halt_rotation()
            return
        pose = self._robot_pose()
        if pose is None:
            self._halt_rotation()
            return
        a = math.atan2(self.last_seen_odom[1] - pose[1], self.last_seen_odom[0] - pose[0])
        self._rotate_to_bearing(math.atan2(math.sin(a - pose[2]), math.cos(a - pose[2])))

    def tick(self):
        now = time.time()
        if self.scan is None:
            return
        # 相机丢人/雷达失效：停车、清估计（让云台进搜索、下次从头重建估计）
        if (self.cx is None or now - self.last_pos_time > self.lost_timeout
                or now - self.last_scan_time > self.lost_timeout):
            self._stop()
            # 丢目标：朝最后看到的位置再转一段(常能把人重新转进视野)；超时/转到位就自动停。
            # 注意底盘锁存最后速度，这个函数在任何不该转的情况下都会显式发 0。
            self._rotate_toward_last_seen(now)
            self.est_odom = None
            self.size_ema = None    # 丢失清 EMA，重认时从头平滑
            return

        z = self._measure_person_odom()
        if z is not None:
            self._update_estimate(z)
        if self.est_odom is None:
            self._halt_rotation()
            return  # 尚未建立估计

        rob = self._robot_pose()
        if rob is None:
            self._halt_rotation()
            return
        dx = self.est_odom[0] - rob[0]
        dy = self.est_odom[1] - rob[1]
        dist = math.hypot(dx, dy)
        # 记住"最后看到人的 odom 位置"，供丢失后继续转向用（此刻 cx 是新鲜的、估计也有效）
        self.last_seen_odom = self.est_odom
        self.last_seen_time = now

        # 近距滞回硬停（最高优先级）。两个进停触发，任一满足就 hold_stop：
        #  ① 雷达距离近：dist < standoff×close_stop_frac；
        #  ② 检测框够大：person_size ≥ close_stop_px——**不依赖雷达**。近距+侧向时云台指向侧方、
        #     雷达扇区易采到人身后背景→距离读偏远(dist 会假装很远)→追幽灵前冲，此时只有框尺寸可靠。
        # 只有"雷达退远 且 框也变小"连续 resume_frames 帧才解除——单帧误采的"偏远"或抖动不足以解锁。
        close_by_box = self.size_ema is not None and self.close_stop_px > 0 \
            and self.size_ema >= self.close_stop_px
        close_by_dist = dist < self.standoff * self.close_stop_frac
        self.get_logger().info(
            "follow: dist=%.2f size=%.0f ema=%.0f hold=%s%s" % (
                dist, self.person_size if self.person_size else -1,
                self.size_ema if self.size_ema else -1, self.hold_stop,
                " [box-close]" if close_by_box else ""),
            throttle_duration_sec=1.0)
        if close_by_dist or close_by_box:
            self.hold_stop = True
            self.far_count = 0
        elif dist > self.standoff + self.stop_deadband:   # 雷达退远且未 close_by_box 才累计
            self.far_count += 1
            if self.far_count >= self.resume_frames:
                self.hold_stop = False
        else:
            self.far_count = 0
        # hold_stop 或到 standoff 死区内：取消 Nav2 目标（不位移），改为原地转向对准人
        if self.hold_stop or dist <= self.standoff + self.stop_deadband:
            self._stop()
            self._rotate_toward_person()
            return
        # 出了近距区，交还 /cmd_vel 给 Nav2 前必须先清零，否则残留转速会和 Nav2 打架
        self._halt_rotation()

        # 目标 = 沿 robot→估计 方向后退 standoff，并对离车距离封顶（防超 costmap）
        ux, uy = dx / dist, dy / dist
        gd = min(dist - self.standoff, self.max_goal_dist)
        gx = rob[0] + ux * gd
        gy = rob[1] + uy * gd
        gyaw = math.atan2(dy, dx)

        ps = PoseStamped()
        ps.header.frame_id = self.goal_frame
        ps.header.stamp = rclpy.time.Time().to_msg()
        ps.pose.position.x = gx
        ps.pose.position.y = gy
        qx, qy, qz, qw = yaw_to_quat(gyaw)
        ps.pose.orientation.x = qx; ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz; ps.pose.orientation.w = qw
        self.pub_goal_dbg.publish(ps)

        # 重发节流：goal 移动超阈值 或 距上次超周期 才发
        moved = (self.last_goal_xy is None or
                 math.hypot(gx - self.last_goal_xy[0], gy - self.last_goal_xy[1]) > self.resend_dist)
        stale = (now - self.last_sent_time) > self.resend_period
        if not (moved or stale):
            return
        if not self.ac.server_is_ready():
            self.get_logger().warn("navigate_to_pose 未就绪（Nav2 起了吗）", throttle_duration_sec=3.0)
            return
        goal = NavigateToPose.Goal()
        goal.pose = ps
        fut = self.ac.send_goal_async(goal)   # 发即抢占上一个（单目标 server）
        fut.add_done_callback(self._store_handle)  # 存 handle 以便 _stop() 取消
        self.stopped = False
        self.last_goal_xy = (gx, gy)
        self.last_sent_time = now


def main():
    rclpy.init()
    node = PersonGoalBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 退出兜底：底盘锁存最后速度，Ctrl-C 时若正在原地转向必须发 0，否则车会一直转
        try:
            node.pub_cmd.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():          # 信号处理器可能已 shutdown 过，避免二次 shutdown 抛 RCLError
            rclpy.shutdown()


if __name__ == "__main__":
    main()
