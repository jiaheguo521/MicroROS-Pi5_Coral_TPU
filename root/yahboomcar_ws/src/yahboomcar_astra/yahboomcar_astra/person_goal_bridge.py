"""路 B「跟人目标桥」v3：维护"人在 odom 系的位置估计"作为唯一真相源，喂 Nav2。

与 v2 的区别（v2 每帧现算、无记忆，遇遮挡/漂移/误检就抽）：
  * **有记忆**：把每帧测得的人点 EMA 进一个 odom 系估计 `est_odom`；
    测量在 odom 空间做离群门控——椅子挡在中间→该方向雷达返回骤近→测点跳→判遮挡→
    **保持估计不动**，Nav2 继续朝估计走并绕开椅子，而不是把椅子当成人而停车。
  * **抗漂移**：估计做平滑，且相机持续重测→里程计漂移不累积进目标。
  * **解耦云台**：云台不再积分 cx（那会与底盘转向环耦合震荡）。跟随时云台回正(0)、
    由底盘转向对准人；**只有丢失目标才慢扫搜索**。两环都读同一 odom 估计、互不反馈。

链路：/Current_point(cx) →(云台角+残余cx)方位 → /scan 扇区最近距离 → 人在激光帧
      → TF 转 odom → EMA+门控进 est_odom → 沿 robot→est 方向后退 standoff 得 odom 目标
      → NavigateToPose（移动超阈值重发抢占）→ Nav2 planner 真绕行。

前提（阶段0探针已核实，见 memory pathb-nav2-probe）：先起 yahboomcar_bringup（提供
/odom + odom→base_footprint→base_link 的 TF）。本节点不发 /cmd_vel——底盘全交给 Nav2。
Re-ID 后续作为上游节点接入（只改喂给 /Current_point 的 cx，本桥不动）——那是干净插槽。
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
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
        self.declare_parameter("standoff", 1.0)           # 停在人前多远 (m)
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
        self.declare_parameter("gimbal_tilt", -30)        # servo_s2 俯仰(负=低头)
        # 云台 v3：跟随时回正(0)由底盘转向对准；丢失才慢扫搜索。不积分 cx→不与底盘耦合。
        self.declare_parameter("enable_gimbal_pan", True)
        self.declare_parameter("pan_limit_deg", 70.0)     # 云台 pan 机械上限 (±deg)
        self.declare_parameter("pan_max_step_deg", 4.0)   # 每个云台拍最多转多少(跟踪/搜索速度，限速抑抖)
        self.declare_parameter("pan_deadband_deg", 3.0)   # 人在云台视线±此角内就不动(非必要不动)
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
        self.gimbal_tilt = int(g("gimbal_tilt").value)
        self.enable_gimbal_pan = g("enable_gimbal_pan").value
        self.pan_limit = math.radians(g("pan_limit_deg").value)
        self.pan_max_step = math.radians(g("pan_max_step_deg").value)
        self.pan_deadband = math.radians(g("pan_deadband_deg").value)
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
        self.ac = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # --- 状态 ---
        self.cx = None
        self.last_pos_time = 0.0
        self.scan = None
        self.last_scan_time = 0.0
        self.est_odom = None        # 人在 odom 的位置估计 (px,py)——唯一真相源
        self.reject_count = 0
        self.last_goal_xy = None
        self.last_sent_time = 0.0
        self.cam_yaw = 0.0          # 云台 pan 角(rad, base 系 +左)：跟随时→0，丢失时扫
        self.search_dir = 1.0
        self.active_goal_handle = None
        self.stopped = False        # 是否已取消目标停车（避免重复取消）

        # 云台俯仰固定(低头)；pan 初始回中
        s2 = Int32(); s2.data = self.gimbal_tilt; self.pub_s2.publish(s2)
        s1 = Int32(); s1.data = 0; self.pub_s1.publish(s1)

        self.timer = self.create_timer(0.2, self.tick)                     # 5Hz 估计+决策
        self.gtimer = self.create_timer(self.gimbal_period, self._gimbal_tick)  # 10Hz 云台
        self.get_logger().info(
            "person_goal_bridge v3 up: hfov=%.1f sign=%.0f standoff=%.2f pan=%s tilt=%d"
            % (self.hfov_deg, self.bearing_sign, self.standoff, self.enable_gimbal_pan, self.gimbal_tilt))

    def on_position(self, msg):
        self.cx = msg.anglex
        self.last_pos_time = time.time()

    def on_scan(self, msg):
        self.scan = msg
        self.last_scan_time = time.time()

    # ---- 云台 v3：指向 odom 估计(与底盘解耦、不震荡)、丢失才慢扫；把人锁在视野内直到底盘转过来 ----
    def _gimbal_tick(self):
        if not self.enable_gimbal_pan:
            return
        now = time.time()
        detected = self.cx is not None and (now - self.last_pos_time) <= self.lost_timeout
        if not detected:
            # 丢失：在 ±pan_limit 间慢扫搜索
            self.cam_yaw += self.search_dir * self.pan_max_step
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

    def tick(self):
        now = time.time()
        if self.scan is None:
            return
        # 相机丢人/雷达失效：停车、清估计（让云台进搜索、下次从头重建估计）
        if (self.cx is None or now - self.last_pos_time > self.lost_timeout
                or now - self.last_scan_time > self.lost_timeout):
            self._stop()
            self.est_odom = None
            return

        z = self._measure_person_odom()
        if z is not None:
            self._update_estimate(z)
        if self.est_odom is None:
            return  # 尚未建立估计

        rob = self._robot_pose()
        if rob is None:
            return
        dx = self.est_odom[0] - rob[0]
        dy = self.est_odom[1] - rob[1]
        dist = math.hypot(dx, dy)

        # 到 standoff：取消目标停车（滞回，别在人身边抽搐/原地拧）
        if dist <= self.standoff + self.stop_deadband:
            self._stop()
            return

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
        node.destroy_node()
        if rclpy.ok():          # 信号处理器可能已 shutdown 过，避免二次 shutdown 抛 RCLError
            rclpy.shutdown()


if __name__ == "__main__":
    main()
