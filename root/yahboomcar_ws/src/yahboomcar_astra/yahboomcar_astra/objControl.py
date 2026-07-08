#ros lib
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from yahboomcar_msgs.msg import Position
from std_msgs.msg import Int32
#common lib
import math
import time
from yahboomcar_astra.astra_common import *
print("import done")


class object_Control(Node):
    '''目标(人)跟随控制器：云台 + 底盘。
    订阅 /Current_point(相机方位) + /scan(雷达测距)，输出舵机 + /cmd_vel。
    可原样替代 colorTracker（云台逻辑相同），多做底盘定距跟随。'''

    def __init__(self, name):
        super().__init__(name)
        #publisher
        self.pub_cmdVel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_Servo1 = self.create_publisher(Int32, "servo_s1", 10)
        self.pub_Servo2 = self.create_publisher(Int32, "servo_s2", 10)
        #subscriber
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.registerScan, 1)
        self.sub_JoyState = self.create_subscription(Bool, '/JoyState', self.JoyStateCallback, 1)
        self.sub_position = self.create_subscription(Position, "/Current_point", self.positionCallback, 1)
        #target from camera
        self.Center_x = 0
        self.Center_y = 0
        self.Center_r = 0
        self.last_pos_time = 0.0
        self.new_pos = False
        #laser range straight ahead (smoothed) + last full scan for avoidance
        self.laser_dist = None
        self.last_laser_time = 0.0
        self.scan = None
        #state
        self.Joy_active = False
        self.Robot_Run = False
        self.img_flip = False
        #gimbal
        self.linear_PID = (20.0, 0.0, 1.0)
        self.scale = 1000
        self.search_dir = 1
        self.PWMServo_X = 0
        self.PWMServo_Y = -50
        self.s1_init_angle = Int32(); self.s1_init_angle.data = self.PWMServo_X
        self.s2_init_angle = Int32(); self.s2_init_angle.data = self.PWMServo_Y
        self.declare_param()
        self.PID_init()
        self.pub_Servo1.publish(self.s1_init_angle)
        self.pub_Servo2.publish(self.s2_init_angle)
        #control loop (decoupled from image rate)
        self.timer = self.create_timer(0.05, self.control_loop)
        print("init done")

    def declare_param(self):
        #gimbal PID
        self.declare_parameter("linear_Kp", 20.0)
        self.declare_parameter("linear_Ki", 0.0)
        self.declare_parameter("linear_Kd", 1.0)          # 降低：Kd 对检测框噪声最敏感，是颤动主因
        self.declare_parameter("scale", 1000)
        self.declare_parameter("gimbal_max_step", 3.0)    # 云台每次最多转 (deg，限速抑颤)
        #lost-target search
        self.declare_parameter("search_enable", True)     # 丢目标时云台扫描找人
        self.declare_parameter("search_range", 60.0)      # pan 扫描幅度 (±deg)
        self.declare_parameter("search_step", 1.0)        # 每拍扫描步进 (deg，20Hz)
        self.declare_parameter("search_tilt", -30.0)      # 搜索时俯仰角 (deg)
        #base follow
        self.declare_parameter("target_dist", 1.0)        # 想保持的距离 (m)
        self.declare_parameter("dead_zone", 0.15)         # 距离死区 (±m)
        self.declare_parameter("linear_kp", 0.6)          # 前后 P 增益 (m/s per m)
        self.declare_parameter("max_linear", 0.3)         # 前后限速 (m/s)
        self.declare_parameter("front_angle", 0.0)        # 雷达正前方角 (deg，装反改180)
        self.declare_parameter("front_half_width", 12.0)  # 前窗半宽 (deg)
        self.declare_parameter("angular_kp", -0.03)       # 底盘转向增益 (对云台偏角，符号反了取正)
        self.declare_parameter("max_angular", 1.0)        # 转向限速 (rad/s)
        self.declare_parameter("gimbal_deadband", 5.0)    # 云台偏角死区 (deg)
        self.declare_parameter("laser_smooth", 0.5)       # 距离 EMA (0~1，越大越平滑)
        self.declare_parameter("lost_timeout", 0.5)       # 多久没目标算丢 (s)
        self.declare_parameter("laser_timeout", 0.5)      # 雷达数据多旧算失效 (s)
        self.declare_parameter("gimbal_px_deadband", 15.0)  # 云台像素死区 (px，抑制颤动)
        # 避障已由 nav2_collision_monitor 接管(见 launch/follow_collision.launch.py)，
        # 本节点不再做反应式避障。
        self.get_param()

    def get_param(self):
        self.linear_PID = (
            self.get_parameter('linear_Kp').get_parameter_value().double_value,
            self.get_parameter('linear_Ki').get_parameter_value().double_value,
            self.get_parameter('linear_Kd').get_parameter_value().double_value)
        self.scale = self.get_parameter('scale').get_parameter_value().integer_value
        self.gimbal_max_step = self.get_parameter('gimbal_max_step').get_parameter_value().double_value
        self.search_enable = self.get_parameter('search_enable').get_parameter_value().bool_value
        self.search_range = self.get_parameter('search_range').get_parameter_value().double_value
        self.search_step = self.get_parameter('search_step').get_parameter_value().double_value
        self.search_tilt = self.get_parameter('search_tilt').get_parameter_value().double_value
        self.target_dist = self.get_parameter('target_dist').get_parameter_value().double_value
        self.dead_zone = self.get_parameter('dead_zone').get_parameter_value().double_value
        self.linear_kp = self.get_parameter('linear_kp').get_parameter_value().double_value
        self.max_linear = self.get_parameter('max_linear').get_parameter_value().double_value
        self.front_angle = self.get_parameter('front_angle').get_parameter_value().double_value
        self.front_half_width = self.get_parameter('front_half_width').get_parameter_value().double_value
        self.angular_kp = self.get_parameter('angular_kp').get_parameter_value().double_value
        self.max_angular = self.get_parameter('max_angular').get_parameter_value().double_value
        self.gimbal_deadband = self.get_parameter('gimbal_deadband').get_parameter_value().double_value
        self.laser_smooth = self.get_parameter('laser_smooth').get_parameter_value().double_value
        self.lost_timeout = self.get_parameter('lost_timeout').get_parameter_value().double_value
        self.laser_timeout = self.get_parameter('laser_timeout').get_parameter_value().double_value
        self.gimbal_px_deadband = self.get_parameter('gimbal_px_deadband').get_parameter_value().double_value

    def PID_init(self):
        self.gimbal_pid = simplePID(
            [0, 0],
            [self.linear_PID[0] / float(self.scale), self.linear_PID[0] / float(self.scale)],
            [self.linear_PID[1] / float(self.scale), self.linear_PID[1] / float(self.scale)],
            [self.linear_PID[2] / float(self.scale), self.linear_PID[2] / float(self.scale)])

    def positionCallback(self, msg):
        if not isinstance(msg, Position): return
        self.Center_x = msg.anglex
        self.Center_y = msg.angley
        self.Center_r = msg.distance
        self.last_pos_time = time.time()
        self.new_pos = True

    def JoyStateCallback(self, msg):
        if not isinstance(msg, Bool): return
        self.Joy_active = msg.data
        self.pub_cmdVel.publish(Twist())

    def registerScan(self, scan_data):
        '''缓存整帧扫描；正前方窄窗内有效最小距离(=人距离) EMA 平滑。'''
        if not isinstance(scan_data, LaserScan): return
        self.scan = scan_data
        self.last_laser_time = time.time()
        d = self._sector_min(0.0, self.front_half_width)  # 正前方窄窗 = 人
        if d is not None:
            if self.laser_dist is None:
                self.laser_dist = d
            else:
                a = self.laser_smooth
                self.laser_dist = a * self.laser_dist + (1 - a) * d

    def _sector_min(self, center_deg, half_deg):
        '''以正前方(front_angle)为 0、center_deg 为中心、±half_deg 扇区内的有效最小距离；
        剔除 0.0/inf/超量程；无有效返回 None。绕 ±π 也正确。'''
        scan = self.scan
        if scan is None: return None
        fa = math.radians(self.front_angle) + math.radians(center_deg)
        hw = math.radians(half_deg)
        amin = scan.angle_min; inc = scan.angle_increment
        rmin = scan.range_min; rmax = scan.range_max
        best = None
        for i, r in enumerate(scan.ranges):
            ang = amin + inc * i
            delta = math.atan2(math.sin(ang - fa), math.cos(ang - fa))
            if abs(delta) <= hw and math.isfinite(r) and rmin < r < rmax:
                if best is None or r < best: best = r
        return best

    def control_loop(self):
        self.get_param()
        now = time.time()
        target_present = (self.Center_r != 0) and (now - self.last_pos_time < self.lost_timeout)
        if self.Joy_active or not target_present:
            if self.Robot_Run:
                self.pub_cmdVel.publish(Twist())
                self.Robot_Run = False
            #丢目标：底盘停，云台慢速左右扫描找人。
            #仅当检测器还活着(在发 /Current_point)时扫；检测器被关掉→计数归 0→停扫，
            #避免"检测器 Ctrl-C 后云台永远转"。手柄接管时也不扫。
            detector_alive = self.count_publishers('/Current_point') > 0
            if (not self.Joy_active) and (not target_present) and self.search_enable and detector_alive:
                self._search_sweep()
            return
        #云台跟随：只在收到新检测时更新一次（20Hz 定时器里反复更新会累加同一误差→颤动）
        if self.new_pos:
            self.gimbal(self.Center_x, self.Center_y)
            self.new_pos = False
        #底盘：转向追云台偏角，前后按雷达定距
        twist = Twist()
        if abs(self.PWMServo_X) > self.gimbal_deadband:
            twist.angular.z = self._clamp(self.angular_kp * self.PWMServo_X, self.max_angular)
        if self.laser_dist is not None and (now - self.last_laser_time) < self.laser_timeout:
            err = self.laser_dist - self.target_dist
            if abs(err) > self.dead_zone:
                twist.linear.x = self._clamp(self.linear_kp * err, self.max_linear)
        #避障(防撞/减速)由 nav2_collision_monitor 在 /cmd_vel_raw→/cmd_vel 上处理，此处不做。
        self.pub_cmdVel.publish(twist)
        self.Robot_Run = True

    @staticmethod
    def _clamp(v, lim):
        return max(-lim, min(lim, v))

    def _search_sweep(self):
        '''丢目标时：底盘不动，云台 pan 慢速左右往复扫描找人。'''
        self.PWMServo_X += self.search_dir * self.search_step
        if self.PWMServo_X >= self.search_range:
            self.PWMServo_X = self.search_range; self.search_dir = -1
        elif self.PWMServo_X <= -self.search_range:
            self.PWMServo_X = -self.search_range; self.search_dir = 1
        self.PWMServo_Y = self.search_tilt
        s1 = Int32(); s1.data = int(self.PWMServo_X)
        s2 = Int32(); s2.data = int(self.PWMServo_Y)
        self.pub_Servo1.publish(s1)
        self.pub_Servo2.publish(s2)

    def gimbal(self, point_x, point_y):
        #像素死区：接近居中就当误差为 0，避免舵机在检测框抖动下反复微调（颤动）
        ex = point_x - 320
        ey = point_y - 240
        if abs(ex) < self.gimbal_px_deadband: ex = 0
        if abs(ey) < self.gimbal_px_deadband: ey = 0
        [x_Pid, y_Pid] = self.gimbal_pid.update([ex, ey])
        #限速：每次最多转 gimbal_max_step 度，压平过冲/颤动
        x_Pid = self._clamp(x_Pid, self.gimbal_max_step)
        y_Pid = self._clamp(y_Pid, self.gimbal_max_step)
        if self.img_flip == True:
            self.PWMServo_X += x_Pid
            self.PWMServo_Y += y_Pid
        else:
            self.PWMServo_X -= x_Pid
            self.PWMServo_Y += y_Pid
        if self.PWMServo_X >= 90: self.PWMServo_X = 90
        elif self.PWMServo_X <= -90: self.PWMServo_X = -90
        if self.PWMServo_Y >= 20: self.PWMServo_Y = 20
        elif self.PWMServo_Y <= -90: self.PWMServo_Y = -90
        servo1_angle = Int32(); servo1_angle.data = int(self.PWMServo_X)
        servo2_angle = Int32(); servo2_angle.data = int(self.PWMServo_Y)
        self.pub_Servo1.publish(servo1_angle)
        self.pub_Servo2.publish(servo2_angle)


class simplePID:
    '''very simple discrete PID controller'''

    def __init__(self, target, P, I, D):
        if (not (np.size(P) == np.size(I) == np.size(D)) or ((np.size(target) == 1) and np.size(P) != 1) or (
                np.size(target) != 1 and (np.size(P) != np.size(target) and (np.size(P) != 1)))):
            raise TypeError('input parameters shape is not compatable')
        self.Kp = np.array(P)
        self.Ki = np.array(I)
        self.Kd = np.array(D)
        self.last_error = 0
        self.integrator = 0
        self.timeOfLastCall = None
        self.setPoint = np.array(target)
        self.integrator_max = float('inf')

    def update(self, current_value):
        current_value = np.array(current_value)
        if (np.size(current_value) != np.size(self.setPoint)):
            raise TypeError('current_value and target do not have the same shape')
        if (self.timeOfLastCall is None):
            self.timeOfLastCall = time.perf_counter()
            return np.zeros(np.size(current_value))
        error = self.setPoint - current_value
        P = error
        currentTime = time.perf_counter()
        deltaT = (currentTime - self.timeOfLastCall)
        self.integrator = self.integrator + (error * deltaT)
        I = self.integrator
        D = (error - self.last_error) / deltaT
        self.last_error = error
        self.timeOfLastCall = currentTime
        return self.Kp * P + self.Ki * I + self.Kd * D


def main():
    rclpy.init()
    object_control = object_Control("ObjectControl")
    print("start it")
    try:
        rclpy.spin(object_control)
    except KeyboardInterrupt:
        pass
    finally:
        object_control.pub_cmdVel.publish(Twist())   # 退出兜底：停车(底盘会锁存最后速度)
        object_control.pub_Servo1.publish(object_control.s1_init_angle)  # 云台回中，避免停在扫描位
        object_control.pub_Servo2.publish(object_control.s2_init_angle)
        object_control.destroy_node()
        rclpy.shutdown()
