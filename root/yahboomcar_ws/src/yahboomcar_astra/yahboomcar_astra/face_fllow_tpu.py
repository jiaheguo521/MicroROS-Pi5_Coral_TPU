#ros lib
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, LaserScan, Image
from yahboomcar_msgs.msg import Position
from std_msgs.msg import Int32, Bool,UInt16
#common lib
import os
import threading
import math
import cv2
from yahboomcar_astra.astra_common import *
from yahboomcar_msgs.msg import Position
from cv_bridge import CvBridge
from std_msgs.msg import Int32, Bool,UInt16
import tflite_runtime.interpreter as tflite
print("import done")


class EdgeTPUDetector:
    '''Coral Edge TPU 人脸检测器（SSD MobileNet v2 postprocess）
    输入 [1,H,W,3] uint8 RGB；输出 boxes/classes/scores/count 四张量。'''

    def __init__(self, model_path):
        delegate = tflite.load_delegate('libedgetpu.so.1')
        self.interpreter = tflite.Interpreter(
            model_path=model_path, experimental_delegates=[delegate])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, self.in_h, self.in_w, _ = self.input_details[0]['shape']
        print("edgetpu model loaded:", model_path)

    def detect(self, frame, conf_thr):
        '''返回最高分人脸框 (x1,y1,x2,y2) 像素坐标，无脸返回 None'''
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.in_w, self.in_h))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        count = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
        best_i = -1
        best_score = conf_thr
        for i in range(count):
            if scores[i] >= best_score:
                best_score = scores[i]
                best_i = i
        if best_i < 0:
            return None
        h, w = frame.shape[:2]
        ymin, xmin, ymax, xmax = boxes[best_i]
        x1 = int(xmin * w); y1 = int(ymin * h)
        x2 = int(xmax * w); y2 = int(ymax * h)
        return (x1, y1, x2, y2)


class faceTracker(Node):
    def __init__(self,name):
        super().__init__(name)
        #create the publisher
        self.pub_Servo1 = self.create_publisher(Int32,"servo_s1" , 10)
        self.pub_Servo2 = self.create_publisher(Int32,"servo_s2" , 10)
        #create the subscriber
        #self.sub_depth = self.create_subscription(Image,"/image_raw", self.depth_img_Callback, 1)
        self.sub_JoyState = self.create_subscription(Bool,'/JoyState',  self.JoyStateCallback,1)
        #self.sub_position = self.create_subscription(Position,"/Current_point",self.positionCallback,1)
        self.bridge = CvBridge()
        self.minDist = 1500
        self.Center_x = 0
        self.Center_y = 0
        self.Center_r = 0
        self.Center_prevx = 0
        self.Center_prevr = 0
        self.prev_time = 0
        self.prev_dist = 0
        self.prev_angular = 0
        self.Joy_active = False
        self.Robot_Run = False
        self.img_flip = False
        self.dist = []
        self.encoding = ['8UC3']
        self.linear_PID = (20.0, 0.0, 1.0)
        self.angular_PID = (0.5, 0.0, 2.0)
        self.scale = 1000
        self.end = 0
        self.PWMServo_X = 0
        self.PWMServo_Y = -60
        self.s1_init_angle = Int32()
        self.s1_init_angle.data = self.PWMServo_X
        self.s2_init_angle = Int32()
        self.s2_init_angle.data = self.PWMServo_Y
        self.PID_init()
        self.declare_param()
        self.detector = EdgeTPUDetector(self.model_path)
        self.capture = cv.VideoCapture(0)
        self.timer = self.create_timer(0.001, self.on_timer)
        print("init done")

    def declare_param(self):
        #PID
        self.declare_parameter("Kp",20)
        self.Kp = self.get_parameter('Kp').get_parameter_value().integer_value
        self.declare_parameter("Ki",0)
        self.Ki = self.get_parameter('Ki').get_parameter_value().integer_value
        self.declare_parameter("Kd",0.9)
        self.Kd = self.get_parameter('Kd').get_parameter_value().integer_value
        #Edge TPU
        self.declare_parameter("model_path",
            "/root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        # 0.3: this SSD scores near/large faces low (~0.09) though it localises them
        # well; at 0.5 near-face recall is ~0.33, at 0.3 ~0.64 with precision still 1.0
        # (benchmark deploy_set). See docs/TPU_vs_CPU_Benchmark_Results.md.
        self.declare_parameter("conf_threshold",0.3)
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.declare_parameter("show_image",True)
        self.show_image = self.get_parameter('show_image').get_parameter_value().bool_value

    def get_param(self):
        self.Kd = self.get_parameter('Kd').get_parameter_value().integer_value
        self.Ki = self.get_parameter('Ki').get_parameter_value().integer_value
        self.Kp = self.get_parameter('Kp').get_parameter_value().integer_value
        self.linear_PID = (self.Kp,self.Ki,self.Kd)

    def on_timer(self):
        self.get_param()

        ret, frame = self.capture.read()
        start = time.time()
        fps = 1 / (start - self.end)
        self.end = start
        box = self.detector.detect(frame, self.conf_threshold)
        if box is not None:
            (x1, y1, x2, y2) = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            self.execute(cx, cy)
        if self.show_image:
            cv.putText(frame, "FPS : " + str(int(fps)), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
            if box is not None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.imshow('frame', frame)
            if (cv.waitKey(10) & 0xFF) in (ord('q'), 113):
                self.capture.release()
                cv.destroyAllWindows()



    def PID_init(self):
        self.PID_controller = simplePID(
            [0, 0],
            [self.linear_PID[0] / float(self.scale), self.linear_PID[0] / float(self.scale)],
            [self.linear_PID[1] / float(self.scale), self.linear_PID[1] / float(self.scale)],
            [self.linear_PID[2] / float(self.scale), self.linear_PID[2] / float(self.scale)])
        self.pub_Servo1.publish(self.s1_init_angle)
        self.pub_Servo2.publish(self.s2_init_angle)

    def JoyStateCallback(self, msg):
        if not isinstance(msg, Bool): return
        self.Joy_active = msg.data
        self.pub_cmdVel.publish(Twist())

    def positionCallback(self, msg):
        if not isinstance(msg, Position): return
        self.Center_x = msg.anglex
        self.Center_y = msg.angley
        self.Center_r = msg.distance

    def execute(self, point_x, point_y):
        [x_Pid, y_Pid] = self.PID_controller.update([point_x - 320, point_y - 240])
        if self.img_flip == True:
            self.PWMServo_X += x_Pid
            self.PWMServo_Y += y_Pid
        else:
            self.PWMServo_X  -= x_Pid
            self.PWMServo_Y  += y_Pid

        if self.PWMServo_X  >= 90:
            self.PWMServo_X  = 90
        elif self.PWMServo_X  <= -90:
            self.PWMServo_X  = -90
        if self.PWMServo_Y >= 20:
            self.PWMServo_Y = 20
        elif self.PWMServo_Y <= -90:
            self.PWMServo_Y = -90

        # rospy.loginfo("target_servox: {}, target_servoy: {}".format(self.target_servox, self.target_servoy))
        print("servo1",self.PWMServo_X)
        servo1_angle = Int32()
        servo1_angle.data = int(self.PWMServo_X)
        servo2_angle = Int32()
        servo2_angle.data = int(self.PWMServo_Y)
        self.pub_Servo1.publish(servo1_angle)
        self.pub_Servo2.publish(servo2_angle)



class simplePID:
    '''very simple discrete PID controller'''

    def __init__(self, target, P, I, D):
        '''Create a discrete PID controller
        each of the parameters may be a vector if they have the same length
        Args:
        target (double) -- the target value(s)
        P, I, D (double)-- the PID parameter
        '''
        # check if parameter shapes are compatabile.
        if (not (np.size(P) == np.size(I) == np.size(D)) or ((np.size(target) == 1) and np.size(P) != 1) or (
                np.size(target) != 1 and (np.size(P) != np.size(target) and (np.size(P) != 1)))):
            raise TypeError('input parameters shape is not compatable')
        #rospy.loginfo('P:{}, I:{}, D:{}'.format(P, I, D))
        self.Kp = np.array(P)
        self.Ki = np.array(I)
        self.Kd = np.array(D)
        self.last_error = 0
        self.integrator = 0
        self.timeOfLastCall = None
        self.setPoint = np.array(target)
        self.integrator_max = float('inf')

    def update(self, current_value):
        '''Updates the PID controller.
        Args:
            current_value (double): vector/number of same legth as the target given in the constructor
        Returns:
            controll signal (double): vector of same length as the target
        '''
        current_value = np.array(current_value)
        if (np.size(current_value) != np.size(self.setPoint)):
            raise TypeError('current_value and target do not have the same shape')
        if (self.timeOfLastCall is None):
            # the PID was called for the first time. we don't know the deltaT yet
            # no controll signal is applied
            self.timeOfLastCall = time.perf_counter()
            return np.zeros(np.size(current_value))
        error = self.setPoint - current_value
        P = error
        currentTime = time.perf_counter()
        deltaT = (currentTime - self.timeOfLastCall)
        # integral of the error is current error * time since last update
        self.integrator = self.integrator + (error * deltaT)
        I = self.integrator
        # derivative is difference in error / time since last update
        D = (error - self.last_error) / deltaT
        self.last_error = error
        self.timeOfLastCall = currentTime
        # return controll signal
        return self.Kp * P + self.Ki * I + self.Kd * D



def main():
    rclpy.init()
    face_Tracker = faceTracker("FaceTracker")
    rclpy.spin(face_Tracker)
