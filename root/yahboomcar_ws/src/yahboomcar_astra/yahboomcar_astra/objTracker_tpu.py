#ros lib
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, LaserScan, Image
from yahboomcar_msgs.msg import Position
from cv_bridge import CvBridge
#common lib
import os
import threading
import math
import cv2
from yahboomcar_astra.astra_common import *
from yahboomcar_msgs.msg import Position
import tflite_runtime.interpreter as tflite
print("import done")


class EdgeTPUDetector:
    '''Coral Edge TPU 目标检测器（SSD MobileNet v2 COCO postprocess）
    输入 [1,H,W,3] uint8 RGB；输出 boxes/classes/scores/count 四张量。
    只保留 target_id 那一类，返回最高分框。'''

    def __init__(self, model_path, label_path):
        self.labels = self._load_labels(label_path)
        delegate = tflite.load_delegate('libedgetpu.so.1')
        self.interpreter = tflite.Interpreter(
            model_path=model_path, experimental_delegates=[delegate])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, self.in_h, self.in_w, _ = self.input_details[0]['shape']
        print("edgetpu model loaded:", model_path)

    @staticmethod
    def _load_labels(label_path):
        '''Coral coco_labels.txt：每行 "id  name" → {name: id}'''
        labels = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    labels[parts[1]] = int(parts[0])
                else:
                    labels[parts[0]] = len(labels)
        return labels

    def label_id(self, target_label):
        return self.labels.get(target_label, -1)

    def detect(self, frame, conf_thr, target_id):
        '''返回 target_id 类的最高分框 (x1,y1,x2,y2) 像素坐标，无则返回 None'''
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.in_w, self.in_h))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        count = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
        best_i = -1
        best_score = conf_thr
        for i in range(count):
            if int(classes[i]) == target_id and scores[i] >= best_score:
                best_score = scores[i]
                best_i = i
        if best_i < 0:
            return None
        h, w = frame.shape[:2]
        ymin, xmin, ymax, xmax = boxes[best_i]
        x1 = int(xmin * w); y1 = int(ymin * h)
        x2 = int(xmax * w); y2 = int(ymax * h)
        return (x1, y1, x2, y2)


class Object_Identify(Node):
    def __init__(self, name):
        super().__init__(name)
        #create a publisher
        self.pub_position = self.create_publisher(Position, "/Current_point", 10)
        self.pub_img = self.create_publisher(Image, '/image_raw', 500)
        self.bridge = CvBridge()
        self.end = 0
        self.hit_count = 0
        self.declare_param()
        self.detector = EdgeTPUDetector(self.model_path, self.label_path)
        self.target_id = self.detector.label_id(self.target_label)
        if self.target_id < 0:
            print("WARN: target_label '%s' not in labels, no detection will match" % self.target_label)
        self.capture = cv.VideoCapture(0)
        self.timer = self.create_timer(0.001, self.on_timer)
        print("init done")

    def declare_param(self):
        self.declare_parameter("target_label", "person")
        self.target_label = self.get_parameter('target_label').get_parameter_value().string_value
        self.declare_parameter("conf_threshold", 0.5)
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.declare_parameter("model_path",
            "/root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.declare_parameter("label_path",
            "/root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/models/coco_labels.txt")
        self.label_path = self.get_parameter('label_path').get_parameter_value().string_value
        self.declare_parameter("show_image", True)
        self.show_image = self.get_parameter('show_image').get_parameter_value().bool_value
        self.declare_parameter("min_hits", 3)   # 连续 N 帧检到才发目标，滤掉一闪而过的误检
        self.min_hits = self.get_parameter('min_hits').get_parameter_value().integer_value

    def on_timer(self):
        ret, frame = self.capture.read()
        if not ret:
            return
        frame = cv.resize(frame, (640, 480))
        start = time.time()
        fps = 1 / (start - self.end)
        self.end = start
        box = self.detector.detect(frame, self.conf_threshold, self.target_id)
        #去抖：连续 min_hits 帧检到人才发目标，漏检清零
        if box is not None:
            self.hit_count += 1
        else:
            self.hit_count = 0
        if box is not None and self.hit_count >= self.min_hits:
            (x1, y1, x2, y2) = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            size = max(x2 - x1, y2 - y1)
            position = Position()
            position.anglex = cx * 1.0
            position.angley = cy * 1.0
            position.distance = size * 1.0
            self.pub_position.publish(position)
        # 每帧都发 /image_raw：colorTracker 的 execute() 只在 /image_raw 回调里跑
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        if self.show_image:
            cv.putText(frame, "FPS : " + str(int(fps)), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
            if box is not None:
                #未达 min_hits 也画框（黄），达标发目标（绿）
                color = (0, 255, 0) if self.hit_count >= self.min_hits else (0, 255, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv.imshow('frame', frame)
            if (cv.waitKey(10) & 0xFF) in (ord('q'), 113):
                self.capture.release()
                cv.destroyAllWindows()


def main():
    rclpy.init()
    object_identify = Object_Identify("ObjectIdentify")
    print("start it")
    rclpy.spin(object_identify)
