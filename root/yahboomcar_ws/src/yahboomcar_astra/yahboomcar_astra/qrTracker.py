import rclpy
from rclpy.node import Node


import ipywidgets.widgets as widgets
from IPython.display import display
from std_msgs.msg import Int32, Bool,UInt16

from geometry_msgs.msg import Twist

import cv2
import time
import numpy as np

import pyzbar.pyzbar as pyzbar
from PIL import Image

class QR_Tracker(Node):
    def __init__(self,name):
        super().__init__(name)
        self.pub_cmdVel = self.create_publisher(Twist,"/cmd_vel",1)
        self.pub_Buzzer = self.create_publisher(UInt16,"/beep",1)
        self.pub_Servo1 = self.create_publisher(Int32,"servo_s1" , 10)
        self.pub_Servo2 = self.create_publisher(Int32,"servo_s2" , 10)

        self.PWMServo_X = 0
        self.PWMServo_Y = -50
        self.s1_init_angle = Int32()
        self.s1_init_angle.data = self.PWMServo_X
        self.s2_init_angle = Int32()
        self.s2_init_angle.data = self.PWMServo_Y

        self.pub_Servo2.publish(self.s2_init_angle)
        self.pub_Servo1.publish(self.s1_init_angle)

        #self.capture = cv2.VideoCapture(0)
        #self.timer = self.create_timer(0.001, self.task_processing)



    def detect_qrcode(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            return barcodeData, (x, y, w, h)
        return None, (0, 0, 0, 0)

    def pub_vel(self, x, y, z):
        twist = Twist()
        twist.linear.x = x
        twist.linear.y = y
        twist.angular.z = z
        self.pub_cmdVel.publish(twist)
    

    def robot_action(self,data):
        if data == "forward":
            self.pub_vel(0.3,0.0,0.0)

        elif data == "back":
            self.pub_vel(-0.3,0.0,0.0)
            
        elif data == "left":
            self.pub_vel(0.0,0.0,1.0)
            
        elif data == "right":
            self.pub_vel(0.0,0.0,-1.0)

        elif data == "turnright":
            self.pub_vel(0.3,0.0,-0.5)

        elif data == "turnleft":
            self.pub_vel(0.3,0.0,0.5)

        elif data == "stop":
            self.pub_vel(0.0,0.0,0.0)




def main():
    rclpy.init()
    QRdetect = QR_Tracker("QR_Tracker")
    print("start it")
    capture = cv2.VideoCapture(0)
    t_start = time.time()
    m_fps = 0
    fps = 0
    while capture.isOpened():
            ret, frame = capture.read()
            action = cv2.waitKey(10) & 0xFF
            payload, (x, y, w, h) = QRdetect.detect_qrcode(frame.copy())
            if payload != None:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 255), 2)
                cv2.putText(frame, payload, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 225, 255), 2)
                QRdetect.robot_action(payload)
            else:
                QRdetect.pub_vel(0.0,0.0,0.0)

            m_fps = m_fps + 1
            fps = m_fps / (time.time() - t_start)
            if (time.time() - t_start) >= 2:
                m_fps = fps
                t_start = time.time() - 1
            cv2.putText(frame, "FPS " + str(int(fps)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.imshow('frame', frame)
            if action == ord('q') or action == 113:
                capture.release()
                cv2.destroyAllWindows()
    rclpy.spin(QRdetect)




















