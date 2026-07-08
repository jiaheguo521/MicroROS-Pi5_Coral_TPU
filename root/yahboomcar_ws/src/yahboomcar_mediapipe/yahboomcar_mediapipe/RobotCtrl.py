#!/usr/bin/env python3
# encoding: utf-8
import threading
import cv2 as cv
import numpy as np
from yahboomcar_mediapipe.media_library import *
from time import sleep, time
import rclpy
from rclpy.node import Node


class HandCtrlArm(Node):
    def __init__(self,name):
        super().__init__(name)
        self.pub_Servo1 = self.create_publisher(Int32,"servo_s1" , 10)
        self.pub_Servo2 = self.create_publisher(Int32,"servo_s2" , 10)

        self.PWMServo_X = 0
        self.PWMServo_Y = -45
        self.s1_init_angle = Int32()
        self.s1_init_angle.data = self.PWMServo_X
        self.s2_init_angle = Int32()
        self.s2_init_angle.data = self.PWMServo_Y
        self.media_ros = Media_ROS()

        self.pub_Servo1.publish(self.s1_init_angle)
        self.pub_Servo2.publish(self.s2_init_angle)

        self.hand_detector = HandDetector()
        self.arm_status = True
        self.locking = True
        self.init = True
        self.pTime = 0
        self.add_lock = self.remove_lock = 0
        self.event = threading.Event()
        self.event.set()

    def process(self, frame):
        frame = cv.flip(frame, 1)
        frame, lmList, bbox = self.hand_detector.findHands(frame)
        if len(lmList) != 0:
            threading.Thread(target=self.arm_ctrl_threading, args=(lmList,bbox)).start()
        else:
            self.media_ros.pub_vel(0.0,0.0,0.0)
        self.cTime = time()
        fps = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime
        text = "FPS : " + str(int(fps))
        cv.putText(frame, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        self.media_ros.pub_imgMsg(frame)
        return frame

    def arm_ctrl_threading(self, lmList,bbox):
        if self.event.is_set():
            self.event.clear()
            fingers = self.hand_detector.fingersUp(lmList)
            self.hand_detector.draw = True
            gesture = self.hand_detector.get_gesture(lmList)
            self.arm_status = False
            point_x = lmList[9][1]
            point_y = lmList[9][2]

            print("x",point_x)
            if point_y >= 270: x = -0.2
            elif point_y <= 210: x = 0.2
            else: x = 0.0
            if point_x >= 350: y = -0.2
            elif point_x <= 290: y = 0.2
            else: y = 0.0
            self.media_ros.pub_vel(x,0.0,y)
            print("angle: {},value: {}".format(x,y))
            self.arm_status = True
            self.event.set()


def main():
    rclpy.init()
    handctrlarm = HandCtrlArm('handctrl')
    capture = cv.VideoCapture(0)
    print("capture get FPS : ", capture.get(cv.CAP_PROP_FPS))
    while capture.isOpened():
        ret, frame = capture.read()
        action = cv.waitKey(1) & 0xFF
        frame = handctrlarm.process(frame)
        if action == ord('q'):
            handctrlarm.media_ros.cancel()
            break
        cv.imshow('frame', frame)
    capture.release()
    cv.destroyAllWindows()
    rclpy.spin(handctrlarm)
