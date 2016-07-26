#!/usr/bin/env python

import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

sub_topic_ = '/usb_cam/image_raw'

def image_callback(img_msg):
    bridge = CvBridge()
    cv_img = None
    try:
        cv_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except Exception as e:
        print (e)
        
    if cv_img is None:
        rospy.logwarn("empty image")
        return

    #cv_img = cv2.GaussianBlur(cv_img, (7, 7), 1, 1)
    im_edge = cv2.Canny(cv_img, 10, 150)

    # minLineLength = 10
    # maxLineGap = 10
    # lines = cv2.HoughLinesP(im_edge, 2, np.pi/180, 10, minLineLength, maxLineGap)

    # for x1,y1,x2,y2 in lines[0]:
    #      cv2.line(cv_img, (x1,y1) ,(x2,y2), (0,255,0), 2)
    

    gray = cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    cv_img[dst>0.01*dst.max()]=[0,0,255]


    wname = 'image'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.imshow(wname, cv_img)

    wname = 'edges'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.imshow(wname, im_edge)

    cv2.waitKey(3)

def onInit():
    rospy.Subscriber(sub_topic_, Image, image_callback)

def main():
    rospy.init_node('landing_marker_close_perception', anonymous=True)
    onInit()
    rospy.spin()

if __name__ == "__main__":
    main()
