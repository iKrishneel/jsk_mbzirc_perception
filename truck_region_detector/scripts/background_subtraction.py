
#!/usr/bin/env python

import roslib
import rospy

import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TruckRegionDetector:
    def __init__(self):
        self.b_subtractor_ = cv2.BackgroundSubtractorMOG(history=5,
                                                         nmixtures=2, 
                                                         backgroundRatio=0.001)
        self.on_init()
        self.__image_init = None
        self.__icount = 0
        
    def remove_init_region(self, im_cur, im_prev):
        if im_cur.shape != im_prev.shape:
            rospy.logwarn('incorrect size')
            return -1
            
        im_diff = cv2.absdiff(im_cur, im_prev)
        return im_diff

    def image_callback(self, image_msgs):
        bridge = CvBridge()
        cv_img = None
        try:
            cv_img = bridge.imgmsg_to_cv2(image_msgs, "bgr8")
        except Exception as e:
            print (e)
    
        fg_mask = self.b_subtractor_.apply(cv_img)
        
        
        if not self.__image_init is None:
            im_diff = self.remove_init_region(fg_mask,
                                              self.__image_init)

            cv2.namedWindow('init_image', cv2.WINDOW_NORMAL)
            cv2.imshow('init_image', im_diff)
            #cv2.imshow('init_image', self.__image_init)

        if self.__icount == 2:
            print "updating"
            self.__image_init = fg_mask
        self.__icount += 1
    
        cv2.namedWindow('fore_image', cv2.WINDOW_NORMAL)
        cv2.imshow('fore_image', cv_img)
        cv2.waitKey(3)


    def subscribe(self):
        image_topic = '/camera/left/rgb/image_rect'
        rospy.Subscriber(image_topic, Image, self.image_callback)

    def on_init(self):
        self.subscribe()

def main():
    rospy.init_node('background_substractor')
    trd = TruckRegionDetector()

    rospy.spin()

if __name__ == "__main__":
    main()
