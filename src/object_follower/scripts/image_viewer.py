#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageViewer:
    def __init__(self):
        rospy.init_node('image_viewer', anonymous=True)
        self.bridge = CvBridge()
        
        # Subscribe to both raw and debug images
        rospy.Subscriber('/camera/image_raw', Image, self.raw_image_callback)
        rospy.Subscriber('/object_follower/debug_image', Image, self.debug_image_callback)
        
        rospy.loginfo("Image Viewer started - Press 'q' to quit")
        
    def raw_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow('Raw Camera', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"Error displaying raw image: {e}")
            
    def debug_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow('Object Detection Debug', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"Error displaying debug image: {e}")

if __name__ == '__main__':
    try:
        viewer = ImageViewer()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()