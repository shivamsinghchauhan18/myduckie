#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from duckietown_msgs.msg import AprilTagDetectionArray, AprilTagDetection
from std_msgs.msg import Bool
import apriltag

class AprilTagDetector:
    def __init__(self):
        rospy.init_node('apriltag_detector', anonymous=True)
        
        # Initialize AprilTag detector
        self.detector = apriltag.Detector()
        self.bridge = CvBridge()
        
        # Publishers
        self.tag_pub = rospy.Publisher('/apriltag_detections', AprilTagDetectionArray, queue_size=1)
        self.stop_signal_pub = rospy.Publisher('/apriltag_stop_signal', Bool, queue_size=1)
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        # Parameters
        self.stop_distance_threshold = rospy.get_param('~stop_distance_threshold', 0.3)  # meters
        self.tag_size = rospy.get_param('~tag_size', 0.065)  # meters
        
        rospy.loginfo("AprilTag Detector initialized")
    
    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect AprilTags
            detections = self.detector.detect(gray)
            
            # Create detection array message
            detection_array = AprilTagDetectionArray()
            detection_array.header = msg.header
            
            should_stop = False
            
            for detection in detections:
                # Create detection message
                tag_msg = AprilTagDetection()
                tag_msg.id = detection.tag_id
                tag_msg.family = detection.tag_family.decode('utf-8')
                
                # Calculate pose (simplified distance estimation)
                # Using tag corners to estimate distance
                corners = detection.corners
                tag_width_pixels = np.linalg.norm(corners[1] - corners[0])
                
                # Rough distance estimation (camera focal length approximation)
                focal_length = 320  # approximate for Duckietown camera
                distance = (self.tag_size * focal_length) / tag_width_pixels
                
                # Check if we should stop
                if distance <= self.stop_distance_threshold:
                    should_stop = True
                    rospy.loginfo(f"AprilTag {detection.tag_id} detected at distance {distance:.2f}m - STOPPING")
                
                # Fill pose information (simplified)
                tag_msg.pose.pose.position.x = 0
                tag_msg.pose.pose.position.y = 0
                tag_msg.pose.pose.position.z = distance
                
                detection_array.detections.append(tag_msg)
            
            # Publish detections
            if detection_array.detections:
                self.tag_pub.publish(detection_array)
            
            # Publish stop signal
            stop_msg = Bool()
            stop_msg.data = should_stop
            self.stop_signal_pub.publish(stop_msg)
            
        except Exception as e:
            rospy.logerr(f"Error in image processing: {e}")
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = AprilTagDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass