#!/usr/bin/env python3

"""
Object Detector Node for DuckieBot
Detects and tracks objects using computer vision techniques
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs

class ObjectDetector:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('object_detector', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.target_pub = rospy.Publisher('/object_follower/target_position', Point, queue_size=1)
        self.target_found_pub = rospy.Publisher('/object_follower/target_found', Bool, queue_size=1)
        self.distance_pub = rospy.Publisher('/object_follower/target_distance', Float32, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/object_follower/debug_image', Image, queue_size=1)
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        # Parameters
        self.detection_method = rospy.get_param('~detection_method', 'color')  # 'color' or 'cnn'
        
        # Color detection parameters (HSV)
        self.lower_color = np.array([0, 100, 100])  # Red lower bound
        self.upper_color = np.array([10, 255, 255])  # Red upper bound
        
        # Object tracking parameters
        self.last_known_position = None
        self.tracking_confidence = 0.0
        
        # Camera parameters (to be calibrated)
        self.focal_length = 525.0  # pixels
        self.known_object_width = 0.1  # meters (10cm)
        
        rospy.loginfo("Object Detector initialized")
    
    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            if self.detection_method == 'color':
                self.detect_by_color(cv_image)
            elif self.detection_method == 'cnn':
                self.detect_by_cnn(cv_image)
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")
    
    def detect_by_color(self, image):
        """Simple color-based object detection"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for target color
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        debug_image = image.copy()
        target_found = False
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 500:  # Minimum area threshold
                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Calculate center point
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Estimate distance based on object width
                distance = self.estimate_distance(w)
                
                # Publish target position (normalized coordinates)
                target_point = Point()
                target_point.x = (center_x - image.shape[1] // 2) / (image.shape[1] // 2)  # -1 to 1
                target_point.y = (center_y - image.shape[0] // 2) / (image.shape[0] // 2)  # -1 to 1
                target_point.z = distance
                
                self.target_pub.publish(target_point)
                self.distance_pub.publish(Float32(distance))
                
                # Draw bounding box and center
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(debug_image, f"Dist: {distance:.2f}m", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                target_found = True
                self.last_known_position = target_point
                self.tracking_confidence = 1.0
        
        # Publish target found status
        self.target_found_pub.publish(Bool(target_found))
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            rospy.logwarn(f"Could not publish debug image: {str(e)}")
    
    def detect_by_cnn(self, image):
        """CNN-based object detection (placeholder for future implementation)"""
        # TODO: Implement YOLO/MobileNet detection
        rospy.logwarn("CNN detection not yet implemented")
        self.target_found_pub.publish(Bool(False))
    
    def estimate_distance(self, pixel_width):
        """Estimate distance to object based on its pixel width"""
        if pixel_width > 0:
            distance = (self.known_object_width * self.focal_length) / pixel_width
            return max(0.1, min(distance, 5.0))  # Clamp between 10cm and 5m
        return 0.0
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = ObjectDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass