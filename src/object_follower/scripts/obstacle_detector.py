#!/usr/bin/env python3

"""
Obstacle Detector Node for DuckieBot
Detects obstacles using computer vision and depth estimation
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge

class ObstacleDetector:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('obstacle_detector', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.obstacle_pub = rospy.Publisher('/object_follower/obstacle_detected', Bool, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/object_follower/obstacle_debug_image', Image, queue_size=1)
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        # Parameters
        self.detection_method = rospy.get_param('~detection_method', 'edge')  # 'edge' or 'depth'
        self.min_obstacle_area = rospy.get_param('~min_obstacle_area', 1000)
        self.safe_distance_pixels = rospy.get_param('~safe_distance_pixels', 100)
        
        # Image processing parameters
        self.roi_height_ratio = 0.6  # Use bottom 60% of image
        self.edge_threshold_low = 50
        self.edge_threshold_high = 150
        
        rospy.loginfo("Obstacle Detector initialized")
    
    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            if self.detection_method == 'edge':
                self.detect_by_edges(cv_image)
            elif self.detection_method == 'depth':
                self.detect_by_depth(cv_image)
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")
    
    def detect_by_edges(self, image):
        """Simple edge-based obstacle detection"""
        height, width = image.shape[:2]
        
        # Define ROI (Region of Interest) - bottom portion of image
        roi_start = int(height * (1 - self.roi_height_ratio))
        roi = image[roi_start:height, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.edge_threshold_low, self.edge_threshold_high)
        
        # Morphological operations to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create debug image
        debug_image = image.copy()
        cv2.rectangle(debug_image, (0, roi_start), (width, height), (255, 0, 0), 2)
        
        obstacle_detected = False
        
        # Check for obstacles in the path
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_obstacle_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Adjust coordinates for full image
                y += roi_start
                
                # Check if obstacle is in the center path
                center_x = x + w // 2
                if width * 0.3 < center_x < width * 0.7:  # Center 40% of image
                    # Check if obstacle is close (bottom part of ROI)
                    if y > height * 0.7:
                        obstacle_detected = True
                        
                        # Draw warning rectangle
                        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.putText(debug_image, "OBSTACLE!", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # Draw non-threatening obstacles in yellow
                    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # Draw center path indicators
        cv2.line(debug_image, (int(width * 0.3), roi_start), 
                (int(width * 0.3), height), (0, 255, 0), 2)
        cv2.line(debug_image, (int(width * 0.7), roi_start), 
                (int(width * 0.7), height), (0, 255, 0), 2)
        
        # Publish obstacle detection result
        self.obstacle_pub.publish(Bool(obstacle_detected))
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            rospy.logwarn(f"Could not publish debug image: {str(e)}")
    
    def detect_by_depth(self, image):
        """Depth-based obstacle detection (placeholder for stereo vision)"""
        # TODO: Implement stereo vision depth estimation
        rospy.logwarn("Depth-based detection not yet implemented")
        self.obstacle_pub.publish(Bool(False))

if __name__ == '__main__':
    try:
        detector = ObstacleDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass