#!/usr/bin/env python3

"""
Enhanced Object Detector Node for DuckieBot - Phase 2
Supports multiple detection methods: color, contour, and template matching
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float32, String
from cv_bridge import CvBridge
# Removed tf2 imports - not needed for basic object detection

class EnhancedObjectDetector:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('enhanced_object_detector', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.target_pub = rospy.Publisher('/object_follower/target_position', Point, queue_size=1)
        self.target_found_pub = rospy.Publisher('/object_follower/target_found', Bool, queue_size=1)
        self.distance_pub = rospy.Publisher('/object_follower/target_distance', Float32, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/object_follower/debug_image', Image, queue_size=1)
        self.detection_info_pub = rospy.Publisher('/object_follower/detection_info', String, queue_size=1)
        
        # Subscribers - Handle both local and DuckieBot camera topics
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        # DuckieBot-specific topic (with robot namespace)
        robot_name = rospy.get_param('~robot_name', 'blueduckie')  # Get robot name
        compressed_topic = f"/{robot_name}/camera_node/image/compressed"
        self.compressed_image_sub = rospy.Subscriber(compressed_topic, CompressedImage, self.compressed_image_callback)
        
        # Fallback for generic topic
        self.compressed_fallback_sub = rospy.Subscriber('/camera_node/image/compressed', CompressedImage, self.compressed_image_callback)
        
        # Parameters
        self.detection_method = rospy.get_param('~detection_method', 'color')  # 'color', 'contour', 'template'
        self.target_color = rospy.get_param('~target_color', 'red')  # 'red', 'blue', 'green', 'yellow'
        
        # Color detection parameters for different colors
        self.color_ranges = {
            'red': [np.array([0, 50, 50]), np.array([10, 255, 255])],
            'red2': [np.array([170, 50, 50]), np.array([180, 255, 255])],  # Second red range
            'blue': [np.array([100, 50, 50]), np.array([130, 255, 255])],
            'green': [np.array([40, 50, 50]), np.array([80, 255, 255])],
            'yellow': [np.array([20, 50, 50]), np.array([30, 255, 255])]
        }
        
        # Enhanced tracking parameters
        self.last_known_position = None
        self.tracking_confidence = 0.0
        self.consecutive_detections = 0
        self.max_tracking_loss = 10  # frames
        self.tracking_loss_count = 0
        
        # Kalman filter for object tracking
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kalman_initialized = False
        
        # Camera parameters (improved calibration)
        self.focal_length = 525.0  # pixels
        self.known_object_width = 0.1  # meters (10cm)
        
        # Performance metrics
        self.detection_count = 0
        self.false_positive_count = 0
        self.start_time = rospy.Time.now()
        
        rospy.loginfo(f"Enhanced Object Detector initialized - Method: {self.detection_method}, Color: {self.target_color}")
    
    def compressed_image_callback(self, msg):
        """Handle compressed images from DuckieBot camera"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                # Debug: Log image reception
                rospy.loginfo_throttle(5, f"Enhanced detector processing compressed image: {cv_image.shape}")
                
                if self.detection_method == 'color':
                    self.detect_by_color(cv_image)
                elif self.detection_method == 'contour':
                    self.detect_by_contour(cv_image)
                elif self.detection_method == 'template':
                    self.detect_by_template(cv_image)
            else:
                rospy.logwarn("Failed to decode compressed image")
                
        except Exception as e:
            rospy.logerr(f"Error processing compressed image: {str(e)}")
    
    def image_callback(self, msg):
        """Handle regular images (for local testing)"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Debug: Log image reception
            rospy.loginfo_throttle(5, f"Enhanced detector processing image: {cv_image.shape}")
            
            if self.detection_method == 'color':
                self.detect_by_color(cv_image)
            elif self.detection_method == 'contour':
                self.detect_by_contour(cv_image)
            elif self.detection_method == 'template':
                self.detect_by_template(cv_image)
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")
    
    def detect_by_color(self, image):
        """Enhanced color-based object detection with multiple color support"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for target color
        if self.target_color == 'red':
            # Red wraps around in HSV, so we need two ranges
            mask1 = cv2.inRange(hsv, self.color_ranges['red'][0], self.color_ranges['red'][1])
            mask2 = cv2.inRange(hsv, self.color_ranges['red2'][0], self.color_ranges['red2'][1])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            color_range = self.color_ranges.get(self.target_color, self.color_ranges['red'])
            mask = cv2.inRange(hsv, color_range[0], color_range[1])
        
        # Enhanced morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug: Log detection attempts
        rospy.loginfo_throttle(3, f"Enhanced detector found {len(contours)} contours for {self.target_color} detection")
        
        self.process_detections(image, contours, mask)
    
    def detect_by_contour(self, image):
        """Contour-based detection using shape analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by shape (looking for circular objects)
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if 0.3 < circularity < 1.2:  # Somewhat circular
                        filtered_contours.append(contour)
        
        self.process_detections(image, filtered_contours, thresh)
    
    def detect_by_template(self, image):
        """Template matching detection (placeholder for future implementation)"""
        # TODO: Implement template matching
        rospy.logwarn_throttle(5, "Template matching not yet implemented")
        self.target_found_pub.publish(Bool(False))
    
    def process_detections(self, image, contours, mask):
        """Process detected contours and update tracking"""
        debug_image = image.copy()
        target_found = False
        best_detection = None
        best_confidence = 0.0
        
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    # Calculate bounding box and center
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Calculate confidence based on area and shape
                    confidence = min(area / 5000.0, 1.0)  # Normalize by expected area
                    
                    # Add aspect ratio check
                    aspect_ratio = float(w) / h
                    if 0.5 < aspect_ratio < 2.0:  # Reasonable aspect ratio
                        confidence *= 1.2
                    
                    # Prefer detections closer to previous position if tracking
                    if self.last_known_position is not None:
                        prev_x = (self.last_known_position.x + 1) * image.shape[1] / 2
                        prev_y = (self.last_known_position.y + 1) * image.shape[0] / 2
                        distance = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                        confidence *= max(0.5, 1.0 - distance / 200.0)  # Closer = higher confidence
                    
                    if confidence > best_confidence and confidence > 0.1:  # Lower threshold
                        best_confidence = confidence
                        best_detection = (center_x, center_y, w, h, area, contour)
                        target_found = True
        
        # Process best detection
        if target_found and best_detection:
            center_x, center_y, w, h, area, contour = best_detection
            
            # Kalman filter prediction and update
            if not self.kalman_initialized:
                self.kalman.statePre = np.array([center_x, center_y, 0, 0], dtype=np.float32)
                self.kalman.statePost = np.array([center_x, center_y, 0, 0], dtype=np.float32)
                self.kalman_initialized = True
            
            # Predict and update Kalman filter
            prediction = self.kalman.predict()
            measurement = np.array([[center_x], [center_y]], dtype=np.float32)
            self.kalman.correct(measurement)
            
            # Use Kalman filtered position
            filtered_x, filtered_y = prediction[0], prediction[1]
            
            # Estimate distance
            distance = self.estimate_distance(w)
            
            # Publish target position (normalized coordinates)
            target_point = Point()
            target_point.x = (filtered_x - image.shape[1] // 2) / (image.shape[1] // 2)  # -1 to 1
            target_point.y = (filtered_y - image.shape[0] // 2) / (image.shape[0] // 2)  # -1 to 1
            target_point.z = distance
            
            self.target_pub.publish(target_point)
            self.distance_pub.publish(Float32(distance))
            
            # Draw enhanced debug information
            cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 2)
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(debug_image, (int(filtered_x), int(filtered_y)), 5, (0, 0, 255), -1)
            cv2.putText(debug_image, f"Dist: {distance:.2f}m", (x, y-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_image, f"Conf: {best_confidence:.2f}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Update tracking
            self.last_known_position = target_point
            self.tracking_confidence = best_confidence
            self.consecutive_detections += 1
            self.tracking_loss_count = 0
            self.detection_count += 1
            
            # Publish detection info
            info_msg = f"Method: {self.detection_method}, Color: {self.target_color}, Confidence: {best_confidence:.2f}"
            self.detection_info_pub.publish(String(info_msg))
            
        else:
            # Handle tracking loss
            self.tracking_loss_count += 1
            if self.tracking_loss_count > self.max_tracking_loss:
                self.consecutive_detections = 0
                self.tracking_confidence = 0.0
                target_found = False
            else:
                # Use prediction if recently lost
                if self.kalman_initialized:
                    prediction = self.kalman.predict()
                    cv2.circle(debug_image, (int(prediction[0]), int(prediction[1])), 
                              10, (255, 255, 0), 2)
                    cv2.putText(debug_image, "PREDICTED", (int(prediction[0]), int(prediction[1])-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Add performance info to debug image
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        detection_rate = self.detection_count / max(elapsed, 1.0)
        cv2.putText(debug_image, f"Rate: {detection_rate:.1f} Hz", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_image, f"Tracking: {self.consecutive_detections}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Publish target found status
        self.target_found_pub.publish(Bool(target_found))
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            rospy.logwarn(f"Could not publish debug image: {str(e)}")
    
    def estimate_distance(self, pixel_width):
        """Enhanced distance estimation with bounds checking"""
        if pixel_width > 0:
            distance = (self.known_object_width * self.focal_length) / pixel_width
            return max(0.1, min(distance, 5.0))  # Clamp between 10cm and 5m
        return 0.0
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = EnhancedObjectDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass