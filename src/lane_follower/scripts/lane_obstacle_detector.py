#!/usr/bin/env python3

"""
Lane Obstacle Detector for Advanced Lane Following System
Detects obstacles in the lane and publishes safety warnings
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import threading

class LaneObstacleDetector:
    def __init__(self):
        rospy.init_node('lane_obstacle_detector', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.obstacle_detected_pub = rospy.Publisher('/lane_follower/obstacle_detected', Bool, queue_size=1)
        self.obstacle_position_pub = rospy.Publisher('/lane_follower/obstacle_position', Point, queue_size=1)
        self.obstacle_info_pub = rospy.Publisher('/lane_follower/obstacle_info', String, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/lane_follower/obstacle_debug', Image, queue_size=1)
        
        # Subscribers
        robot_name = rospy.get_param('~robot_name', 'blueduckie')
        compressed_topic = f"/{robot_name}/camera_node/image/compressed"
        
        self.compressed_image_sub = rospy.Subscriber(compressed_topic, CompressedImage, 
                                           self.compressed_image_callback, 
                                           queue_size=1, buff_size=2**24)
        
        # Fallback for generic topic
        self.compressed_fallback_sub = rospy.Subscriber('/camera_node/image/compressed', 
                                                       CompressedImage, self.compressed_image_callback, 
                                                       queue_size=1, buff_size=2**24)
        
        # Detection parameters
        self.detection_method = rospy.get_param('~detection_method', 'edge')
        self.min_obstacle_area = rospy.get_param('~min_obstacle_area', 1000)
        self.safe_distance_pixels = rospy.get_param('~safe_distance_pixels', 60)
        self.obstacle_height_threshold = rospy.get_param('~obstacle_height_threshold', 50)
        
        # Image processing parameters
        self.image_width = 640
        self.image_height = 480
        self.roi_top_ratio = 0.4  # Focus on closer area for obstacles
        self.roi_bottom_ratio = 1.0
        
        # Edge detection parameters
        self.canny_low = 100
        self.canny_high = 200
        self.blur_kernel = 5
        
        # Obstacle tracking
        self.last_obstacle_position = None
        self.obstacle_confidence = 0.0
        self.consecutive_detections = 0
        self.detection_threshold = 3  # Require 3 consecutive detections
        
        # Performance metrics
        self.detection_count = 0
        self.false_positive_count = 0
        self.last_process_time = 0
        self.min_process_interval = 0.1  # Process at 10 FPS
        
        # Threading control
        self._processing = False
        
        rospy.loginfo("Lane Obstacle Detector initialized - Monitoring for obstacles")
    
    def compressed_image_callback(self, msg):
        """Handle compressed images from DuckieBot camera"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                self.process_image(cv_image)
            else:
                rospy.logwarn("Failed to decode compressed image for obstacle detection")
                
        except Exception as e:
            rospy.logerr(f"Error processing compressed image for obstacles: {str(e)}")
    
    def process_image(self, image):
        """Process image with rate limiting and threading"""
        current_time = rospy.Time.now().to_sec()
        if current_time - self.last_process_time < self.min_process_interval:
            return

        if hasattr(self, '_processing') and self._processing:
            return

        self._processing = True
        self.last_process_time = current_time

        # Process asynchronously
        thread = threading.Thread(target=self._async_process_image, args=(image.copy(),))
        thread.daemon = True
        thread.start()
    
    def _async_process_image(self, image):
        """Asynchronous obstacle detection processing"""
        try:
            # Resize image for consistent processing
            if image.shape[1] != self.image_width or image.shape[0] != self.image_height:
                image = cv2.resize(image, (self.image_width, self.image_height))
            
            # Apply region of interest (focus on lane area)
            roi_image = self.apply_roi(image)
            
            # Detect obstacles using selected method
            obstacles = []
            if self.detection_method == 'edge':
                obstacles = self.detect_obstacles_by_edges(roi_image)
            elif self.detection_method == 'contour':
                obstacles = self.detect_obstacles_by_contours(roi_image)
            elif self.detection_method == 'combined':
                obstacles_edge = self.detect_obstacles_by_edges(roi_image)
                obstacles_contour = self.detect_obstacles_by_contours(roi_image)
                obstacles = self.combine_detections(obstacles_edge, obstacles_contour)
            
            # Filter and validate obstacles
            valid_obstacles = self.filter_obstacles(obstacles, image.shape)
            
            # Determine if obstacle is in danger zone
            obstacle_detected, closest_obstacle = self.evaluate_obstacle_threat(valid_obstacles)
            
            # Create debug visualization
            debug_image = self.create_debug_visualization(image, valid_obstacles, obstacle_detected)
            
            # Publish results
            self.publish_obstacle_results(obstacle_detected, closest_obstacle, valid_obstacles, debug_image)
            
            self.detection_count += 1
            
        except Exception as e:
            rospy.logerr(f"Error in obstacle detection: {str(e)}")
        finally:
            self._processing = False
    
    def apply_roi(self, image):
        """Apply region of interest focusing on lane area ahead"""
        height, width = image.shape[:2]
        
        # Define ROI for obstacle detection (narrower than lane detection)
        roi_top = int(height * self.roi_top_ratio)
        roi_bottom = int(height * self.roi_bottom_ratio)
        
        # Focus on center lane area
        left_margin = width // 4
        right_margin = 3 * width // 4
        
        vertices = np.array([
            [left_margin, roi_bottom],
            [left_margin + 50, roi_top],
            [right_margin - 50, roi_top],
            [right_margin, roi_bottom]
        ], dtype=np.int32)
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        
        # Apply mask
        roi_image = cv2.bitwise_and(image, image, mask=mask)
        
        return roi_image
    
    def detect_obstacles_by_edges(self, image):
        """Detect obstacles using edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Find contours from edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_obstacle_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by height (obstacles should have some vertical extent)
                if h > self.obstacle_height_threshold:
                    obstacles.append({
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'center': (x + w//2, y + h//2),
                        'method': 'edge'
                    })
        
        return obstacles
    
    def detect_obstacles_by_contours(self, image):
        """Detect obstacles using color-based contour detection"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for non-road colors (obstacles are typically not yellow/white)
        # Exclude yellow lanes
        yellow_lower = np.array([15, 50, 50])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Exclude white lanes
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([255, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Combine lane masks
        lane_mask = cv2.bitwise_or(yellow_mask, white_mask)
        
        # Invert to get non-lane areas (potential obstacles)
        obstacle_mask = cv2.bitwise_not(lane_mask)
        
        # Apply morphological operations to clean up
        kernel = np.ones((5,5), np.uint8)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_obstacle_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio and height
                aspect_ratio = w / max(h, 1)
                if h > self.obstacle_height_threshold and aspect_ratio < 3:  # Not too wide
                    obstacles.append({
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'center': (x + w//2, y + h//2),
                        'method': 'contour'
                    })
        
        return obstacles
    
    def combine_detections(self, obstacles_edge, obstacles_contour):
        """Combine obstacle detections from different methods"""
        all_obstacles = obstacles_edge + obstacles_contour
        
        # Remove duplicates by checking overlap
        unique_obstacles = []
        for obstacle in all_obstacles:
            is_duplicate = False
            x1, y1, w1, h1 = obstacle['bbox']
            
            for existing in unique_obstacles:
                x2, y2, w2, h2 = existing['bbox']
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                # If significant overlap, consider it a duplicate
                if overlap_area > 0.5 * min(w1 * h1, w2 * h2):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_obstacles.append(obstacle)
        
        return unique_obstacles
    
    def filter_obstacles(self, obstacles, image_shape):
        """Filter obstacles based on position and characteristics"""
        height, width = image_shape[:2]
        valid_obstacles = []
        
        for obstacle in obstacles:
            x, y, w, h = obstacle['bbox']
            center_x, center_y = obstacle['center']
            
            # Filter by position (must be in lower part of image - closer to robot)
            if center_y > height * 0.6:  # In lower 40% of image
                # Filter by size (not too small or too large)
                if self.min_obstacle_area < obstacle['area'] < width * height * 0.3:
                    # Filter by aspect ratio (reasonable obstacle shape)
                    aspect_ratio = w / max(h, 1)
                    if 0.3 < aspect_ratio < 4.0:
                        valid_obstacles.append(obstacle)
        
        return valid_obstacles
    
    def evaluate_obstacle_threat(self, obstacles):
        """Evaluate if obstacles pose a threat to the robot"""
        if not obstacles:
            return False, None
        
        # Find closest obstacle (lowest y-coordinate = closest to robot)
        closest_obstacle = min(obstacles, key=lambda obs: obs['center'][1])
        
        # Check if obstacle is in danger zone
        center_x, center_y = closest_obstacle['center']
        
        # Danger zone: close to robot (high y-coordinate) and in center lane
        danger_y_threshold = self.image_height - self.safe_distance_pixels
        danger_x_min = self.image_width // 3
        danger_x_max = 2 * self.image_width // 3
        
        is_threat = (center_y > danger_y_threshold and 
                    danger_x_min < center_x < danger_x_max)
        
        # Update obstacle confidence
        if is_threat:
            self.consecutive_detections += 1
            self.obstacle_confidence = min(1.0, self.consecutive_detections / self.detection_threshold)
        else:
            self.consecutive_detections = max(0, self.consecutive_detections - 1)
            self.obstacle_confidence = max(0.0, self.consecutive_detections / self.detection_threshold)
        
        # Only report threat if confidence is high enough
        confirmed_threat = is_threat and self.obstacle_confidence >= 1.0
        
        return confirmed_threat, closest_obstacle if confirmed_threat else None
    
    def create_debug_visualization(self, image, obstacles, obstacle_detected):
        """Create debug visualization showing obstacle detection"""
        debug_image = image.copy()
        
        # Draw all detected obstacles
        for obstacle in obstacles:
            x, y, w, h = obstacle['bbox']
            center_x, center_y = obstacle['center']
            
            # Color based on threat level
            color = (0, 0, 255) if obstacle_detected else (0, 255, 255)  # Red if threat, yellow otherwise
            
            # Draw bounding box
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            cv2.circle(debug_image, (center_x, center_y), 5, color, -1)
            
            # Add obstacle info
            info_text = f"{obstacle['method']}: {obstacle['area']:.0f}"
            cv2.putText(debug_image, info_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw danger zone
        danger_y = self.image_height - self.safe_distance_pixels
        danger_x_min = self.image_width // 3
        danger_x_max = 2 * self.image_width // 3
        
        cv2.rectangle(debug_image, (danger_x_min, danger_y), 
                     (danger_x_max, self.image_height), (255, 0, 0), 2)
        cv2.putText(debug_image, "DANGER ZONE", (danger_x_min, danger_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Add status information
        status_text = [
            f"Obstacles: {len(obstacles)}",
            f"Threat: {obstacle_detected}",
            f"Confidence: {self.obstacle_confidence:.2f}"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(debug_image, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_image
    
    def publish_obstacle_results(self, obstacle_detected, closest_obstacle, all_obstacles, debug_image):
        """Publish obstacle detection results"""
        # Publish obstacle detection status
        self.obstacle_detected_pub.publish(Bool(obstacle_detected))
        
        # Publish closest obstacle position if detected
        if closest_obstacle:
            center_x, center_y = closest_obstacle['center']
            
            # Normalize position to [-1, 1] range
            normalized_x = (center_x - self.image_width//2) / (self.image_width//2)
            normalized_y = (center_y - self.image_height//2) / (self.image_height//2)
            
            position = Point()
            position.x = normalized_x
            position.y = normalized_y
            position.z = closest_obstacle['area']  # Use area as size indicator
            
            self.obstacle_position_pub.publish(position)
        
        # Publish obstacle information
        info_msg = f"Obstacles: {len(all_obstacles)}, Threat: {obstacle_detected}, Confidence: {self.obstacle_confidence:.2f}"
        self.obstacle_info_pub.publish(String(info_msg))
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            rospy.logwarn(f"Could not publish obstacle debug image: {str(e)}")

if __name__ == '__main__':
    try:
        detector = LaneObstacleDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass