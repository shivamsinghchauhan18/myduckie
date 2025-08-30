#!/usr/bin/env python3

"""
Advanced Lane Detector Node for DuckieTown - Professional Grade
Uses advanced computer vision techniques for robust lane detection
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float32, String, Header
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import threading
import time

class AdvancedLaneDetector:
    def __init__(self):
        rospy.init_node('advanced_lane_detector', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.lane_pose_pub = rospy.Publisher('/lane_follower/lane_pose', Point, queue_size=1)
        self.lane_found_pub = rospy.Publisher('/lane_follower/lane_found', Bool, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/lane_follower/debug_image', Image, queue_size=1)
        self.lane_center_pub = rospy.Publisher('/lane_follower/lane_center', Point, queue_size=1)
        self.lane_angle_pub = rospy.Publisher('/lane_follower/lane_angle', Float32, queue_size=1)
        self.detection_info_pub = rospy.Publisher('/lane_follower/detection_info', String, queue_size=1)
        
        # Subscribers - Handle both local and DuckieBot camera topics
        robot_name = rospy.get_param('~robot_name', 'blueduckie')
        compressed_topic = f"/{robot_name}/camera_node/image/compressed"
        
        self.compressed_image_sub = rospy.Subscriber(compressed_topic, CompressedImage, 
                                           self.compressed_image_callback, 
                                           queue_size=1, buff_size=2**24)
        
        # Fallback for generic topic
        self.compressed_fallback_sub = rospy.Subscriber('/camera_node/image/compressed', 
                                                       CompressedImage, self.compressed_image_callback, 
                                                       queue_size=1, buff_size=2**24)
        
        # Lane detection parameters
        self.image_width = 640
        self.image_height = 480
        self.roi_top_ratio = 0.6  # Region of interest starts at 60% from top
        self.roi_bottom_ratio = 1.0  # Full bottom
        
        # Color thresholds for yellow and white lanes
        self.yellow_lower = np.array([15, 80, 80], dtype=np.uint8)
        self.yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
        self.white_lower = np.array([0, 0, 200], dtype=np.uint8)
        self.white_upper = np.array([255, 30, 255], dtype=np.uint8)
        
        # Canny edge detection parameters
        self.canny_low = 50
        self.canny_high = 150
        
        # Hough line parameters
        self.hough_rho = 1
        self.hough_theta = np.pi/180
        self.hough_threshold = 30
        self.hough_min_line_len = 20
        self.hough_max_line_gap = 300
        
        # Lane tracking state
        self.last_left_lane = None
        self.last_right_lane = None
        self.lane_confidence = 0.0
        self.consecutive_detections = 0
        self.max_tracking_loss = 5
        self.tracking_loss_count = 0
        
        # Performance metrics
        self.detection_count = 0
        self.start_time = rospy.Time.now()
        self.last_process_time = 0
        self.min_process_interval = 0.05  # Process at most 20 FPS
        
        # Kalman filter for lane tracking
        self.kalman_left = cv2.KalmanFilter(4, 2)
        self.kalman_right = cv2.KalmanFilter(4, 2)
        self.setup_kalman_filters()
        
        # Threading control
        self._processing = False
        
        rospy.loginfo("Advanced Lane Detector initialized - Ready for lane detection")
    
    def setup_kalman_filters(self):
        """Initialize Kalman filters for lane tracking"""
        for kalman in [self.kalman_left, self.kalman_right]:
            kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
            kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
            kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
    
    def compressed_image_callback(self, msg):
        """Handle compressed images from DuckieBot camera"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                rospy.loginfo_throttle(5, f"Lane detector processing image: {cv_image.shape}")
                self.process_image(cv_image)
            else:
                rospy.logwarn("Failed to decode compressed image")
                
        except Exception as e:
            rospy.logerr(f"Error processing compressed image: {str(e)}")
    
    def process_image(self, image):
        """Process image with rate limiting and threading"""
        current_time = rospy.Time.now().to_sec()
        if current_time - self.last_process_time < self.min_process_interval:
            return

        if hasattr(self, '_processing') and self._processing:
            rospy.loginfo_throttle(1, "Skipping frame - still processing")
            return

        self._processing = True
        self.last_process_time = current_time

        # Process asynchronously in separate thread
        thread = threading.Thread(target=self._async_process_image, args=(image.copy(),))
        thread.daemon = True
        thread.start()
    
    def _async_process_image(self, image):
        """Asynchronous image processing in separate thread"""
        try:
            # Resize image for consistent processing
            if image.shape[1] != self.image_width or image.shape[0] != self.image_height:
                image = cv2.resize(image, (self.image_width, self.image_height))
            
            # Apply region of interest
            roi_image = self.apply_roi(image)
            
            # Detect lanes using multiple methods
            lanes_color = self.detect_lanes_by_color(roi_image)
            lanes_edge = self.detect_lanes_by_edges(roi_image)
            
            # Combine and filter lane detections
            combined_lanes = self.combine_lane_detections(lanes_color, lanes_edge)
            
            # Track lanes with Kalman filtering
            left_lane, right_lane = self.track_lanes_with_kalman(combined_lanes)
            
            # Calculate lane pose
            lane_pose = self.calculate_lane_pose(left_lane, right_lane, image.shape)
            
            # Create debug visualization
            debug_image = self.create_debug_visualization(image, left_lane, right_lane, lane_pose)
            
            # Publish results
            self.publish_lane_results(lane_pose, left_lane, right_lane, debug_image)
            
            self.detection_count += 1
            
        except Exception as e:
            rospy.logerr(f"Error in lane detection: {str(e)}")
        finally:
            self._processing = False
    
    def apply_roi(self, image):
        """Apply region of interest to focus on road area"""
        height, width = image.shape[:2]
        
        # Define ROI vertices (trapezoid shape)
        roi_top = int(height * self.roi_top_ratio)
        roi_bottom = int(height * self.roi_bottom_ratio)
        
        vertices = np.array([
            [0, roi_bottom],
            [width//4, roi_top],
            [3*width//4, roi_top],
            [width, roi_bottom]
        ], dtype=np.int32)
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        
        # Apply mask
        roi_image = cv2.bitwise_and(image, image, mask=mask)
        
        return roi_image
    
    def detect_lanes_by_color(self, image):
        """Detect lanes using color thresholding"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for yellow and white lanes
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3,3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and convert contours to lines
        lanes = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                # Fit line to contour
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                
                # Convert to line endpoints
                height = image.shape[0]
                y1 = height
                y2 = int(height * 0.6)
                x1 = int(x + (y1 - y) * vx / vy)
                x2 = int(x + (y2 - y) * vx / vy)
                
                if 0 <= x1 < image.shape[1] and 0 <= x2 < image.shape[1]:
                    lanes.append(((x1, y1), (x2, y2)))
        
        return lanes
    
    def detect_lanes_by_edges(self, image):
        """Detect lanes using edge detection and Hough transform"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Apply Hough line transform
        lines = cv2.HoughLinesP(edges, self.hough_rho, self.hough_theta, 
                               self.hough_threshold, minLineLength=self.hough_min_line_len,
                               maxLineGap=self.hough_max_line_gap)
        
        lanes = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Filter lines by slope (avoid horizontal lines)
                if abs(x2 - x1) > 10:  # Avoid division by zero
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) > 0.3:  # Filter out nearly horizontal lines
                        lanes.append(((x1, y1), (x2, y2)))
        
        return lanes
    
    def combine_lane_detections(self, lanes_color, lanes_edge):
        """Combine and filter lane detections from different methods"""
        all_lanes = lanes_color + lanes_edge
        
        if not all_lanes:
            return [], []
        
        # Separate left and right lanes based on position and slope
        left_lanes = []
        right_lanes = []
        
        image_center = self.image_width // 2
        
        for lane in all_lanes:
            (x1, y1), (x2, y2) = lane
            
            # Calculate slope and position
            if abs(x2 - x1) > 10:
                slope = (y2 - y1) / (x2 - x1)
                center_x = (x1 + x2) // 2
                
                # Classify as left or right lane
                if center_x < image_center and slope < 0:  # Left lane (negative slope)
                    left_lanes.append(lane)
                elif center_x > image_center and slope > 0:  # Right lane (positive slope)
                    right_lanes.append(lane)
        
        # Average multiple detections for each side
        left_lane = self.average_lanes(left_lanes) if left_lanes else None
        right_lane = self.average_lanes(right_lanes) if right_lanes else None
        
        return left_lane, right_lane
    
    def average_lanes(self, lanes):
        """Average multiple lane detections"""
        if not lanes:
            return None
        
        x1_sum = y1_sum = x2_sum = y2_sum = 0
        
        for (x1, y1), (x2, y2) in lanes:
            x1_sum += x1
            y1_sum += y1
            x2_sum += x2
            y2_sum += y2
        
        count = len(lanes)
        avg_lane = ((x1_sum // count, y1_sum // count), 
                   (x2_sum // count, y2_sum // count))
        
        return avg_lane
    
    def track_lanes_with_kalman(self, lane_detection):
        """Track lanes using Kalman filtering for stability"""
        left_lane, right_lane = lane_detection
        
        # Update Kalman filters with detections
        if left_lane:
            (x1, y1), (x2, y2) = left_lane
            measurement = np.array([[np.float32(x1)], [np.float32(y1)]])
            self.kalman_left.correct(measurement)
            self.last_left_lane = left_lane
        
        if right_lane:
            (x1, y1), (x2, y2) = right_lane
            measurement = np.array([[np.float32(x1)], [np.float32(y1)]])
            self.kalman_right.correct(measurement)
            self.last_right_lane = right_lane
        
        # Get predictions
        left_prediction = self.kalman_left.predict()
        right_prediction = self.kalman_right.predict()
        
        # Use predictions if no detection
        if not left_lane and self.last_left_lane:
            x_pred, y_pred = int(left_prediction[0]), int(left_prediction[1])
            left_lane = ((x_pred, self.image_height), (x_pred - 50, int(self.image_height * 0.6)))
        
        if not right_lane and self.last_right_lane:
            x_pred, y_pred = int(right_prediction[0]), int(right_prediction[1])
            right_lane = ((x_pred, self.image_height), (x_pred + 50, int(self.image_height * 0.6)))
        
        return left_lane, right_lane
    
    def calculate_lane_pose(self, left_lane, right_lane, image_shape):
        """Calculate lane pose (lateral offset and heading angle)"""
        height, width = image_shape[:2]
        image_center = width // 2
        
        # Default values
        lateral_offset = 0.0
        heading_angle = 0.0
        lane_found = False
        
        if left_lane and right_lane:
            # Both lanes detected - ideal case
            (x1_l, y1_l), (x2_l, y2_l) = left_lane
            (x1_r, y1_r), (x2_r, y2_r) = right_lane
            
            # Calculate lane center at bottom of image
            lane_center_x = (x1_l + x1_r) // 2
            
            # Calculate lateral offset (normalized to [-1, 1])
            lateral_offset = (lane_center_x - image_center) / (image_center)
            
            # Calculate heading angle from lane direction
            left_slope = (y2_l - y1_l) / max(abs(x2_l - x1_l), 1)
            right_slope = (y2_r - y1_r) / max(abs(x2_r - x1_r), 1)
            avg_slope = (left_slope + right_slope) / 2
            heading_angle = np.arctan(avg_slope)
            
            lane_found = True
            self.lane_confidence = 1.0
            
        elif left_lane or right_lane:
            # Single lane detected
            lane = left_lane if left_lane else right_lane
            (x1, y1), (x2, y2) = lane
            
            # Estimate lane center based on single lane
            if left_lane:
                # Assume standard lane width and estimate right edge
                estimated_center = x1 + 100  # Approximate lane width in pixels
            else:
                # Assume standard lane width and estimate left edge
                estimated_center = x1 - 100
            
            lateral_offset = (estimated_center - image_center) / image_center
            
            # Calculate heading angle
            slope = (y2 - y1) / max(abs(x2 - x1), 1)
            heading_angle = np.arctan(slope)
            
            lane_found = True
            self.lane_confidence = 0.7
        
        # Create Point message for lane pose
        lane_pose = Point()
        lane_pose.x = lateral_offset  # Lateral offset
        lane_pose.y = heading_angle   # Heading angle
        lane_pose.z = 1.0 if lane_found else 0.0  # Lane found status
        
        return lane_pose
    
    def create_debug_visualization(self, image, left_lane, right_lane, lane_pose):
        """Create debug visualization with lane overlays"""
        debug_image = image.copy()
        
        # Draw detected lanes
        if left_lane:
            (x1, y1), (x2, y2) = left_lane
            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(debug_image, "LEFT", (x2-30, y2-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if right_lane:
            (x1, y1), (x2, y2) = right_lane
            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(debug_image, "RIGHT", (x2+10, y2-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw lane center and robot position
        image_center = debug_image.shape[1] // 2
        cv2.line(debug_image, (image_center, 0), (image_center, debug_image.shape[0]), 
                (255, 255, 255), 2)
        
        # Draw lane pose information
        info_text = [
            f"Lateral Offset: {lane_pose.d:.3f}",
            f"Heading Angle: {lane_pose.phi:.3f}",
            f"In Lane: {lane_pose.in_lane}",
            f"Confidence: {self.lane_confidence:.2f}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(debug_image, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_image
    
    def publish_lane_results(self, lane_pose, left_lane, right_lane, debug_image):
        """Publish all lane detection results"""
        # Publish lane pose
        self.lane_pose_pub.publish(lane_pose)
        
        # Publish lane found status
        self.lane_found_pub.publish(Bool(lane_pose.z > 0.5))
        
        # Publish lane center point
        if left_lane and right_lane:
            (x1_l, y1_l), _ = left_lane
            (x1_r, y1_r), _ = right_lane
            center_x = (x1_l + x1_r) // 2
            center_point = Point()
            center_point.x = (center_x - self.image_width//2) / (self.image_width//2)
            center_point.y = 0.0
            center_point.z = 0.0
            self.lane_center_pub.publish(center_point)
        
        # Publish lane angle
        self.lane_angle_pub.publish(Float32(lane_pose.y))
        
        # Publish detection info
        info_msg = f"Lanes: L={left_lane is not None}, R={right_lane is not None}, Conf: {self.lane_confidence:.2f}"
        self.detection_info_pub.publish(String(info_msg))
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            rospy.logwarn(f"Could not publish debug image: {str(e)}")
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = AdvancedLaneDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass