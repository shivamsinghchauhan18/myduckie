#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32, Bool
from cv_bridge import CvBridge
import requests
import json
import time
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self):
        rospy.init_node('yolo_detector', anonymous=True)
        
        # Initialize YOLO model
        model_path = rospy.get_param('~model_path', 'yolov8n.pt')
        self.model = YOLO(model_path)
        rospy.loginfo(f"Loaded YOLO model: {model_path}")
        self.bridge = CvBridge()
        
        # Server configuration
        self.server_url = rospy.get_param('~server_url', 'http://localhost:5000')
        self.bot_id = rospy.get_param('~bot_id', 'duckiebot_01')
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.safe_distance = 0.8  # meters
        self.emergency_distance = 0.3  # meters
        
        # Camera parameters (adjust based on your camera)
        self.focal_length = 525.0  # pixels
        self.known_widths = {
            # Standard COCO objects
            'person': 0.6,      # meters
            'car': 1.8,         # meters
            'truck': 2.5,       # meters
            'bicycle': 0.6,     # meters
            'motorcycle': 0.8,  # meters
            'bus': 2.5,         # meters
            'stop sign': 0.3,   # meters
            'traffic light': 0.3, # meters
            
            # Duckietown-specific objects
            'duckiebot': 0.18,  # meters (actual Duckiebot width)
            'duckie': 0.08,     # meters (rubber duck)
            'cone': 0.06,       # meters (traffic cone)
            'building': 0.3,    # meters (typical building width in Duckietown)
            'tree': 0.1,        # meters (decorative trees)
            'road_sign': 0.1,   # meters (Duckietown road signs)
            'barrier': 0.2,     # meters (road barriers)
            'intersection_sign': 0.1  # meters
        }
        
        # Publishers
        self.detection_pub = rospy.Publisher('/yolo/detections', String, queue_size=10)
        self.distance_pub = rospy.Publisher('/yolo/distances', String, queue_size=10)
        self.safety_pub = rospy.Publisher('/yolo/safety_status', String, queue_size=10)
        self.emergency_pub = rospy.Publisher('/yolo/emergency', Bool, queue_size=10)
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback)
        
        # State variables
        self.current_detections = []
        self.last_server_update = time.time()
        
        rospy.loginfo("YOLOv8 Detector initialized")
    
    def calculate_distance(self, bbox_width, object_class):
        """Calculate distance using object width and known real-world dimensions"""
        if object_class not in self.known_widths:
            return None
        
        real_width = self.known_widths[object_class]
        distance = (real_width * self.focal_length) / bbox_width
        return distance
    
    def image_callback(self, msg):
        try:
            # Convert compressed image to cv2
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Run YOLO detection
            results = self.model(cv_image, conf=self.confidence_threshold)
            
            detections = []
            distances = []
            emergency_detected = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Calculate distance
                        bbox_width = x2 - x1
                        distance = self.calculate_distance(bbox_width, class_name)
                        
                        detection_data = {
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'distance': distance,
                            'center_x': float((x1 + x2) / 2),
                            'center_y': float((y1 + y2) / 2)
                        }
                        
                        detections.append(detection_data)
                        
                        if distance is not None:
                            distances.append({
                                'object': class_name,
                                'distance': distance,
                                'position': detection_data['center_x']
                            })
                            
                            # Check for emergency situation
                            if distance < self.emergency_distance:
                                emergency_detected = True
            
            self.current_detections = detections
            
            # Publish detection results
            self.detection_pub.publish(json.dumps(detections))
            self.distance_pub.publish(json.dumps(distances))
            
            # Determine safety status
            safety_status = self.evaluate_safety(distances)
            self.safety_pub.publish(safety_status)
            
            # Publish emergency status
            self.emergency_pub.publish(emergency_detected)
            
            # Send data to server
            self.send_to_server(detections, distances, safety_status, emergency_detected)
            
        except Exception as e:
            rospy.logerr(f"Error in image processing: {e}")
    
    def evaluate_safety(self, distances):
        """Evaluate overall safety status based on detected objects and distances"""
        if not distances:
            return "SAFE"
        
        min_distance = min([d['distance'] for d in distances if d['distance'] is not None])
        
        if min_distance < self.emergency_distance:
            return "EMERGENCY"
        elif min_distance < self.safe_distance:
            return "CAUTION"
        else:
            return "SAFE"
    
    def send_to_server(self, detections, distances, safety_status, emergency):
        """Send detection data to server via API"""
        try:
            # Throttle server updates to avoid overwhelming
            current_time = time.time()
            if current_time - self.last_server_update < 0.5:  # 2 Hz max
                return
            
            data = {
                'bot_id': self.bot_id,
                'timestamp': current_time,
                'detections': detections,
                'distances': distances,
                'safety_status': safety_status,
                'emergency': emergency
            }
            
            response = requests.post(
                f"{self.server_url}/api/detection_update",
                json=data,
                timeout=1.0
            )
            
            if response.status_code == 200:
                self.last_server_update = current_time
                # Process server response for commands
                server_response = response.json()
                if 'command' in server_response:
                    self.handle_server_command(server_response['command'])
            
        except requests.exceptions.RequestException as e:
            rospy.logwarn(f"Failed to send data to server: {e}")
    
    def handle_server_command(self, command):
        """Handle commands received from server"""
        rospy.loginfo(f"Received server command: {command}")
        # Commands will be handled by other nodes through topics
        cmd_pub = rospy.Publisher('/yolo/server_command', String, queue_size=10)
        cmd_pub.publish(json.dumps(command))

if __name__ == '__main__':
    try:
        detector = YOLOv8Detector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass