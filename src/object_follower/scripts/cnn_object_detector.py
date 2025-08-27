#!/usr/bin/env python3

"""
CNN Object Detector Node for DuckieBot - Phase 3
Integrates deep learning models (YOLO, MobileNet) with existing color detection
"""

import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float32, String, Int32
from cv_bridge import CvBridge
import time
import os

class CNNObjectDetector:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('cnn_object_detector', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.target_pub = rospy.Publisher('/object_follower/target_position', Point, queue_size=1)
        self.target_found_pub = rospy.Publisher('/object_follower/target_found', Bool, queue_size=1)
        self.distance_pub = rospy.Publisher('/object_follower/target_distance', Float32, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/object_follower/cnn_debug_image', Image, queue_size=1)
        self.detection_info_pub = rospy.Publisher('/object_follower/cnn_detection_info', String, queue_size=1)
        self.confidence_pub = rospy.Publisher('/object_follower/detection_confidence', Float32, queue_size=1)
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.target_class_sub = rospy.Subscriber('/object_follower/target_class', Int32, self.target_class_callback)
        
        # Parameters
        self.detection_mode = rospy.get_param('~detection_mode', 'hybrid')  # 'cnn', 'color', 'hybrid'
        self.model_type = rospy.get_param('~model_type', 'yolov5n')  # 'yolov5n', 'mobilenet'
        self.target_class = rospy.get_param('~target_class', 0)  # 0=person, others=COCO classes
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)
        
        # CNN Model initialization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_loaded = False
        self.load_cnn_model()
        
        # COCO class names
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange'
        ]
        
        # Tracking variables
        self.last_detection_time = time.time()
        self.detection_history = []
        self.target_id = None
        
        # Performance metrics
        self.processing_times = []
        self.detection_count = 0
        self.frame_count = 0
        
        # Color detection fallback (from Phase 2)
        self.color_ranges = {
            'red': [np.array([0, 50, 50]), np.array([10, 255, 255])],
            'red2': [np.array([170, 50, 50]), np.array([180, 255, 255])],
            'blue': [np.array([100, 50, 50]), np.array([130, 255, 255])],
            'green': [np.array([40, 50, 50]), np.array([80, 255, 255])],
        }
        
        rospy.loginfo(f"CNN Object Detector initialized - Mode: {self.detection_mode}, Model: {self.model_type}")
        rospy.loginfo(f"Target class: {self.coco_classes[self.target_class] if self.target_class < len(self.coco_classes) else 'Unknown'}")
        
    def load_cnn_model(self):
        """Load the specified CNN model"""
        try:
            if self.model_type == 'yolov5n':
                # Try to load YOLOv5 nano
                try:
                    import ultralytics
                    from ultralytics import YOLO
                    self.model = YOLO('yolov5n.pt')  # Will download if not exists
                    self.model_loaded = True
                    rospy.loginfo("✅ YOLOv5 nano model loaded successfully")
                except ImportError:
                    rospy.logwarn("⚠️ ultralytics not installed, will try alternative method")
                    self.load_torch_hub_yolo()
                except Exception as e:
                    rospy.logwarn(f"⚠️ Could not load YOLOv5 with ultralytics: {e}")
                    self.load_torch_hub_yolo()
                    
            elif self.model_type == 'mobilenet':
                # Load MobileNet for classification
                import torchvision.models as models
                self.model = models.mobilenet_v3_small(pretrained=True)
                self.model.eval()
                self.model_loaded = True
                rospy.loginfo("✅ MobileNet model loaded successfully")
                
        except Exception as e:
            rospy.logwarn(f"⚠️ Could not load CNN model: {e}")
            rospy.logwarn("⚠️ Falling back to color detection only")
            self.detection_mode = 'color'
            self.model_loaded = False
    
    def load_torch_hub_yolo(self):
        """Alternative YOLOv5 loading method"""
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            self.model.eval()
            self.model_loaded = True
            rospy.loginfo("✅ YOLOv5 loaded via torch.hub")
        except Exception as e:
            rospy.logwarn(f"⚠️ torch.hub YOLOv5 loading failed: {e}")
            self.model_loaded = False
    
    def target_class_callback(self, msg):
        """Update target class to follow"""
        self.target_class = msg.data
        class_name = self.coco_classes[self.target_class] if self.target_class < len(self.coco_classes) else 'Unknown'
        rospy.loginfo(f"🎯 Target class changed to: {class_name} (ID: {self.target_class})")
    
    def image_callback(self, msg):
        """Main image processing callback"""
        try:
            start_time = time.time()
            
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.frame_count += 1
            
            # Choose detection method
            if self.detection_mode == 'cnn' and self.model_loaded:
                self.detect_with_cnn(cv_image)
            elif self.detection_mode == 'color':
                self.detect_with_color(cv_image)
            elif self.detection_mode == 'hybrid' and self.model_loaded:
                self.detect_hybrid(cv_image)
            else:
                # Fallback to color detection
                self.detect_with_color(cv_image)
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
                
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")
    
    def detect_with_cnn(self, image):
        """CNN-based object detection"""
        try:
            # Run inference
            results = self.model(image)
            
            # Process results
            detections = []
            if hasattr(results, 'pandas'):
                # YOLOv5 results
                df = results.pandas().xyxy[0]  # Get detections as pandas DataFrame
                for _, detection in df.iterrows():
                    if detection['class'] == self.target_class and detection['confidence'] > self.confidence_threshold:
                        detections.append({
                            'bbox': [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']],
                            'confidence': detection['confidence'],
                            'class': detection['class']
                        })
            else:
                # Handle different result formats
                try:
                    boxes = results[0].boxes if hasattr(results[0], 'boxes') else results.xyxy[0]
                    for box in boxes:
                        if hasattr(box, 'cls') and hasattr(box, 'conf'):
                            class_id = int(box.cls)
                            confidence = float(box.conf)
                            if class_id == self.target_class and confidence > self.confidence_threshold:
                                bbox = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else box[:4].tolist()
                                detections.append({
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'class': class_id
                                })
                except:
                    rospy.logwarn_throttle(5, "Could not parse CNN results")
            
            self.process_cnn_detections(image, detections)
            
        except Exception as e:
            rospy.logwarn_throttle(5, f"CNN detection error: {e}")
            # Fallback to color detection
            self.detect_with_color(image)
    
    def detect_with_color(self, image):
        """Color-based detection (fallback from Phase 2)"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create red mask (most common target)
        mask1 = cv2.inRange(hsv, self.color_ranges['red'][0], self.color_ranges['red'][1])
        mask2 = cv2.inRange(hsv, self.color_ranges['red2'][0], self.color_ranges['red2'][1])
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Publish detection
                self.publish_detection(image, center_x, center_y, w, h, 0.8, "color_red")
                return
        
        # No detection
        self.publish_no_detection()
    
    def detect_hybrid(self, image):
        """Hybrid detection combining CNN and color"""
        # Try CNN first
        cnn_success = False
        try:
            self.detect_with_cnn(image)
            cnn_success = True
        except:
            pass
        
        # If CNN fails or low confidence, use color detection
        if not cnn_success:
            self.detect_with_color(image)
    
    def process_cnn_detections(self, image, detections):
        """Process CNN detection results"""
        if not detections:
            self.publish_no_detection()
            return
        
        # Select best detection (highest confidence)
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        # Extract bbox
        x1, y1, x2, y2 = map(int, best_detection['bbox'])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        # Get class name
        class_name = self.coco_classes[best_detection['class']] if best_detection['class'] < len(self.coco_classes) else 'unknown'
        
        # Publish detection
        self.publish_detection(image, center_x, center_y, width, height, best_detection['confidence'], f"cnn_{class_name}")
    
    def publish_detection(self, image, center_x, center_y, width, height, confidence, method):
        """Publish detection results"""
        # Calculate normalized position
        img_height, img_width = image.shape[:2]
        norm_x = (center_x - img_width // 2) / (img_width // 2)  # -1 to 1
        norm_y = (center_y - img_height // 2) / (img_height // 2)  # -1 to 1
        
        # Estimate distance (simple approximation)
        distance = self.estimate_distance(width, height)
        
        # Publish target position
        target_point = Point()
        target_point.x = norm_x
        target_point.y = norm_y
        target_point.z = distance
        self.target_pub.publish(target_point)
        
        # Publish target found
        self.target_found_pub.publish(Bool(True))
        
        # Publish distance
        self.distance_pub.publish(Float32(distance))
        
        # Publish confidence
        self.confidence_pub.publish(Float32(confidence))
        
        # Publish detection info
        info = f"Method: {method}, Confidence: {confidence:.2f}, Distance: {distance:.2f}m"
        self.detection_info_pub.publish(String(info))
        
        # Create and publish debug image
        debug_image = image.copy()
        cv2.rectangle(debug_image, (int(center_x - width//2), int(center_y - height//2)), 
                     (int(center_x + width//2), int(center_y + height//2)), (0, 255, 0), 2)
        cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(debug_image, f"{method}: {confidence:.2f}", 
                   (center_x - 50, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(debug_image, f"Dist: {distance:.2f}m", 
                   (center_x - 30, center_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            rospy.logwarn(f"Could not publish debug image: {e}")
        
        self.detection_count += 1
        rospy.loginfo_throttle(2, f"🎯 {method} detection: confidence={confidence:.2f}, distance={distance:.2f}m")
    
    def publish_no_detection(self):
        """Publish when no target is found"""
        self.target_found_pub.publish(Bool(False))
        self.confidence_pub.publish(Float32(0.0))
    
    def estimate_distance(self, width, height):
        """Estimate distance based on bounding box size"""
        # Simple approximation - adjust based on known object sizes
        if self.target_class == 0:  # Person
            # Average person height ~1.7m, pixel height estimation
            focal_length = 500  # Approximate camera focal length
            real_height = 1.7  # meters
            distance = (real_height * focal_length) / height
        else:
            # Generic object distance estimation
            area = width * height
            distance = max(0.5, min(5.0, 5000.0 / area))  # Empirical formula
        
        return max(0.3, min(distance, 5.0))  # Clamp between 30cm and 5m

if __name__ == '__main__':
    try:
        detector = CNNObjectDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass