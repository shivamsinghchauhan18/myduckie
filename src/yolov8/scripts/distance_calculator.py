#!/usr/bin/env python3

import rospy
import json
import numpy as np
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Point

class DistanceCalculator:
    def __init__(self):
        rospy.init_node('distance_calculator', anonymous=True)
        
        # Camera calibration parameters
        self.camera_height = rospy.get_param('~camera_height', 0.1)  # meters from ground
        self.camera_angle = rospy.get_param('~camera_angle', 0.0)    # radians
        self.image_height = rospy.get_param('~image_height', 480)
        self.image_width = rospy.get_param('~image_width', 640)
        
        # Publishers
        self.refined_distance_pub = rospy.Publisher('/yolo/refined_distances', String, queue_size=10)
        self.closest_object_pub = rospy.Publisher('/yolo/closest_object', String, queue_size=10)
        
        # Subscribers
        self.detection_sub = rospy.Subscriber('/yolo/detections', String, self.detection_callback)
        
        rospy.loginfo("Distance Calculator initialized")
    
    def detection_callback(self, msg):
        try:
            detections = json.loads(msg.data)
            refined_distances = []
            
            for detection in detections:
                # Use multiple methods to calculate distance
                bbox_distance = detection.get('distance')
                ground_distance = self.calculate_ground_distance(detection)
                stereo_distance = self.estimate_stereo_distance(detection)
                
                # Combine distance estimates
                final_distance = self.combine_distance_estimates(
                    bbox_distance, ground_distance, stereo_distance
                )
                
                refined_detection = {
                    'class': detection['class'],
                    'bbox_distance': bbox_distance,
                    'ground_distance': ground_distance,
                    'final_distance': final_distance,
                    'position': detection['center_x'],
                    'confidence': detection['confidence'],
                    'threat_level': self.calculate_threat_level(final_distance, detection['class'])
                }
                
                refined_distances.append(refined_detection)
            
            # Publish refined distances
            self.refined_distance_pub.publish(json.dumps(refined_distances))
            
            # Find and publish closest object
            if refined_distances:
                closest = min(refined_distances, 
                            key=lambda x: x['final_distance'] if x['final_distance'] else float('inf'))
                self.closest_object_pub.publish(json.dumps(closest))
            
        except Exception as e:
            rospy.logerr(f"Error in distance calculation: {e}")
    
    def calculate_ground_distance(self, detection):
        """Calculate distance using ground plane assumption"""
        try:
            # Get bottom center of bounding box
            bbox = detection['bbox']
            bottom_y = bbox[3]  # y2
            
            # Convert pixel coordinates to real-world distance
            # This assumes the object is on the ground plane
            pixel_from_horizon = self.image_height / 2 - bottom_y
            
            if pixel_from_horizon <= 0:
                return None
            
            # Simple ground plane distance calculation
            distance = self.camera_height / np.tan(pixel_from_horizon * 0.001)  # Rough approximation
            return max(0.1, min(distance, 10.0))  # Clamp between 0.1m and 10m
            
        except Exception:
            return None
    
    def estimate_stereo_distance(self, detection):
        """Estimate distance using object size and position heuristics"""
        try:
            bbox = detection['bbox']
            object_height = bbox[3] - bbox[1]
            object_width = bbox[2] - bbox[0]
            
            # Use object dimensions to estimate distance
            # Larger objects in image are typically closer
            size_factor = (object_height * object_width) / (self.image_height * self.image_width)
            
            # Rough heuristic based on object size in image
            estimated_distance = 2.0 / (size_factor + 0.1)
            return max(0.2, min(estimated_distance, 8.0))
            
        except Exception:
            return None
    
    def combine_distance_estimates(self, bbox_dist, ground_dist, stereo_dist):
        """Combine multiple distance estimates using weighted average"""
        distances = []
        weights = []
        
        if bbox_dist is not None and bbox_dist > 0:
            distances.append(bbox_dist)
            weights.append(0.5)  # High weight for bbox method
        
        if ground_dist is not None and ground_dist > 0:
            distances.append(ground_dist)
            weights.append(0.3)  # Medium weight for ground plane
        
        if stereo_dist is not None and stereo_dist > 0:
            distances.append(stereo_dist)
            weights.append(0.2)  # Lower weight for size heuristic
        
        if not distances:
            return None
        
        # Weighted average
        weighted_sum = sum(d * w for d, w in zip(distances, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight
    
    def calculate_threat_level(self, distance, object_class):
        """Calculate threat level based on distance and object type"""
        if distance is None:
            return "UNKNOWN"
        
        # Define threat levels based on object type and distance
        high_threat_objects = ['person', 'car', 'truck', 'bus', 'motorcycle']
        medium_threat_objects = ['bicycle', 'stop sign', 'traffic light']
        
        if distance < 0.3:
            return "CRITICAL"
        elif distance < 0.8:
            if object_class in high_threat_objects:
                return "HIGH"
            else:
                return "MEDIUM"
        elif distance < 2.0:
            if object_class in high_threat_objects:
                return "MEDIUM"
            else:
                return "LOW"
        else:
            return "LOW"

if __name__ == '__main__':
    try:
        calculator = DistanceCalculator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass