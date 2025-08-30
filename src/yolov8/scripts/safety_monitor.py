#!/usr/bin/env python3

import rospy
import json
import time
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist
import numpy as np

class SafetyMonitor:
    def __init__(self):
        rospy.init_node('safety_monitor', anonymous=True)
        
        # Safety parameters
        self.critical_distance = 0.2    # meters
        self.warning_distance = 0.5     # meters
        self.safe_distance = 1.0        # meters
        self.max_speed_limit = 0.4      # m/s
        self.emergency_timeout = 5.0    # seconds
        
        # Monitoring state
        self.safety_level = "SAFE"
        self.last_detection_time = time.time()
        self.emergency_start_time = None
        self.consecutive_warnings = 0
        self.speed_violations = 0
        
        # Data storage for analysis
        self.detection_history = []
        self.safety_history = []
        self.max_history_length = 100
        
        # Publishers
        self.safety_alert_pub = rospy.Publisher('/yolo/safety_alert', String, queue_size=10)
        self.system_override_pub = rospy.Publisher('/yolo/system_override', Bool, queue_size=10)
        self.safety_metrics_pub = rospy.Publisher('/yolo/safety_metrics', String, queue_size=10)
        
        # Subscribers
        self.detection_sub = rospy.Subscriber('/yolo/detections', String, self.detection_callback)
        self.distance_sub = rospy.Subscriber('/yolo/refined_distances', String, self.distance_callback)
        self.safety_status_sub = rospy.Subscriber('/yolo/safety_status', String, self.safety_status_callback)
        self.emergency_sub = rospy.Subscriber('/yolo/emergency', Bool, self.emergency_callback)
        self.avoidance_status_sub = rospy.Subscriber('/yolo/avoidance_status', String, self.avoidance_status_callback)
        
        # Safety monitoring timer
        self.monitor_timer = rospy.Timer(rospy.Duration(0.5), self.safety_monitoring_loop)
        
        rospy.loginfo("Safety Monitor initialized")
    
    def detection_callback(self, msg):
        try:
            detections = json.loads(msg.data)
            self.last_detection_time = time.time()
            
            # Store detection for analysis
            detection_record = {
                'timestamp': time.time(),
                'count': len(detections),
                'objects': [d['class'] for d in detections]
            }
            
            self.detection_history.append(detection_record)
            if len(self.detection_history) > self.max_history_length:
                self.detection_history.pop(0)
                
        except Exception as e:
            rospy.logerr(f"Error processing detections in safety monitor: {e}")
    
    def distance_callback(self, msg):
        try:
            distances = json.loads(msg.data)
            self.analyze_distance_safety(distances)
        except Exception as e:
            rospy.logerr(f"Error processing distances in safety monitor: {e}")
    
    def safety_status_callback(self, msg):
        self.current_safety_status = msg.data
        
        # Record safety status history
        safety_record = {
            'timestamp': time.time(),
            'status': msg.data
        }
        
        self.safety_history.append(safety_record)
        if len(self.safety_history) > self.max_history_length:
            self.safety_history.pop(0)
    
    def emergency_callback(self, msg):
        if msg.data and self.emergency_start_time is None:
            self.emergency_start_time = time.time()
            rospy.logwarn("Emergency state detected by safety monitor")
        elif not msg.data and self.emergency_start_time is not None:
            emergency_duration = time.time() - self.emergency_start_time
            rospy.loginfo(f"Emergency resolved after {emergency_duration:.2f} seconds")
            self.emergency_start_time = None
    
    def avoidance_status_callback(self, msg):
        try:
            self.avoidance_status = json.loads(msg.data)
        except Exception as e:
            rospy.logerr(f"Error parsing avoidance status: {e}")
    
    def analyze_distance_safety(self, distances):
        """Analyze distance data for safety violations"""
        if not distances:
            return
        
        min_distance = float('inf')
        critical_objects = []
        warning_objects = []
        
        for dist_data in distances:
            distance = dist_data.get('final_distance')
            if distance is None:
                continue
                
            min_distance = min(min_distance, distance)
            obj_class = dist_data.get('class', 'unknown')
            
            if distance < self.critical_distance:
                critical_objects.append({
                    'class': obj_class,
                    'distance': distance,
                    'threat_level': 'CRITICAL'
                })
            elif distance < self.warning_distance:
                warning_objects.append({
                    'class': obj_class,
                    'distance': distance,
                    'threat_level': 'WARNING'
                })
        
        # Update safety level based on analysis
        if critical_objects:
            self.safety_level = "CRITICAL"
            self.consecutive_warnings += 1
        elif warning_objects:
            self.safety_level = "WARNING"
            self.consecutive_warnings += 1
        else:
            self.safety_level = "SAFE"
            self.consecutive_warnings = 0
        
        # Generate safety alerts
        if critical_objects:
            self.generate_safety_alert("CRITICAL", critical_objects)
        elif warning_objects and self.consecutive_warnings > 3:
            self.generate_safety_alert("PERSISTENT_WARNING", warning_objects)
    
    def safety_monitoring_loop(self, event):
        """Main safety monitoring loop"""
        try:
            current_time = time.time()
            
            # Check for system timeouts
            self.check_system_timeouts(current_time)
            
            # Check emergency duration
            self.check_emergency_duration(current_time)
            
            # Analyze safety trends
            self.analyze_safety_trends()
            
            # Publish safety metrics
            self.publish_safety_metrics()
            
            # Check for system override conditions
            self.check_override_conditions()
            
        except Exception as e:
            rospy.logerr(f"Error in safety monitoring loop: {e}")
    
    def check_system_timeouts(self, current_time):
        """Check for system component timeouts"""
        detection_timeout = current_time - self.last_detection_time
        
        if detection_timeout > 10.0:  # No detections for 10 seconds
            self.generate_safety_alert("SYSTEM_TIMEOUT", {
                'component': 'detection_system',
                'timeout_duration': detection_timeout
            })
    
    def check_emergency_duration(self, current_time):
        """Check if emergency state has lasted too long"""
        if self.emergency_start_time is not None:
            emergency_duration = current_time - self.emergency_start_time
            
            if emergency_duration > self.emergency_timeout:
                self.generate_safety_alert("PROLONGED_EMERGENCY", {
                    'duration': emergency_duration,
                    'action': 'requesting_human_intervention'
                })
    
    def analyze_safety_trends(self):
        """Analyze safety trends over time"""
        if len(self.safety_history) < 10:
            return
        
        recent_history = self.safety_history[-10:]
        
        # Count safety status occurrences
        status_counts = {}
        for record in recent_history:
            status = record['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Check for concerning trends
        if status_counts.get('EMERGENCY', 0) > 3:
            self.generate_safety_alert("FREQUENT_EMERGENCIES", {
                'count': status_counts['EMERGENCY'],
                'timeframe': '10_recent_samples'
            })
        elif status_counts.get('CAUTION', 0) > 7:
            self.generate_safety_alert("PERSISTENT_CAUTION", {
                'count': status_counts['CAUTION'],
                'timeframe': '10_recent_samples'
            })
    
    def check_override_conditions(self):
        """Check if system override is needed"""
        override_needed = False
        override_reason = ""
        
        # Check for critical safety conditions
        if self.safety_level == "CRITICAL" and self.consecutive_warnings > 5:
            override_needed = True
            override_reason = "Persistent critical safety violations"
        
        # Check for emergency timeout
        if (self.emergency_start_time is not None and 
            time.time() - self.emergency_start_time > self.emergency_timeout):
            override_needed = True
            override_reason = "Emergency timeout exceeded"
        
        if override_needed:
            self.system_override_pub.publish(True)
            self.generate_safety_alert("SYSTEM_OVERRIDE", {
                'reason': override_reason,
                'timestamp': time.time()
            })
    
    def generate_safety_alert(self, alert_type, alert_data):
        """Generate and publish safety alert"""
        alert = {
            'type': alert_type,
            'timestamp': time.time(),
            'safety_level': self.safety_level,
            'data': alert_data
        }
        
        self.safety_alert_pub.publish(json.dumps(alert))
        
        # Log based on severity
        if alert_type in ["CRITICAL", "SYSTEM_OVERRIDE", "PROLONGED_EMERGENCY"]:
            rospy.logerr(f"SAFETY ALERT: {alert_type} - {alert_data}")
        elif alert_type in ["PERSISTENT_WARNING", "FREQUENT_EMERGENCIES"]:
            rospy.logwarn(f"Safety Warning: {alert_type} - {alert_data}")
        else:
            rospy.loginfo(f"Safety Notice: {alert_type} - {alert_data}")
    
    def publish_safety_metrics(self):
        """Publish comprehensive safety metrics"""
        current_time = time.time()
        
        # Calculate metrics
        recent_detections = [d for d in self.detection_history 
                           if current_time - d['timestamp'] < 60.0]
        
        recent_safety = [s for s in self.safety_history 
                        if current_time - s['timestamp'] < 60.0]
        
        metrics = {
            'timestamp': current_time,
            'current_safety_level': self.safety_level,
            'consecutive_warnings': self.consecutive_warnings,
            'detections_per_minute': len(recent_detections),
            'emergency_active': self.emergency_start_time is not None,
            'emergency_duration': (current_time - self.emergency_start_time 
                                 if self.emergency_start_time else 0),
            'safety_distribution': self.calculate_safety_distribution(recent_safety),
            'system_health': self.assess_system_health()
        }
        
        self.safety_metrics_pub.publish(json.dumps(metrics))
    
    def calculate_safety_distribution(self, safety_records):
        """Calculate distribution of safety statuses"""
        if not safety_records:
            return {}
        
        distribution = {}
        for record in safety_records:
            status = record['status']
            distribution[status] = distribution.get(status, 0) + 1
        
        # Convert to percentages
        total = len(safety_records)
        for status in distribution:
            distribution[status] = (distribution[status] / total) * 100
        
        return distribution
    
    def assess_system_health(self):
        """Assess overall system health"""
        current_time = time.time()
        
        # Check component responsiveness
        detection_lag = current_time - self.last_detection_time
        
        if detection_lag > 5.0:
            return "DEGRADED"
        elif self.consecutive_warnings > 10:
            return "UNSTABLE"
        elif self.safety_level == "CRITICAL":
            return "CRITICAL"
        else:
            return "HEALTHY"

if __name__ == '__main__':
    try:
        monitor = SafetyMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass