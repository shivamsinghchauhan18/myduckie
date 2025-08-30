#!/usr/bin/env python3

"""
Lane Performance Monitor for Advanced Lane Following System
Comprehensive performance tracking and analysis
"""

import rospy
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Bool, Float32, String
# Using standard ROS messages
import numpy as np
from collections import deque

class LanePerformanceMonitor:
    def __init__(self):
        rospy.init_node('lane_performance_monitor', anonymous=True)
        
        # Subscribers
        rospy.Subscriber('/lane_follower/lane_pose', Point, self.lane_pose_callback)
        rospy.Subscriber('/lane_follower/lane_found', Bool, self.lane_found_callback)
        rospy.Subscriber('/lane_follower/lane_center', Point, self.lane_center_callback)
        rospy.Subscriber('/lane_follower/detection_info', String, self.detection_info_callback)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        rospy.Subscriber('/lane_follower/control_status', Bool, self.control_status_callback)
        
        # Performance metrics storage
        self.window_size = 100
        
        # Metric queues
        self.lateral_errors = deque(maxlen=self.window_size)
        self.heading_errors = deque(maxlen=self.window_size)
        self.velocities = deque(maxlen=self.window_size)
        self.angular_velocities = deque(maxlen=self.window_size)
        self.lane_detection_states = deque(maxlen=self.window_size)
        self.control_states = deque(maxlen=self.window_size)
        self.response_times = deque(maxlen=50)
        
        # Tracking variables
        self.last_lane_found = False
        self.lane_loss_count = 0
        self.lane_acquisition_count = 0
        self.total_detections = 0
        self.start_time = rospy.Time.now()
        self.last_detection_time = None
        
        # Control performance
        self.control_smoothness = deque(maxlen=self.window_size)
        self.previous_linear_vel = 0.0
        self.previous_angular_vel = 0.0
        
        # Lane following specific metrics
        self.lane_centering_accuracy = deque(maxlen=self.window_size)
        self.heading_stability = deque(maxlen=self.window_size)
        self.speed_consistency = deque(maxlen=self.window_size)
        
        # Detection method tracking
        self.detection_methods = {}
        
        # Statistics timers
        self.stats_timer = rospy.Timer(rospy.Duration(10.0), self.print_detailed_stats)
        self.quick_stats_timer = rospy.Timer(rospy.Duration(3.0), self.print_quick_stats)
        
        rospy.loginfo("Lane Performance Monitor started - Tracking lane following metrics")
    
    def lane_pose_callback(self, msg):
        # Track lateral error (distance from lane center)
        lateral_error = abs(msg.x)
        self.lateral_errors.append(lateral_error)
        
        # Track heading error (angle deviation)
        heading_error = abs(msg.y)
        self.heading_errors.append(heading_error)
        
        # Calculate lane centering accuracy
        centering_accuracy = max(0, 1.0 - lateral_error)  # 1.0 = perfect center, 0 = edge
        self.lane_centering_accuracy.append(centering_accuracy)
        
        # Calculate heading stability
        heading_stability = max(0, 1.0 - heading_error)
        self.heading_stability.append(heading_stability)
        
        # Record response time if lane was just acquired
        if msg.z > 0.5 and not self.last_lane_found:
            current_time = rospy.Time.now()
            if self.last_detection_time:
                response_time = (current_time - self.last_detection_time).to_sec()
                self.response_times.append(response_time)
    
    def lane_found_callback(self, msg):
        current_found = msg.data
        self.lane_detection_states.append(current_found)
        
        # Track lane acquisition and loss
        if current_found and not self.last_lane_found:
            self.lane_acquisition_count += 1
            self.last_detection_time = rospy.Time.now()
            rospy.loginfo("✅ Lane acquired!")
        elif not current_found and self.last_lane_found:
            self.lane_loss_count += 1
            rospy.logwarn("❌ Lane lost!")
        
        if current_found:
            self.total_detections += 1
            
        self.last_lane_found = current_found
    
    def lane_center_callback(self, msg):
        # Track lane center positioning
        center_error = abs(msg.x)  # Distance from ideal center
        self.lane_centering_accuracy.append(max(0, 1.0 - center_error))
    
    def detection_info_callback(self, msg):
        # Parse detection method information
        try:
            info = msg.data
            if "Lanes:" in info:
                # Extract lane detection info
                parts = info.split(", ")
                for part in parts:
                    if "Conf:" in part:
                        conf_str = part.split("Conf: ")[1]
                        confidence = float(conf_str)
                        # Track detection confidence
                        pass
        except Exception as e:
            rospy.logwarn(f"Error parsing detection info: {e}")
    
    def cmd_vel_callback(self, msg):
        # Track velocity and control performance
        linear_vel = msg.linear.x
        angular_vel = msg.angular.z
        
        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(linear_vel**2 + angular_vel**2)
        self.velocities.append(velocity_magnitude)
        self.angular_velocities.append(abs(angular_vel))
        
        # Calculate control smoothness
        linear_smoothness = abs(linear_vel - self.previous_linear_vel)
        angular_smoothness = abs(angular_vel - self.previous_angular_vel)
        total_smoothness = linear_smoothness + angular_smoothness
        self.control_smoothness.append(total_smoothness)
        
        # Calculate speed consistency
        if len(self.velocities) > 1:
            speed_variation = abs(velocity_magnitude - self.velocities[-2])
            speed_consistency = max(0, 1.0 - speed_variation * 10)  # Scale factor
            self.speed_consistency.append(speed_consistency)
        
        self.previous_linear_vel = linear_vel
        self.previous_angular_vel = angular_vel
    
    def control_status_callback(self, msg):
        self.control_states.append(msg.data)
    
    def calculate_lane_following_score(self):
        """Calculate comprehensive lane following performance score (0-100)"""
        if not self.lateral_errors or not self.heading_errors:
            return 0
        
        # Lane centering performance (40% weight)
        avg_lateral_error = np.mean(self.lateral_errors)
        centering_score = max(0, 100 * (1 - avg_lateral_error * 2))  # Scale for lane width
        
        # Heading stability (30% weight)
        avg_heading_error = np.mean(self.heading_errors)
        heading_score = max(0, 100 * (1 - avg_heading_error * 3))  # Scale for angle sensitivity
        
        # Control smoothness (20% weight)
        if self.control_smoothness:
            smoothness_score = max(0, 100 * (1 - np.mean(self.control_smoothness) * 5))
        else:
            smoothness_score = 50
        
        # Lane detection consistency (10% weight)
        if self.lane_detection_states:
            detection_rate = sum(self.lane_detection_states) / len(self.lane_detection_states)
            detection_score = detection_rate * 100
        else:
            detection_score = 0
        
        # Weighted overall score
        overall_score = (centering_score * 0.4 + heading_score * 0.3 + 
                        smoothness_score * 0.2 + detection_score * 0.1)
        
        return overall_score
    
    def calculate_driving_quality_metrics(self):
        """Calculate detailed driving quality metrics"""
        metrics = {}
        
        # Centering accuracy
        if self.lane_centering_accuracy:
            metrics['centering_accuracy'] = np.mean(self.lane_centering_accuracy) * 100
            metrics['centering_consistency'] = (1 - np.std(self.lane_centering_accuracy)) * 100
        
        # Heading stability
        if self.heading_stability:
            metrics['heading_stability'] = np.mean(self.heading_stability) * 100
            metrics['heading_consistency'] = (1 - np.std(self.heading_stability)) * 100
        
        # Speed management
        if self.speed_consistency:
            metrics['speed_consistency'] = np.mean(self.speed_consistency) * 100
        
        if self.velocities:
            metrics['avg_speed'] = np.mean(self.velocities)
            metrics['speed_variation'] = np.std(self.velocities)
        
        # Control quality
        if self.control_smoothness:
            metrics['control_smoothness'] = max(0, 100 - np.mean(self.control_smoothness) * 100)
        
        return metrics
    
    def print_quick_stats(self, event):
        """Print quick status update"""
        if not self.lateral_errors:
            return
        
        lane_score = self.calculate_lane_following_score()
        current_detection = self.lane_detection_states[-1] if self.lane_detection_states else False
        
        # Status icons
        lane_icon = "🛣️" if current_detection else "❌"
        quality_icon = "🟢" if lane_score > 80 else "🟡" if lane_score > 60 else "🔴"
        
        # Current performance
        current_lateral = self.lateral_errors[-1] if self.lateral_errors else 0
        current_heading = self.heading_errors[-1] if self.heading_errors else 0
        
        rospy.loginfo(f"{lane_icon} Lane: {current_detection} | {quality_icon} Score: {lane_score:.1f}% | "
                     f"Lateral: {current_lateral:.3f} | Heading: {current_heading:.3f}")
    
    def print_detailed_stats(self, event):
        """Print comprehensive performance statistics"""
        if not self.lateral_errors:
            rospy.loginfo("📊 Waiting for lane following data...")
            return
        
        elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
        
        # Calculate all metrics
        lane_score = self.calculate_lane_following_score()
        quality_metrics = self.calculate_driving_quality_metrics()
        
        # Lane tracking metrics
        avg_lateral_error = np.mean(self.lateral_errors)
        std_lateral_error = np.std(self.lateral_errors)
        max_lateral_error = np.max(self.lateral_errors)
        
        avg_heading_error = np.mean(self.heading_errors)
        std_heading_error = np.std(self.heading_errors)
        
        # Control metrics
        avg_velocity = np.mean(self.velocities) if self.velocities else 0
        avg_angular_vel = np.mean(self.angular_velocities) if self.angular_velocities else 0
        avg_smoothness = np.mean(self.control_smoothness) if self.control_smoothness else 0
        
        # Detection metrics
        detection_rate = self.total_detections / max(elapsed_time, 1.0)
        if self.lane_detection_states:
            lane_uptime = (sum(self.lane_detection_states) / len(self.lane_detection_states)) * 100
        else:
            lane_uptime = 0
        
        # Response time
        avg_response_time = np.mean(self.response_times) if self.response_times else 0
        
        # Print comprehensive report
        rospy.loginfo("=" * 70)
        rospy.loginfo("🛣️  LANE FOLLOWING PERFORMANCE REPORT")
        rospy.loginfo("=" * 70)
        
        rospy.loginfo(f"🎯 LANE TRACKING PERFORMANCE:")
        rospy.loginfo(f"   Lateral Error: {avg_lateral_error:.4f} ± {std_lateral_error:.4f} (max: {max_lateral_error:.4f})")
        rospy.loginfo(f"   Heading Error: {avg_heading_error:.4f} ± {std_heading_error:.4f} radians")
        rospy.loginfo(f"   Lane Uptime: {lane_uptime:.1f}%")
        rospy.loginfo(f"   Detection Rate: {detection_rate:.1f} Hz")
        
        rospy.loginfo(f"🚗 DRIVING QUALITY:")
        if 'centering_accuracy' in quality_metrics:
            rospy.loginfo(f"   Centering Accuracy: {quality_metrics['centering_accuracy']:.1f}%")
            rospy.loginfo(f"   Centering Consistency: {quality_metrics['centering_consistency']:.1f}%")
        if 'heading_stability' in quality_metrics:
            rospy.loginfo(f"   Heading Stability: {quality_metrics['heading_stability']:.1f}%")
        if 'speed_consistency' in quality_metrics:
            rospy.loginfo(f"   Speed Consistency: {quality_metrics['speed_consistency']:.1f}%")
        
        rospy.loginfo(f"🎮 CONTROL PERFORMANCE:")
        rospy.loginfo(f"   Average Speed: {avg_velocity:.3f} m/s")
        rospy.loginfo(f"   Average Turn Rate: {avg_angular_vel:.3f} rad/s")
        rospy.loginfo(f"   Control Smoothness: {100-avg_smoothness*100:.1f}% (higher is smoother)")
        
        rospy.loginfo(f"⚡ SYSTEM METRICS:")
        rospy.loginfo(f"   Lane Acquisitions: {self.lane_acquisition_count}")
        rospy.loginfo(f"   Lane Losses: {self.lane_loss_count}")
        rospy.loginfo(f"   Avg Response Time: {avg_response_time:.3f}s")
        
        rospy.loginfo(f"📈 OVERALL ASSESSMENT:")
        grade = ("EXCELLENT" if lane_score > 85 else 
                "GOOD" if lane_score > 70 else 
                "FAIR" if lane_score > 50 else 
                "NEEDS IMPROVEMENT")
        rospy.loginfo(f"   Lane Following Score: {lane_score:.1f}% ({grade})")
        
        # Performance recommendations
        if avg_lateral_error > 0.1:
            rospy.loginfo("💡 RECOMMENDATION: Tune lateral PID gains for better centering")
        if avg_heading_error > 0.2:
            rospy.loginfo("💡 RECOMMENDATION: Adjust heading control parameters")
        if avg_smoothness > 0.05:
            rospy.loginfo("💡 RECOMMENDATION: Increase control smoothing factor")
        
        rospy.loginfo("=" * 70)
    
    def get_performance_summary(self):
        """Return performance summary as dictionary"""
        return {
            'lane_following_score': self.calculate_lane_following_score(),
            'avg_lateral_error': np.mean(self.lateral_errors) if self.lateral_errors else 0,
            'avg_heading_error': np.mean(self.heading_errors) if self.heading_errors else 0,
            'lane_uptime': (sum(self.lane_detection_states) / len(self.lane_detection_states) * 100) if self.lane_detection_states else 0,
            'lane_acquisitions': self.lane_acquisition_count,
            'lane_losses': self.lane_loss_count,
            'avg_velocity': np.mean(self.velocities) if self.velocities else 0,
            'control_smoothness': np.mean(self.control_smoothness) if self.control_smoothness else 0
        }

if __name__ == '__main__':
    try:
        monitor = LanePerformanceMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass