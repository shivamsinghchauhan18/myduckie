#!/usr/bin/env python3

"""
Advanced Lane Controller Node for DuckieBot - Enhanced Lane Following
Implements sophisticated PID control with predictive steering and adaptive behavior
"""

import rospy
import numpy as np
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Bool, Float32, Header, String
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, LanePose
import math

class AdvancedLaneController:
    def __init__(self):
        rospy.init_node('advanced_lane_controller', anonymous=True)
        
        # Publishers - Full DuckieBot integration
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.wheels_pub = rospy.Publisher('/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.duckiebot_vel_pub = rospy.Publisher('/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
        
        # Lane following specific publishers
        self.lane_pose_pub = rospy.Publisher('/lane_follower/lane_pose', LanePose, queue_size=1)
        self.performance_pub = rospy.Publisher('/lane_follower/performance', Float32, queue_size=1)
        self.control_status_pub = rospy.Publisher('/lane_follower/control_status', String, queue_size=1)
        
        # Subscribers
        self.lane_detection_sub = rospy.Subscriber('/lane_follower/lane_detection', Point, self.lane_detection_callback)
        self.lane_found_sub = rospy.Subscriber('/lane_follower/lane_found', Bool, self.lane_found_callback)
        self.curvature_sub = rospy.Subscriber('/lane_follower/lane_curvature', Float32, self.curvature_callback)
        self.obstacle_sub = rospy.Subscriber('/lane_follower/obstacle_detected', Bool, self.obstacle_callback)
        
        # Enhanced control parameters
        self.max_speed = rospy.get_param('~max_speed', 0.3)  # Conservative for lane following
        self.target_speed = rospy.get_param('~target_speed', 0.25)
        self.wheel_distance = rospy.get_param('~wheel_distance', 0.1)
        
        # Advanced PID parameters for lateral control
        self.kp_lateral = rospy.get_param('~kp_lateral', 2.0)
        self.ki_lateral = rospy.get_param('~ki_lateral', 0.1)
        self.kd_lateral = rospy.get_param('~kd_lateral', 0.5)
        
        # Heading control PID
        self.kp_heading = rospy.get_param('~kp_heading', 1.5)
        self.ki_heading = rospy.get_param('~ki_heading', 0.05)
        self.kd_heading = rospy.get_param('~kd_heading', 0.3)
        
        # Curvature-based speed control
        self.kp_speed = rospy.get_param('~kp_speed', 0.8)
        self.min_speed_factor = rospy.get_param('~min_speed_factor', 0.4)
        
        # State variables
        self.lane_detection = None
        self.lane_found = False
        self.lane_curvature = 0.0
        self.obstacle_detected = False
        
        # Enhanced PID state
        self.lateral_error_integral = 0.0
        self.lateral_error_previous = 0.0
        self.heading_error_integral = 0.0
        self.heading_error_previous = 0.0
        
        # Predictive control
        self.prediction_horizon = rospy.get_param('~prediction_horizon', 0.3)  # seconds
        self.lane_history = []
        self.max_history_length = 10
        
        # Adaptive behavior
        self.straight_threshold = rospy.get_param('~straight_threshold', 0.1)
        self.curve_threshold = rospy.get_param('~curve_threshold', 0.3)
        self.sharp_curve_threshold = rospy.get_param('~sharp_curve_threshold', 0.6)
        
        # Performance tracking
        self.control_history = []
        self.lane_center_errors = []
        self.start_time = rospy.Time.now()
        
        # Control modes
        self.control_mode = "NORMAL"  # NORMAL, CURVE, SHARP_CURVE, RECOVERY
        self.recovery_counter = 0
        self.max_recovery_time = 20  # cycles
        
        # Smoothing and filtering
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.7)
        self.prev_angular_vel = 0.0
        self.prev_linear_vel = 0.0
        
        # Control loop timer
        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)  # 20 Hz
        
        rospy.loginfo("🛣️  Advanced Lane Controller initialized - Ready for lane following")
    
    def lane_detection_callback(self, msg):
        """Handle lane detection data (x: lateral offset, y: heading error, z: confidence)"""
        self.lane_detection = msg
        
        # Update lane history for predictive control
        self.lane_history.append({
            'timestamp': rospy.Time.now().to_sec(),
            'lateral_offset': msg.x,
            'heading_error': msg.y,
            'confidence': msg.z
        })
        
        # Keep history within limits
        if len(self.lane_history) > self.max_history_length:
            self.lane_history.pop(0)
    
    def lane_found_callback(self, msg):
        """Handle lane detection status"""
        if msg.data != self.lane_found:
            self.lane_found = msg.data
            if self.lane_found:
                rospy.loginfo("🛣️  Lane detected! Engaging lane following")
                self.control_mode = "NORMAL"
                self.recovery_counter = 0
            else:
                rospy.logwarn("⚠️  Lane lost! Entering recovery mode")
                self.control_mode = "RECOVERY"
                self.reset_pid_state()
    
    def curvature_callback(self, msg):
        """Handle lane curvature information"""
        self.lane_curvature = msg.data
        
        # Adaptive control mode based on curvature
        abs_curvature = abs(self.lane_curvature)
        if abs_curvature > self.sharp_curve_threshold:
            self.control_mode = "SHARP_CURVE"
        elif abs_curvature > self.curve_threshold:
            self.control_mode = "CURVE"
        elif self.lane_found:
            self.control_mode = "NORMAL"
    
    def obstacle_callback(self, msg):
        """Handle obstacle detection"""
        if msg.data and not self.obstacle_detected:
            rospy.logwarn("🚧 Obstacle detected! Emergency stop!")
        self.obstacle_detected = msg.data
    
    def reset_pid_state(self):
        """Reset PID integral terms"""
        self.lateral_error_integral = 0.0
        self.heading_error_integral = 0.0
    
    def predict_lane_position(self):
        """Predict future lane position based on history"""
        if len(self.lane_history) < 3:
            return None
        
        # Simple linear prediction based on recent trend
        recent_data = self.lane_history[-3:]
        
        # Calculate velocity of lateral offset change
        dt = recent_data[-1]['timestamp'] - recent_data[0]['timestamp']
        if dt > 0:
            lateral_velocity = (recent_data[-1]['lateral_offset'] - recent_data[0]['lateral_offset']) / dt
            heading_velocity = (recent_data[-1]['heading_error'] - recent_data[0]['heading_error']) / dt
            
            # Predict position after prediction_horizon
            predicted_lateral = recent_data[-1]['lateral_offset'] + lateral_velocity * self.prediction_horizon
            predicted_heading = recent_data[-1]['heading_error'] + heading_velocity * self.prediction_horizon
            
            return {
                'lateral_offset': predicted_lateral,
                'heading_error': predicted_heading
            }
        
        return None
    
    def calculate_adaptive_control_commands(self):
        """Calculate control commands with adaptive behavior"""
        if not self.lane_detection:
            return 0.0, 0.0
        
        dt = 0.05  # 20 Hz control loop
        
        # Get current measurements
        lateral_error = self.lane_detection.x  # Distance from lane center
        heading_error = self.lane_detection.y  # Angle relative to lane direction
        confidence = self.lane_detection.z
        
        # Adaptive PID gains based on control mode
        kp_lat, ki_lat, kd_lat = self.get_adaptive_gains()
        
        # Predictive control enhancement
        prediction = self.predict_lane_position()
        if prediction and confidence > 0.7:
            # Blend current and predicted errors
            blend_factor = 0.3
            lateral_error = (1 - blend_factor) * lateral_error + blend_factor * prediction['lateral_offset']
            heading_error = (1 - blend_factor) * heading_error + blend_factor * prediction['heading_error']
        
        # Lateral PID control
        self.lateral_error_integral += lateral_error * dt
        self.lateral_error_integral = np.clip(self.lateral_error_integral, -0.5, 0.5)  # Anti-windup
        
        lateral_derivative = (lateral_error - self.lateral_error_previous) / dt
        lateral_control = -(kp_lat * lateral_error + 
                          ki_lat * self.lateral_error_integral + 
                          kd_lat * lateral_derivative)
        
        # Heading PID control
        self.heading_error_integral += heading_error * dt
        self.heading_error_integral = np.clip(self.heading_error_integral, -0.3, 0.3)
        
        heading_derivative = (heading_error - self.heading_error_previous) / dt
        heading_control = -(self.kp_heading * heading_error + 
                           self.ki_heading * self.heading_error_integral + 
                           self.kd_heading * heading_derivative)
        
        # Combine lateral and heading control
        angular_vel = lateral_control + heading_control
        
        # Curvature-based speed adaptation
        speed_factor = self.calculate_speed_factor()
        linear_vel = self.target_speed * speed_factor
        
        # Confidence-based control scaling
        if confidence < 0.5:
            angular_vel *= confidence * 2  # Reduce turning when uncertain
            linear_vel *= max(0.5, confidence * 2)  # Slow down when uncertain
        
        # Apply smoothing
        angular_vel = self.smoothing_factor * self.prev_angular_vel + (1 - self.smoothing_factor) * angular_vel
        linear_vel = self.smoothing_factor * self.prev_linear_vel + (1 - self.smoothing_factor) * linear_vel
        
        # Update previous values
        self.lateral_error_previous = lateral_error
        self.heading_error_previous = heading_error
        self.prev_angular_vel = angular_vel
        self.prev_linear_vel = linear_vel
        
        return linear_vel, angular_vel
    
    def get_adaptive_gains(self):
        """Get PID gains based on current control mode"""
        if self.control_mode == "SHARP_CURVE":
            return self.kp_lateral * 1.5, self.ki_lateral * 0.5, self.kd_lateral * 2.0
        elif self.control_mode == "CURVE":
            return self.kp_lateral * 1.2, self.ki_lateral * 0.8, self.kd_lateral * 1.5
        elif self.control_mode == "RECOVERY":
            return self.kp_lateral * 0.8, self.ki_lateral * 0.3, self.kd_lateral * 0.5
        else:  # NORMAL
            return self.kp_lateral, self.ki_lateral, self.kd_lateral
    
    def calculate_speed_factor(self):
        """Calculate speed reduction factor based on curvature and control mode"""
        abs_curvature = abs(self.lane_curvature)
        
        if self.control_mode == "SHARP_CURVE":
            return max(self.min_speed_factor, 1.0 - abs_curvature * 1.5)
        elif self.control_mode == "CURVE":
            return max(self.min_speed_factor * 1.5, 1.0 - abs_curvature)
        elif self.control_mode == "RECOVERY":
            return self.min_speed_factor
        else:
            return max(0.8, 1.0 - abs_curvature * 0.5)
    
    def control_loop(self, event):
        """Main control loop"""
        if self.obstacle_detected:
            self.emergency_stop()
            return
        
        if not self.lane_found:
            self.handle_lane_loss()
            return
        
        # Calculate control commands
        linear_vel, angular_vel = self.calculate_adaptive_control_commands()
        
        # Apply velocity limits
        linear_vel = np.clip(linear_vel, 0.0, self.max_speed)
        angular_vel = np.clip(angular_vel, -2.0, 2.0)
        
        # Publish commands
        self.publish_velocity_commands(linear_vel, angular_vel)
        
        # Publish lane pose for other nodes
        self.publish_lane_pose()
        
        # Track performance
        self.track_performance(linear_vel, angular_vel)
        
        # Publish control status
        self.publish_control_status()
    
    def handle_lane_loss(self):
        """Handle lane loss with recovery behavior"""
        self.recovery_counter += 1
        
        if self.recovery_counter < self.max_recovery_time:
            # Gentle forward motion while searching
            self.publish_velocity_commands(0.1, 0.0)
            rospy.loginfo_throttle(2, f"🔍 Searching for lane... ({self.recovery_counter}/{self.max_recovery_time})")
        else:
            # Stop after recovery timeout
            self.stop_robot()
            rospy.logwarn_throttle(5, "❌ Lane recovery failed - stopping robot")
    
    def publish_velocity_commands(self, linear_vel, angular_vel):
        """Publish velocity commands to all supported topics"""
        current_time = rospy.Time.now()
        
        # Standard ROS Twist message
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist_msg)
        
        # DuckieBot Twist2DStamped message
        duckiebot_msg = Twist2DStamped()
        duckiebot_msg.header = Header()
        duckiebot_msg.header.stamp = current_time
        duckiebot_msg.header.frame_id = "base_link"
        duckiebot_msg.v = linear_vel
        duckiebot_msg.omega = angular_vel
        self.duckiebot_vel_pub.publish(duckiebot_msg)
        
        # DuckieBot WheelsCmdStamped message
        vel_left = linear_vel - angular_vel * self.wheel_distance / 2.0
        vel_right = linear_vel + angular_vel * self.wheel_distance / 2.0
        
        wheels_msg = WheelsCmdStamped()
        wheels_msg.header = Header()
        wheels_msg.header.stamp = current_time
        wheels_msg.header.frame_id = "base_link"
        wheels_msg.vel_left = vel_left
        wheels_msg.vel_right = vel_right
        self.wheels_pub.publish(wheels_msg)
    
    def publish_lane_pose(self):
        """Publish current lane pose"""
        if not self.lane_detection:
            return
        
        lane_pose = LanePose()
        lane_pose.header = Header()
        lane_pose.header.stamp = rospy.Time.now()
        lane_pose.header.frame_id = "base_link"
        
        lane_pose.x = self.lane_detection.x  # Lateral offset
        lane_pose.y = self.lane_detection.y  # Heading error
        lane_pose.z = 1.0 if self.lane_found else 0.0  # In lane status
        lane_pose.status = 0 if self.lane_found else 1
        lane_pose.v_ref = self.target_speed
        
        self.lane_pose_pub.publish(lane_pose)
    
    def publish_control_status(self):
        """Publish current control status"""
        status_msg = f"Mode: {self.control_mode}, Curvature: {self.lane_curvature:.3f}, Recovery: {self.recovery_counter}"
        self.control_status_pub.publish(String(status_msg))
    
    def track_performance(self, linear_vel, angular_vel):
        """Track control performance metrics"""
        if self.lane_detection:
            self.lane_center_errors.append(abs(self.lane_detection.x))
            
            # Keep only recent errors
            if len(self.lane_center_errors) > 100:
                self.lane_center_errors.pop(0)
            
            # Calculate performance score
            if len(self.lane_center_errors) >= 10:
                avg_error = np.mean(self.lane_center_errors[-20:])
                performance_score = max(0, 100 - avg_error * 200)  # Scale error to 0-100
                self.performance_pub.publish(Float32(performance_score))
    
    def stop_robot(self):
        """Stop the robot"""
        self.publish_velocity_commands(0.0, 0.0)
    
    def emergency_stop(self):
        """Emergency stop with immediate brake"""
        self.publish_velocity_commands(0.0, 0.0)
        self.reset_pid_state()

if __name__ == '__main__':
    try:
        controller = AdvancedLaneController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass