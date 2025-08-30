#!/usr/bin/env python3

"""
Enhanced Lane Controller Node for DuckieTown - Professional Grade
Advanced PID control system for precise lane following
"""

import rospy
import numpy as np
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Bool, Float32, Header
from duckietown_msgs.msg import Twist2DStamped

class EnhancedLaneController:
    def __init__(self):
        rospy.init_node('enhanced_lane_controller', anonymous=True)
        
        # Publishers - Full DuckieBot integration
        self.cmd_vel_pub = rospy.Publisher('/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
        # Additional publishers can be added later if needed
        
        # Performance monitoring publishers
        self.performance_pub = rospy.Publisher('/lane_follower/performance', Float32, queue_size=1)
        self.control_status_pub = rospy.Publisher('/lane_follower/control_status', Bool, queue_size=1)
        
        # Subscribers
        self.lane_pose_sub = rospy.Subscriber('/lane_follower/lane_pose', Point, self.lane_pose_callback)
        self.lane_found_sub = rospy.Subscriber('/lane_follower/lane_found', Bool, self.lane_found_callback)
        self.lane_center_sub = rospy.Subscriber('/lane_follower/lane_center', Point, self.lane_center_callback)
        self.obstacle_sub = rospy.Subscriber('/lane_follower/obstacle_detected', Bool, self.obstacle_callback)
        
        # Enhanced control parameters
        self.max_speed = rospy.get_param('~max_speed', 0.3)  # Conservative speed for lane following
        self.min_speed = rospy.get_param('~min_speed', 0.1)
        
        # Robot parameters
        self.wheel_distance = rospy.get_param('~wheel_distance', 0.1)  # meters between wheels
        
        # PID parameters for lateral control (steering)
        self.kp_lateral = rospy.get_param('~kp_lateral', 2.0)
        self.ki_lateral = rospy.get_param('~ki_lateral', 0.1)
        self.kd_lateral = rospy.get_param('~kd_lateral', 0.5)
        
        # PID parameters for heading control
        self.kp_heading = rospy.get_param('~kp_heading', 1.5)
        self.ki_heading = rospy.get_param('~ki_heading', 0.05)
        self.kd_heading = rospy.get_param('~kd_heading', 0.3)
        
        # Speed control parameters
        self.kp_speed = rospy.get_param('~kp_speed', 1.0)
        self.target_speed = rospy.get_param('~target_speed', 0.25)
        
        # State variables
        self.current_lane_pose = None
        self.lane_found = False
        self.lane_center = None
        self.obstacle_detected = False
        
        # PID state variables
        self.lateral_error_integral = 0.0
        self.lateral_error_previous = 0.0
        self.heading_error_integral = 0.0
        self.heading_error_previous = 0.0
        
        # Performance tracking
        self.control_history = []
        self.start_time = rospy.Time.now()
        
        # Advanced control features
        self.adaptive_speed = rospy.get_param('~adaptive_speed', True)
        self.curve_detection = rospy.get_param('~curve_detection', True)
        self.predictive_control = rospy.get_param('~predictive_control', True)
        
        # Curve detection parameters
        self.curve_threshold = rospy.get_param('~curve_threshold', 0.3)
        self.curve_speed_factor = rospy.get_param('~curve_speed_factor', 0.7)
        
        # Safety parameters
        self.max_lateral_error = rospy.get_param('~max_lateral_error', 0.8)
        self.emergency_stop_threshold = rospy.get_param('~emergency_stop_threshold', 1.0)
        
        # Control smoothing
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.8)
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0
        
        # Control loop timer
        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)  # 20 Hz
        
        rospy.loginfo("Enhanced Lane Controller - Professional lane following ready")
    
    def lane_pose_callback(self, msg):
        self.current_lane_pose = msg
    
    def lane_found_callback(self, msg):
        self.lane_found = msg.data
        if not self.lane_found:
            self.reset_pid_state()
    
    def lane_center_callback(self, msg):
        self.lane_center = msg
    
    def obstacle_callback(self, msg):
        if msg.data and not self.obstacle_detected:
            rospy.logwarn("🚧 Obstacle detected! Emergency stop!")
        self.obstacle_detected = msg.data
    
    def reset_pid_state(self):
        """Reset PID integral terms when lane is lost"""
        self.lateral_error_integral = 0.0
        self.heading_error_integral = 0.0
    
    def control_loop(self, event):
        """Enhanced control loop with advanced features"""
        if self.obstacle_detected:
            self.emergency_stop()
            return
        
        if not self.lane_found or self.current_lane_pose is None:
            self.handle_lane_loss()
            return
        
        # Calculate control commands
        linear_vel, angular_vel = self.calculate_advanced_control_commands()
        
        # Apply safety limits
        linear_vel, angular_vel = self.apply_safety_limits(linear_vel, angular_vel)
        
        # Apply smoothing
        linear_vel, angular_vel = self.apply_smoothing(linear_vel, angular_vel)
        
        # Publish commands
        self.publish_velocity_commands(linear_vel, angular_vel)
        
        # Track performance
        self.track_performance(linear_vel, angular_vel)
        
        # Publish control status
        self.control_status_pub.publish(Bool(True))
    
    def calculate_advanced_control_commands(self):
        """Advanced PID control with multiple features"""
        dt = 0.05  # 20 Hz control loop
        
        # Extract lane pose information
        lateral_error = self.current_lane_pose.x  # Lateral offset
        heading_error = self.current_lane_pose.y  # Heading angle error
        
        # LATERAL CONTROL (steering based on position error)
        self.lateral_error_integral += lateral_error * dt
        lateral_error_derivative = (lateral_error - self.lateral_error_previous) / dt
        
        # Anti-windup for lateral integral
        self.lateral_error_integral = np.clip(self.lateral_error_integral, -0.5, 0.5)
        
        lateral_control = -(self.kp_lateral * lateral_error + 
                           self.ki_lateral * self.lateral_error_integral + 
                           self.kd_lateral * lateral_error_derivative)
        
        # HEADING CONTROL (steering based on angle error)
        self.heading_error_integral += heading_error * dt
        heading_error_derivative = (heading_error - self.heading_error_previous) / dt
        
        # Anti-windup for heading integral
        self.heading_error_integral = np.clip(self.heading_error_integral, -0.3, 0.3)
        
        heading_control = -(self.kp_heading * heading_error + 
                           self.ki_heading * self.heading_error_integral + 
                           self.kd_heading * heading_error_derivative)
        
        # COMBINED STEERING CONTROL
        angular_vel = lateral_control + heading_control
        
        # ADAPTIVE SPEED CONTROL
        if self.adaptive_speed:
            linear_vel = self.calculate_adaptive_speed(lateral_error, heading_error, angular_vel)
        else:
            linear_vel = self.target_speed
        
        # CURVE DETECTION AND HANDLING
        if self.curve_detection:
            linear_vel, angular_vel = self.handle_curves(linear_vel, angular_vel, heading_error)
        
        # PREDICTIVE CONTROL
        if self.predictive_control:
            angular_vel = self.apply_predictive_control(angular_vel, lateral_error, heading_error)
        
        # Update previous errors
        self.lateral_error_previous = lateral_error
        self.heading_error_previous = heading_error
        
        return linear_vel, angular_vel
    
    def calculate_adaptive_speed(self, lateral_error, heading_error, angular_vel):
        """Calculate adaptive speed based on lane conditions"""
        # Base speed
        base_speed = self.target_speed
        
        # Reduce speed based on lateral error
        lateral_factor = max(0.5, 1.0 - abs(lateral_error) * 2.0)
        
        # Reduce speed based on heading error
        heading_factor = max(0.6, 1.0 - abs(heading_error) * 1.5)
        
        # Reduce speed based on steering intensity
        steering_factor = max(0.7, 1.0 - abs(angular_vel) * 0.5)
        
        # Combined adaptive speed
        adaptive_speed = base_speed * lateral_factor * heading_factor * steering_factor
        
        # Ensure minimum speed
        adaptive_speed = max(self.min_speed, adaptive_speed)
        
        return adaptive_speed
    
    def handle_curves(self, linear_vel, angular_vel, heading_error):
        """Detect and handle curves with appropriate speed reduction"""
        # Detect curve based on heading error magnitude
        is_curve = abs(heading_error) > self.curve_threshold
        
        if is_curve:
            # Reduce speed in curves
            linear_vel *= self.curve_speed_factor
            
            # Increase steering sensitivity in curves
            angular_vel *= 1.2
            
            rospy.loginfo_throttle(2, f"🌀 Curve detected! Reducing speed to {linear_vel:.2f}")
        
        return linear_vel, angular_vel
    
    def apply_predictive_control(self, angular_vel, lateral_error, heading_error):
        """Apply predictive control to anticipate lane changes"""
        # Predict future lateral error based on current heading
        predicted_lateral_error = lateral_error + heading_error * 0.5
        
        # Add predictive component to steering
        predictive_gain = 0.3
        predictive_control = -predictive_gain * predicted_lateral_error
        
        # Combine with current control
        enhanced_angular_vel = angular_vel + predictive_control
        
        return enhanced_angular_vel
    
    def apply_safety_limits(self, linear_vel, angular_vel):
        """Apply safety limits to control commands"""
        # Check for excessive lateral error
        if self.current_lane_pose and abs(self.current_lane_pose.x) > self.emergency_stop_threshold:
            rospy.logwarn(f"⚠️ Excessive lateral error: {self.current_lane_pose.x:.3f}")
            linear_vel = 0.0
            angular_vel = 0.0
        
        # Limit velocities
        linear_vel = np.clip(linear_vel, -self.max_speed, self.max_speed)
        angular_vel = np.clip(angular_vel, -2.0, 2.0)
        
        # Ensure forward motion only (safety)
        linear_vel = max(0.0, linear_vel)
        
        return linear_vel, angular_vel
    
    def apply_smoothing(self, linear_vel, angular_vel):
        """Apply smoothing to prevent jerky movements"""
        # Exponential smoothing
        smooth_linear = (self.smoothing_factor * self.prev_linear_vel + 
                        (1 - self.smoothing_factor) * linear_vel)
        smooth_angular = (self.smoothing_factor * self.prev_angular_vel + 
                         (1 - self.smoothing_factor) * angular_vel)
        
        # Update previous values
        self.prev_linear_vel = smooth_linear
        self.prev_angular_vel = smooth_angular
        
        return smooth_linear, smooth_angular
    
    def handle_lane_loss(self):
        """Handle lane loss with recovery behavior"""
        rospy.logwarn_throttle(2, "❌ Lane lost - executing recovery behavior")
        
        # Gradual stop
        recovery_speed = max(0.0, self.prev_linear_vel * 0.9)
        recovery_angular = self.prev_angular_vel * 0.8
        
        self.publish_velocity_commands(recovery_speed, recovery_angular)
        self.control_status_pub.publish(Bool(False))
    
    def track_performance(self, linear_vel, angular_vel):
        """Track control performance metrics"""
        if not self.current_lane_pose:
            return
        
        control_data = {
            'timestamp': rospy.Time.now().to_sec(),
            'linear_vel': linear_vel,
            'angular_vel': angular_vel,
            'lateral_error': self.current_lane_pose.x,
            'heading_error': self.current_lane_pose.y,
            'in_lane': self.current_lane_pose.z > 0.5
        }
        
        self.control_history.append(control_data)
        
        # Keep only last 100 measurements
        if len(self.control_history) > 100:
            self.control_history.pop(0)
        
        # Calculate and publish performance score
        if len(self.control_history) >= 10:
            performance_score = self.calculate_performance_score()
            self.performance_pub.publish(Float32(performance_score))
    
    def calculate_performance_score(self):
        """Calculate comprehensive performance score"""
        if not self.control_history:
            return 0.0
        
        recent_data = self.control_history[-20:]
        
        # Calculate metrics
        lateral_errors = [abs(d['lateral_error']) for d in recent_data]
        heading_errors = [abs(d['heading_error']) for d in recent_data]
        velocity_changes = [abs(recent_data[i]['linear_vel'] - recent_data[i-1]['linear_vel']) 
                           for i in range(1, len(recent_data))]
        in_lane_count = sum(1 for d in recent_data if d['in_lane'])
        
        # Scoring components
        lateral_score = max(0, 100 - np.mean(lateral_errors) * 200)
        heading_score = max(0, 100 - np.mean(heading_errors) * 150)
        smoothness_score = max(0, 100 - np.mean(velocity_changes) * 500)
        lane_score = (in_lane_count / len(recent_data)) * 100
        
        # Weighted overall score
        overall_score = (0.4 * lateral_score + 0.3 * heading_score + 
                        0.2 * smoothness_score + 0.1 * lane_score)
        
        return min(100.0, max(0.0, overall_score))
    
    def publish_velocity_commands(self, linear_vel, angular_vel):
        """Publish velocity commands to DuckieBot car_cmd_switch_node"""
        # DuckieBot Twist2DStamped message
        twist_msg = Twist2DStamped()
        twist_msg.header = Header()
        twist_msg.header.stamp = rospy.Time.now()
        twist_msg.v = linear_vel
        twist_msg.omega = angular_vel
        self.cmd_vel_pub.publish(twist_msg)
    
    def emergency_stop(self):
        """Emergency stop with immediate brake"""
        self.publish_velocity_commands(0.0, 0.0)
        self.reset_pid_state()
        self.control_status_pub.publish(Bool(False))

if __name__ == '__main__':
    try:
        controller = EnhancedLaneController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass