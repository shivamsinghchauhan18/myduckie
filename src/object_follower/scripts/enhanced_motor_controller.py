#!/usr/bin/env python3

"""
Enhanced Motor Controller Node for DuckieBot - Phase 2
Full DuckieBot integration with advanced PID control and performance monitoring
"""

import rospy
import numpy as np
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Bool, Float32, Header
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped

class EnhancedMotorController:
    def __init__(self):
        rospy.init_node('enhanced_motor_controller', anonymous=True)
        
        # Publishers - Multiple message types for compatibility
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.wheels_pub = rospy.Publisher('/duckiebot_driver/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.duckiebot_vel_pub = rospy.Publisher('/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
        
        # Performance monitoring publishers
        self.performance_pub = rospy.Publisher('/object_follower/performance', Float32, queue_size=1)
        
        # Subscribers
        self.target_sub = rospy.Subscriber('/object_follower/target_position', Point, self.target_callback)
        self.target_found_sub = rospy.Subscriber('/object_follower/target_found', Bool, self.target_found_callback)
        self.distance_sub = rospy.Subscriber('/object_follower/target_distance', Float32, self.distance_callback)
        self.obstacle_sub = rospy.Subscriber('/object_follower/obstacle_detected', Bool, self.obstacle_callback)
        
        # Enhanced control parameters
        self.max_speed = rospy.get_param('~max_speed', 0.6)
        self.target_distance = rospy.get_param('~target_distance', 1.0)
        self.distance_tolerance = rospy.get_param('~distance_tolerance', 0.15)
        
        # Robot parameters
        self.wheel_distance = rospy.get_param('~wheel_distance', 0.1)  # meters between wheels
        
        # Enhanced PID parameters
        self.kp_lateral = rospy.get_param('~kp_lateral', 2.5)
        self.ki_lateral = rospy.get_param('~ki_lateral', 0.15)
        self.kd_lateral = rospy.get_param('~kd_lateral', 0.8)
        
        self.kp_distance = rospy.get_param('~kp_distance', 1.2)
        self.ki_distance = rospy.get_param('~ki_distance', 0.08)
        self.kd_distance = rospy.get_param('~kd_distance', 0.3)
        
        # State variables
        self.target_position = None
        self.target_found = False
        self.current_distance = 0.0
        self.obstacle_detected = False
        
        # Enhanced PID state
        self.lateral_error_integral = 0.0
        self.lateral_error_previous = 0.0
        self.distance_error_integral = 0.0
        self.distance_error_previous = 0.0
        
        # Performance tracking
        self.control_history = []
        self.start_time = rospy.Time.now()
        
        # Control loop timer
        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)  # 20 Hz
        
        rospy.loginfo("Enhanced Motor Controller with DuckieBot support initialized")
    
    def target_callback(self, msg):
        self.target_position = msg
    
    def target_found_callback(self, msg):
        self.target_found = msg.data
        if not self.target_found:
            self.reset_pid_state()
    
    def distance_callback(self, msg):
        self.current_distance = msg.data
    
    def obstacle_callback(self, msg):
        if msg.data and not self.obstacle_detected:
            rospy.logwarn("🚧 Obstacle detected! Emergency stop!")
        self.obstacle_detected = msg.data
    
    def reset_pid_state(self):
        """Reset PID integral terms when target is lost"""
        self.lateral_error_integral = 0.0
        self.distance_error_integral = 0.0
    
    def control_loop(self, event):
        """Enhanced control loop with performance monitoring"""
        if self.obstacle_detected:
            self.emergency_stop()
            return
        
        if not self.target_found or self.target_position is None:
            self.stop_robot()
            return
        
        # Calculate control commands
        linear_vel, angular_vel = self.calculate_enhanced_control_commands()
        
        # Apply velocity limits
        linear_vel = np.clip(linear_vel, -self.max_speed, self.max_speed)
        angular_vel = np.clip(angular_vel, -2.0, 2.0)
        
        # Publish commands to all topics
        self.publish_velocity_commands(linear_vel, angular_vel)
        
        # Track performance
        self.track_performance(linear_vel, angular_vel)
    
    def calculate_enhanced_control_commands(self):
        """Enhanced PID control with improved algorithms"""
        dt = 0.05  # 20 Hz control loop
        
        # Lateral control (steering)
        lateral_error = self.target_position.x  # -1 to 1, 0 is center
        self.lateral_error_integral += lateral_error * dt
        lateral_error_derivative = (lateral_error - self.lateral_error_previous) / dt
        
        # Anti-windup for integral term
        self.lateral_error_integral = np.clip(self.lateral_error_integral, -0.5, 0.5)
        
        angular_vel = -(self.kp_lateral * lateral_error + 
                       self.ki_lateral * self.lateral_error_integral + 
                       self.kd_lateral * lateral_error_derivative)
        
        self.lateral_error_previous = lateral_error
        
        # Distance control (forward/backward)
        distance_error = self.current_distance - self.target_distance
        self.distance_error_integral += distance_error * dt
        distance_error_derivative = (distance_error - self.distance_error_previous) / dt
        
        # Anti-windup
        self.distance_error_integral = np.clip(self.distance_error_integral, -1.0, 1.0)
        
        linear_vel = (self.kp_distance * distance_error + 
                     self.ki_distance * self.distance_error_integral + 
                     self.kd_distance * distance_error_derivative)
        
        self.distance_error_previous = distance_error
        
        # Adaptive speed based on lateral error (slow down when turning)
        speed_factor = 1.0 - 0.5 * abs(lateral_error)
        linear_vel *= speed_factor
        
        return linear_vel, angular_vel
    
    def track_performance(self, linear_vel, angular_vel):
        """Track control performance metrics"""
        control_data = {
            'timestamp': rospy.Time.now().to_sec(),
            'linear_vel': linear_vel,
            'angular_vel': angular_vel,
            'lateral_error': self.target_position.x if self.target_position else 0,
            'distance_error': abs(self.current_distance - self.target_distance)
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
        """Calculate performance score (0-100)"""
        if not self.control_history:
            return 0.0
        
        recent_data = self.control_history[-20:]  # Last 20 measurements
        
        # Calculate metrics
        lateral_errors = [abs(d['lateral_error']) for d in recent_data]
        distance_errors = [d['distance_error'] for d in recent_data]
        velocity_changes = [abs(recent_data[i]['linear_vel'] - recent_data[i-1]['linear_vel']) 
                           for i in range(1, len(recent_data))]
        
        # Score components (0-100 each)
        accuracy_score = max(0, 100 - np.mean(lateral_errors) * 200)  # Lateral accuracy
        distance_score = max(0, 100 - np.mean(distance_errors) * 100)  # Distance accuracy
        smoothness_score = max(0, 100 - np.mean(velocity_changes) * 500)  # Control smoothness
        
        # Weighted average
        overall_score = (0.4 * accuracy_score + 0.4 * distance_score + 0.2 * smoothness_score)
        
        return min(100.0, max(0.0, overall_score))
    
    def publish_velocity_commands(self, linear_vel, angular_vel):
        """Publish velocity commands to all supported topics"""
        current_time = rospy.Time.now()
        
        # 1. Standard ROS Twist message
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist_msg)
        
        # 2. DuckieBot Twist2DStamped message
        duckiebot_msg = Twist2DStamped()
        duckiebot_msg.header = Header()
        duckiebot_msg.header.stamp = current_time
        duckiebot_msg.header.frame_id = "base_link"
        duckiebot_msg.v = linear_vel
        duckiebot_msg.omega = angular_vel
        self.duckiebot_vel_pub.publish(duckiebot_msg)
        
        # 3. DuckieBot WheelsCmdStamped message
        # Convert linear and angular velocity to wheel velocities
        vel_left = linear_vel - angular_vel * self.wheel_distance / 2.0
        vel_right = linear_vel + angular_vel * self.wheel_distance / 2.0
        
        wheels_msg = WheelsCmdStamped()
        wheels_msg.header = Header()
        wheels_msg.header.stamp = current_time
        wheels_msg.header.frame_id = "base_link"
        wheels_msg.vel_left = vel_left
        wheels_msg.vel_right = vel_right
        self.wheels_pub.publish(wheels_msg)
    
    def stop_robot(self):
        """Stop the robot"""
        self.publish_velocity_commands(0.0, 0.0)
    
    def emergency_stop(self):
        """Emergency stop with immediate brake"""
        self.publish_velocity_commands(0.0, 0.0)
        self.reset_pid_state()

if __name__ == '__main__':
    try:
        controller = EnhancedMotorController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass