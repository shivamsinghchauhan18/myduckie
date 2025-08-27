#!/usr/bin/env python3

"""
Motor Controller Node for DuckieBot
Controls wheel motors based on target tracking and obstacle avoidance
"""

import rospy
import numpy as np
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Bool, Float32
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped

class MotorController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('motor_controller', anonymous=True)
        
        # Publishers
        self.wheels_pub = rospy.Publisher('/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.duckiebot_vel_pub = rospy.Publisher('/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
        
        # Subscribers
        self.target_sub = rospy.Subscriber('/object_follower/target_position', Point, self.target_callback)
        self.target_found_sub = rospy.Subscriber('/object_follower/target_found', Bool, self.target_found_callback)
        self.distance_sub = rospy.Subscriber('/object_follower/target_distance', Float32, self.distance_callback)
        self.obstacle_sub = rospy.Subscriber('/object_follower/obstacle_detected', Bool, self.obstacle_callback)
        
        # Control parameters
        self.max_speed = rospy.get_param('~max_speed', 0.5)  # m/s
        self.target_distance = rospy.get_param('~target_distance', 1.0)  # meters
        self.distance_tolerance = rospy.get_param('~distance_tolerance', 0.2)  # meters
        
        # PID parameters for lateral control
        self.kp_lateral = rospy.get_param('~kp_lateral', 2.0)
        self.ki_lateral = rospy.get_param('~ki_lateral', 0.1)
        self.kd_lateral = rospy.get_param('~kd_lateral', 0.5)
        
        # PID parameters for distance control
        self.kp_distance = rospy.get_param('~kp_distance', 1.0)
        self.ki_distance = rospy.get_param('~ki_distance', 0.05)
        self.kd_distance = rospy.get_param('~kd_distance', 0.2)
        
        # State variables
        self.target_position = None
        self.target_found = False
        self.current_distance = 0.0
        self.obstacle_detected = False
        
        # PID state variables
        self.lateral_error_integral = 0.0
        self.lateral_error_previous = 0.0
        self.distance_error_integral = 0.0
        self.distance_error_previous = 0.0
        
        # Control loop timer
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)  # 10 Hz
        
        rospy.loginfo("Motor Controller initialized")
    
    def target_callback(self, msg):
        self.target_position = msg
    
    def target_found_callback(self, msg):
        self.target_found = msg.data
    
    def distance_callback(self, msg):
        self.current_distance = msg.data
    
    def obstacle_callback(self, msg):
        self.obstacle_detected = msg.data
    
    def control_loop(self, event):
        """Main control loop executed at 10Hz"""
        if self.obstacle_detected:
            self.emergency_stop()
            return
        
        if not self.target_found or self.target_position is None:
            self.stop_robot()
            return
        
        # Calculate control commands
        linear_vel, angular_vel = self.calculate_control_commands()
        
        # Publish commands
        self.publish_velocity_commands(linear_vel, angular_vel)
    
    def calculate_control_commands(self):
        """Calculate linear and angular velocities using PID control"""
        dt = 0.1  # 10Hz control loop
        
        # Lateral control (steering)
        lateral_error = self.target_position.x  # -1 to 1, 0 is center
        self.lateral_error_integral += lateral_error * dt
        lateral_error_derivative = (lateral_error - self.lateral_error_previous) / dt
        
        angular_vel = -(self.kp_lateral * lateral_error + 
                       self.ki_lateral * self.lateral_error_integral + 
                       self.kd_lateral * lateral_error_derivative)
        
        # Clamp angular velocity
        angular_vel = np.clip(angular_vel, -2.0, 2.0)
        
        self.lateral_error_previous = lateral_error
        
        # Distance control (forward/backward)
        distance_error = self.current_distance - self.target_distance
        self.distance_error_integral += distance_error * dt
        distance_error_derivative = (distance_error - self.distance_error_previous) / dt
        
        linear_vel = -(self.kp_distance * distance_error + 
                      self.ki_distance * self.distance_error_integral + 
                      self.kd_distance * distance_error_derivative)
        
        # Clamp linear velocity
        linear_vel = np.clip(linear_vel, -self.max_speed, self.max_speed)
        
        # Reduce speed when turning
        speed_reduction = 1.0 - 0.5 * abs(angular_vel) / 2.0
        linear_vel *= speed_reduction
        
        self.distance_error_previous = distance_error
        
        return linear_vel, angular_vel
    
    def publish_velocity_commands(self, linear_vel, angular_vel):
        """Publish velocity commands to all relevant topics"""
        # Publish to cmd_vel (standard ROS)
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist)
        
        # Publish to DuckieBot car command topic
        duckiebot_cmd = Twist2DStamped()
        duckiebot_cmd.header.stamp = rospy.Time.now()
        duckiebot_cmd.v = linear_vel  # Forward velocity
        duckiebot_cmd.omega = angular_vel  # Angular velocity
        self.duckiebot_vel_pub.publish(duckiebot_cmd)
        
        # Publish to DuckieBot wheels topic
        wheels_cmd = WheelsCmdStamped()
        wheels_cmd.header.stamp = rospy.Time.now()
        
        # Convert twist to wheel velocities
        # DuckieBot specific parameters
        wheel_baseline = 0.1  # meters between wheels
        wheel_radius = 0.0318  # meters
        
        # Differential drive kinematics
        left_wheel_vel = (linear_vel - angular_vel * wheel_baseline / 2.0) / wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * wheel_baseline / 2.0) / wheel_radius
        
        wheels_cmd.vel_left = left_wheel_vel
        wheels_cmd.vel_right = right_wheel_vel
        
        self.wheels_pub.publish(wheels_cmd)
    
    def stop_robot(self):
        """Stop the robot"""
        self.publish_velocity_commands(0.0, 0.0)
        # Reset PID integrals
        self.lateral_error_integral = 0.0
        self.distance_error_integral = 0.0
    
    def emergency_stop(self):
        """Emergency stop when obstacle detected"""
        self.stop_robot()
        rospy.logwarn("Emergency stop: Obstacle detected!")

if __name__ == '__main__':
    try:
        controller = MotorController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass