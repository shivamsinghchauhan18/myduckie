#!/usr/bin/env python3

import rospy
import time
from std_msgs.msg import Bool
from duckietown_msgs.msg import WheelsCmd, Twist2DStamped
from geometry_msgs.msg import Twist

class AprilTagController:
    def __init__(self):
        rospy.init_node('apriltag_controller', anonymous=True)
        
        # State variables
        self.is_stopping = False
        self.stop_start_time = None
        self.stop_duration = rospy.get_param('~stop_duration', 2.0)  # seconds
        
        # Publishers
        self.wheels_pub = rospy.Publisher('/wheels_cmd', WheelsCmd, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.lane_following_enable_pub = rospy.Publisher('/lane_following_enable', Bool, queue_size=1)
        
        # Subscribers
        self.stop_signal_sub = rospy.Subscriber('/apriltag_stop_signal', Bool, self.stop_signal_callback)
        
        rospy.loginfo("AprilTag Controller initialized")
    
    def stop_signal_callback(self, msg):
        if msg.data and not self.is_stopping:
            # Start stopping sequence
            self.start_stop_sequence()
        elif not msg.data and self.is_stopping:
            # Check if stop duration has elapsed
            if self.stop_start_time and (time.time() - self.stop_start_time) >= self.stop_duration:
                self.end_stop_sequence()
    
    def start_stop_sequence(self):
        rospy.loginfo("Starting AprilTag stop sequence")
        self.is_stopping = True
        self.stop_start_time = time.time()
        
        # Disable lane following
        enable_msg = Bool()
        enable_msg.data = False
        self.lane_following_enable_pub.publish(enable_msg)
        
        # Stop the robot
        self.send_stop_command()
    
    def end_stop_sequence(self):
        rospy.loginfo("Ending AprilTag stop sequence - resuming lane following")
        self.is_stopping = False
        self.stop_start_time = None
        
        # Re-enable lane following
        enable_msg = Bool()
        enable_msg.data = True
        self.lane_following_enable_pub.publish(enable_msg)
    
    def send_stop_command(self):
        # Send stop command via wheels
        wheels_msg = WheelsCmd()
        wheels_msg.vel_left = 0.0
        wheels_msg.vel_right = 0.0
        self.wheels_pub.publish(wheels_msg)
        
        # Send stop command via cmd_vel
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(twist_msg)
    
    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            # Check if we need to end stop sequence based on time
            if self.is_stopping and self.stop_start_time:
                elapsed_time = time.time() - self.stop_start_time
                if elapsed_time >= self.stop_duration:
                    self.end_stop_sequence()
                else:
                    # Continue sending stop commands
                    self.send_stop_command()
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = AprilTagController()
        controller.run()
    except rospy.ROSInterruptException:
        pass