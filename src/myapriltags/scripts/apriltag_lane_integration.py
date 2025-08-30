#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool
from duckietown_msgs.msg import WheelsCmd, Twist2DStamped
from geometry_msgs.msg import Twist

class AprilTagLaneIntegration:
    def __init__(self):
        rospy.init_node('apriltag_lane_integration', anonymous=True)
        
        # State variables
        self.lane_following_enabled = True
        self.apriltag_override = False
        
        # Publishers
        self.wheels_cmd_pub = rospy.Publisher('/wheels_cmd', WheelsCmd, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Subscribers
        self.lane_following_enable_sub = rospy.Subscriber('/lane_following_enable', Bool, self.lane_enable_callback)
        self.lane_wheels_sub = rospy.Subscriber('/lane_following/wheels_cmd', WheelsCmd, self.lane_wheels_callback)
        self.lane_cmd_vel_sub = rospy.Subscriber('/lane_following/cmd_vel', Twist, self.lane_cmd_vel_callback)
        self.apriltag_stop_sub = rospy.Subscriber('/apriltag_stop_signal', Bool, self.apriltag_stop_callback)
        
        rospy.loginfo("AprilTag Lane Integration initialized")
    
    def lane_enable_callback(self, msg):
        self.lane_following_enabled = msg.data
        rospy.loginfo(f"Lane following enabled: {self.lane_following_enabled}")
    
    def apriltag_stop_callback(self, msg):
        self.apriltag_override = msg.data
        if self.apriltag_override:
            # Send stop command immediately
            self.send_stop_command()
    
    def lane_wheels_callback(self, msg):
        # Only forward lane following commands if enabled and not overridden by AprilTag
        if self.lane_following_enabled and not self.apriltag_override:
            self.wheels_cmd_pub.publish(msg)
    
    def lane_cmd_vel_callback(self, msg):
        # Only forward lane following commands if enabled and not overridden by AprilTag
        if self.lane_following_enabled and not self.apriltag_override:
            self.cmd_vel_pub.publish(msg)
    
    def send_stop_command(self):
        # Send stop command
        wheels_msg = WheelsCmd()
        wheels_msg.vel_left = 0.0
        wheels_msg.vel_right = 0.0
        self.wheels_cmd_pub.publish(wheels_msg)
        
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(twist_msg)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        integration = AprilTagLaneIntegration()
        integration.run()
    except rospy.ROSInterruptException:
        pass