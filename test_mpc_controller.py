#!/usr/bin/env python3

"""
Test script to verify MPC controller is receiving data and sending commands
"""

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from duckietown_msgs.msg import Twist2DStamped

class MPCTester:
    def __init__(self):
        rospy.init_node('mpc_tester', anonymous=True)
        
        # Subscribe to what MPC controller subscribes to
        self.lane_pose_sub = rospy.Subscriber('/lane_follower/lane_pose', Point, self.lane_pose_callback)
        self.lane_found_sub = rospy.Subscriber('/lane_follower/lane_found', Bool, self.lane_found_callback)
        
        # Subscribe to what MPC controller publishes
        self.cmd_vel_sub = rospy.Subscriber('/blueduckie/car_cmd_switch_node/cmd', Twist2DStamped, self.cmd_vel_callback)
        
        self.lane_pose_count = 0
        self.lane_found_count = 0
        self.cmd_vel_count = 0
        
        rospy.loginfo("MPC Tester started - monitoring data flow")
        
    def lane_pose_callback(self, msg):
        self.lane_pose_count += 1
        if self.lane_pose_count % 10 == 0:
            rospy.loginfo(f"Lane pose received #{self.lane_pose_count}: x={msg.x:.3f}, y={msg.y:.3f}, z={msg.z:.3f}")
    
    def lane_found_callback(self, msg):
        self.lane_found_count += 1
        if self.lane_found_count % 10 == 0:
            rospy.loginfo(f"Lane found received #{self.lane_found_count}: {msg.data}")
    
    def cmd_vel_callback(self, msg):
        self.cmd_vel_count += 1
        rospy.loginfo(f"Control command #{self.cmd_vel_count}: v={msg.v:.3f}, omega={msg.omega:.3f}")

if __name__ == '__main__':
    try:
        tester = MPCTester()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass