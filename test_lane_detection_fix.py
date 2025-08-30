#!/usr/bin/env python3

"""
Quick test and fix for lane detection issues
"""

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

def test_lane_detection():
    """Test lane detection by publishing known good values"""
    rospy.init_node('lane_detection_test', anonymous=True)
    
    # Publishers
    lane_pose_pub = rospy.Publisher('/lane_follower/lane_pose', Point, queue_size=1)
    lane_confidence_pub = rospy.Publisher('/lane_follower/lane_confidence', Float32, queue_size=1)
    
    rate = rospy.Rate(10)  # 10 Hz
    
    rospy.loginfo("Publishing test lane poses...")
    
    while not rospy.is_shutdown():
        # Publish a centered lane pose
        test_pose = Point()
        test_pose.x = 0.0  # Centered
        test_pose.y = 0.0  # No heading error
        test_pose.z = 1.0  # Lane found
        
        lane_pose_pub.publish(test_pose)
        lane_confidence_pub.publish(Float32(0.9))
        
        rate.sleep()

if __name__ == '__main__':
    try:
        test_lane_detection()
    except rospy.ROSInterruptException:
        pass