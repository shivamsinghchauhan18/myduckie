#!/usr/bin/env python3

"""
Simple test script to verify ROS setup and node connectivity
"""

import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import Image

class SystemTester:
    def __init__(self):
        rospy.init_node('system_tester', anonymous=True)
        
        # Test publishers
        self.target_pub = rospy.Publisher('/object_follower/target_position', Point, queue_size=1)
        self.target_found_pub = rospy.Publisher('/object_follower/target_found', Bool, queue_size=1)
        
        # Test subscribers
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        
        self.received_cmd_vel = False
        
        rospy.loginfo("System Tester initialized")
    
    def cmd_vel_callback(self, msg):
        self.received_cmd_vel = True
        rospy.loginfo(f"Received cmd_vel: linear={msg.linear.x:.2f}, angular={msg.angular.z:.2f}")
    
    def test_motor_controller(self):
        """Test if motor controller responds to target data"""
        rospy.loginfo("Testing motor controller...")
        
        # Send a test target
        target = Point()
        target.x = 0.2  # Slightly to the right
        target.y = 0.0
        target.z = 1.5  # 1.5 meters away
        
        # Send target found signal
        self.target_found_pub.publish(Bool(True))
        rospy.sleep(0.1)
        
        # Send target position
        for i in range(10):
            self.target_pub.publish(target)
            rospy.sleep(0.1)
        
        if self.received_cmd_vel:
            rospy.loginfo("✓ Motor controller is working!")
            return True
        else:
            rospy.logwarn("✗ Motor controller not responding")
            return False
    
    def run_tests(self):
        rospy.sleep(2.0)  # Wait for nodes to start
        
        success = True
        
        # Test motor controller
        if not self.test_motor_controller():
            success = False
        
        if success:
            rospy.loginfo("🎉 All tests passed!")
        else:
            rospy.logwarn("❌ Some tests failed")
        
        return success

if __name__ == '__main__':
    try:
        tester = SystemTester()
        tester.run_tests()
    except rospy.ROSInterruptException:
        pass