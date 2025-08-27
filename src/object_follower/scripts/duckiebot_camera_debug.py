#!/usr/bin/env python3

"""
DuckieBot Camera Debug Tool
Diagnose camera and image detection issues
"""

import rospy
import subprocess
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Point

class DuckiebotCameraDebugger:
    def __init__(self):
        rospy.init_node('duckiebot_camera_debugger', anonymous=True)
        
        # Counters
        self.compressed_image_count = 0
        self.raw_image_count = 0
        self.target_detection_count = 0
        
        # Subscribers for all possible camera topics
        rospy.Subscriber('/camera_node/image/compressed', CompressedImage, self.compressed_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.raw_callback)
        rospy.Subscriber('/object_follower/target_found', Bool, self.target_callback)
        rospy.Subscriber('/object_follower/target_position', Point, self.position_callback)
        
        # Status timer
        rospy.Timer(rospy.Duration(3.0), self.print_status)
        
        rospy.loginfo("🔍 DuckieBot Camera Debugger started")
        rospy.loginfo("📊 Will report camera and detection status every 3 seconds")
        
    def compressed_callback(self, msg):
        self.compressed_image_count += 1
        rospy.loginfo_throttle(5, f"📷 Compressed image received: {len(msg.data)} bytes")
    
    def raw_callback(self, msg):
        self.raw_image_count += 1
        rospy.loginfo_throttle(5, f"🖼️ Raw image received: {msg.width}x{msg.height}")
    
    def target_callback(self, msg):
        if msg.data:
            self.target_detection_count += 1
            rospy.loginfo("🎯 TARGET DETECTED!")
    
    def position_callback(self, msg):
        rospy.loginfo(f"📍 Target position: ({msg.x:.2f}, {msg.y:.2f})")
    
    def print_status(self, event):
        rospy.loginfo("=" * 50)
        rospy.loginfo("🔍 DUCKIEBOT DEBUG STATUS")
        rospy.loginfo(f"📷 Compressed images: {self.compressed_image_count}")
        rospy.loginfo(f"🖼️ Raw images: {self.raw_image_count}")
        rospy.loginfo(f"🎯 Target detections: {self.target_detection_count}")
        
        # Check topic availability
        try:
            result = subprocess.run(['rostopic', 'list'], capture_output=True, text=True, timeout=3)
            topics = result.stdout.strip().split('\n')
            
            camera_topics = [t for t in topics if 'camera' in t]
            rospy.loginfo(f"📡 Camera topics: {camera_topics}")
            
            object_topics = [t for t in topics if 'object_follower' in t]
            rospy.loginfo(f"🎯 Object follower topics: {object_topics}")
            
        except:
            rospy.logwarn("⚠️ Could not check topic list")
        
        rospy.loginfo("=" * 50)

if __name__ == '__main__':
    try:
        debugger = DuckiebotCameraDebugger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass