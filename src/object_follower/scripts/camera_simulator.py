#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraSimulator:
    def __init__(self):
        rospy.init_node('camera_simulator', anonymous=True)
        
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)
        
        rospy.loginfo("Camera Simulator started - Publishing synthetic images with moving red object")
        
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_image)  # 10 Hz
        self.start_time = rospy.Time.now()
        
    def create_test_image(self):
        """Create a test image with a bright red object"""
        # Create a 640x480 image with green background
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :] = [40, 120, 40]  # Green background (B, G, R)
        
        # Calculate moving position for red object
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        center_x = int(320 + 150 * np.sin(elapsed * 0.5))  # Slower movement
        center_y = int(240 + 50 * np.cos(elapsed * 0.3))   # Vertical movement too
        
        # Ensure object stays in frame
        center_x = max(60, min(580, center_x))
        center_y = max(60, min(420, center_y))
        
        # Add a bright red circle
        cv2.circle(img, (center_x, center_y), 40, (0, 0, 255), -1)  # Bright red (B, G, R)
        
        # Add a smaller bright red center
        cv2.circle(img, (center_x, center_y), 20, (0, 0, 200), -1)
        
        # Add some random background objects (blue squares)
        for i in range(3):
            x = np.random.randint(50, 590)
            y = np.random.randint(50, 430)
            cv2.rectangle(img, (x, y), (x+30, y+30), (200, 50, 50), -1)  # Blue objects
        
        rospy.loginfo_throttle(2, f"Red object at position: ({center_x}, {center_y})")
        
        return img
    
    def publish_image(self, event):
        """Publish a test image"""
        img = self.create_test_image()
        
        try:
            image_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            image_msg.header.stamp = rospy.Time.now()
            image_msg.header.frame_id = "camera"
            self.image_pub.publish(image_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing image: {e}")

if __name__ == '__main__':
    try:
        simulator = CameraSimulator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass