#!/usr/bin/env python3

import rospy
import json
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from duckietown_msgs.msg import WheelsCmdStamped
import requests

class AvoidanceController:
    def __init__(self):
        rospy.init_node('avoidance_controller', anonymous=True)
        
        # Server configuration
        self.server_url = rospy.get_param('~server_url', 'http://localhost:5000')
        self.bot_id = rospy.get_param('~bot_id', 'duckiebot_01')
        
        # Control parameters
        self.max_linear_speed = 0.3
        self.max_angular_speed = 2.0
        self.safe_distance = 0.8
        self.avoidance_distance = 1.2
        
        # State variables
        self.current_mode = "NORMAL"  # NORMAL, AVOIDING, STOPPED, EMERGENCY
        self.avoidance_direction = None
        self.last_detection_time = rospy.Time.now()
        
        # Publishers
        self.cmd_pub = rospy.Publisher('/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.status_pub = rospy.Publisher('/yolo/avoidance_status', String, queue_size=10)
        
        # Subscribers
        self.closest_object_sub = rospy.Subscriber('/yolo/closest_object', String, self.closest_object_callback)
        self.safety_sub = rospy.Subscriber('/yolo/safety_status', String, self.safety_callback)
        self.emergency_sub = rospy.Subscriber('/yolo/emergency', Bool, self.emergency_callback)
        self.server_cmd_sub = rospy.Subscriber('/yolo/server_command', String, self.server_command_callback)
        
        # Timer for control loop
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("Avoidance Controller initialized")
    
    def closest_object_callback(self, msg):
        try:
            self.closest_object = json.loads(msg.data)
            self.last_detection_time = rospy.Time.now()
        except Exception as e:
            rospy.logerr(f"Error parsing closest object: {e}")
    
    def safety_callback(self, msg):
        self.safety_status = msg.data
    
    def emergency_callback(self, msg):
        if msg.data and self.current_mode != "EMERGENCY":
            self.current_mode = "EMERGENCY"
            self.send_emergency_stop()
            rospy.logwarn("Emergency stop activated!")
    
    def server_command_callback(self, msg):
        try:
            command = json.loads(msg.data)
            self.handle_server_command(command)
        except Exception as e:
            rospy.logerr(f"Error handling server command: {e}")
    
    def handle_server_command(self, command):
        """Handle commands from server"""
        cmd_type = command.get('type')
        
        if cmd_type == 'stop':
            self.current_mode = "STOPPED"
        elif cmd_type == 'resume':
            self.current_mode = "NORMAL"
        elif cmd_type == 'avoid_left':
            self.current_mode = "AVOIDING"
            self.avoidance_direction = "LEFT"
        elif cmd_type == 'avoid_right':
            self.current_mode = "AVOIDING"
            self.avoidance_direction = "RIGHT"
        elif cmd_type == 'emergency_stop':
            self.current_mode = "EMERGENCY"
            self.send_emergency_stop()
        
        rospy.loginfo(f"Executed server command: {cmd_type}")
    
    def control_loop(self, event):
        """Main control loop"""
        try:
            # Check if we have recent detection data
            time_since_detection = (rospy.Time.now() - self.last_detection_time).to_sec()
            
            if time_since_detection > 2.0:  # No recent detections
                if self.current_mode == "AVOIDING":
                    self.current_mode = "NORMAL"
                return
            
            # Get control command based on current mode
            cmd = self.generate_control_command()
            
            # Publish command
            if cmd:
                self.publish_wheels_cmd(cmd['linear'], cmd['angular'])
            
            # Publish status
            status = {
                'mode': self.current_mode,
                'avoidance_direction': self.avoidance_direction,
                'safety_status': getattr(self, 'safety_status', 'UNKNOWN')
            }
            self.status_pub.publish(json.dumps(status))
            
            # Send status to server
            self.send_status_to_server(status)
            
        except Exception as e:
            rospy.logerr(f"Error in control loop: {e}")
    
    def generate_control_command(self):
        """Generate control command based on current situation"""
        if not hasattr(self, 'closest_object'):
            return {'linear': 0.0, 'angular': 0.0}
        
        distance = self.closest_object.get('final_distance')
        threat_level = self.closest_object.get('threat_level', 'LOW')
        object_position = self.closest_object.get('position', 320)  # Center of 640px image
        
        if self.current_mode == "EMERGENCY":
            return {'linear': 0.0, 'angular': 0.0}
        
        elif self.current_mode == "STOPPED":
            return {'linear': 0.0, 'angular': 0.0}
        
        elif self.current_mode == "AVOIDING":
            return self.generate_avoidance_command()
        
        elif distance and distance < self.safe_distance:
            # Determine avoidance strategy
            self.current_mode = "AVOIDING"
            
            # Choose avoidance direction based on object position
            image_center = 320  # Assuming 640px width
            if object_position < image_center - 50:
                self.avoidance_direction = "RIGHT"
            elif object_position > image_center + 50:
                self.avoidance_direction = "LEFT"
            else:
                # Object in center, choose based on threat level
                if threat_level in ["CRITICAL", "HIGH"]:
                    self.avoidance_direction = "RIGHT"  # Default to right
                else:
                    # Slow down and assess
                    return {'linear': 0.1, 'angular': 0.0}
            
            return self.generate_avoidance_command()
        
        else:
            # Normal operation - continue forward
            self.current_mode = "NORMAL"
            return {'linear': self.max_linear_speed * 0.7, 'angular': 0.0}
    
    def generate_avoidance_command(self):
        """Generate avoidance maneuver command"""
        if not hasattr(self, 'closest_object'):
            return {'linear': 0.0, 'angular': 0.0}
        
        distance = self.closest_object.get('final_distance', 0)
        
        if distance < 0.3:  # Too close, stop and turn
            linear_speed = 0.0
            angular_speed = self.max_angular_speed * 0.8
        elif distance < 0.6:  # Close, slow turn
            linear_speed = 0.1
            angular_speed = self.max_angular_speed * 0.6
        else:  # Moderate distance, gentle avoidance
            linear_speed = 0.2
            angular_speed = self.max_angular_speed * 0.4
        
        # Apply direction
        if self.avoidance_direction == "LEFT":
            angular_speed = angular_speed
        else:  # RIGHT
            angular_speed = -angular_speed
        
        return {'linear': linear_speed, 'angular': angular_speed}
    
    def publish_wheels_cmd(self, linear, angular):
        """Convert twist to wheel commands and publish"""
        # Convert linear and angular velocities to wheel speeds
        # Assuming differential drive kinematics
        wheel_base = 0.1  # meters, adjust based on your robot
        
        left_wheel = linear - (angular * wheel_base / 2.0)
        right_wheel = linear + (angular * wheel_base / 2.0)
        
        # Create wheel command message
        wheels_cmd = WheelsCmdStamped()
        wheels_cmd.header.stamp = rospy.Time.now()
        wheels_cmd.vel_left = left_wheel
        wheels_cmd.vel_right = right_wheel
        
        self.cmd_pub.publish(wheels_cmd)
    
    def send_emergency_stop(self):
        """Send emergency stop command"""
        self.publish_wheels_cmd(0.0, 0.0)
        
        # Notify server of emergency
        try:
            data = {
                'bot_id': self.bot_id,
                'event': 'emergency_stop',
                'timestamp': rospy.Time.now().to_sec()
            }
            requests.post(f"{self.server_url}/api/emergency", json=data, timeout=1.0)
        except Exception as e:
            rospy.logwarn(f"Failed to notify server of emergency: {e}")
    
    def send_status_to_server(self, status):
        """Send current status to server"""
        try:
            data = {
                'bot_id': self.bot_id,
                'status': status,
                'timestamp': rospy.Time.now().to_sec()
            }
            requests.post(f"{self.server_url}/api/status_update", json=data, timeout=1.0)
        except Exception:
            pass  # Fail silently to avoid spam

if __name__ == '__main__':
    try:
        controller = AvoidanceController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass