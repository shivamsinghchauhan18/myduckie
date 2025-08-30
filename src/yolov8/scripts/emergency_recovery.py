#!/usr/bin/env python3

import rospy
import json
import time
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from duckietown_msgs.msg import WheelsCmdStamped
import requests

class EmergencyRecovery:
    def __init__(self):
        rospy.init_node('emergency_recovery', anonymous=True)
        
        # Server configuration
        self.server_url = rospy.get_param('~server_url', 'http://localhost:5000')
        self.bot_id = rospy.get_param('~bot_id', 'duckiebot_01')
        
        # Recovery parameters
        self.recovery_timeout = 10.0  # seconds
        self.backup_distance = 0.3    # meters to backup
        self.scan_duration = 3.0      # seconds to scan for clear path
        
        # State variables
        self.in_emergency = False
        self.recovery_stage = "IDLE"  # IDLE, STOPPING, BACKING, SCANNING, TURNING, RESUMING
        self.recovery_start_time = None
        self.clear_direction = None
        
        # Publishers
        self.cmd_pub = rospy.Publisher('/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.recovery_status_pub = rospy.Publisher('/yolo/recovery_status', String, queue_size=10)
        self.override_pub = rospy.Publisher('/yolo/recovery_override', Bool, queue_size=10)
        
        # Subscribers
        self.emergency_sub = rospy.Subscriber('/yolo/emergency', Bool, self.emergency_callback)
        self.safety_sub = rospy.Subscriber('/yolo/safety_status', String, self.safety_callback)
        self.closest_object_sub = rospy.Subscriber('/yolo/closest_object', String, self.closest_object_callback)
        
        # Recovery timer
        self.recovery_timer = rospy.Timer(rospy.Duration(0.2), self.recovery_loop)
        
        rospy.loginfo("Emergency Recovery System initialized")
    
    def emergency_callback(self, msg):
        if msg.data and not self.in_emergency:
            self.start_emergency_recovery()
        elif not msg.data and self.in_emergency:
            self.end_emergency_recovery()
    
    def safety_callback(self, msg):
        self.safety_status = msg.data
    
    def closest_object_callback(self, msg):
        try:
            self.closest_object = json.loads(msg.data)
        except Exception as e:
            rospy.logerr(f"Error parsing closest object in recovery: {e}")
    
    def start_emergency_recovery(self):
        """Initiate emergency recovery sequence"""
        self.in_emergency = True
        self.recovery_stage = "STOPPING"
        self.recovery_start_time = time.time()
        self.override_pub.publish(True)  # Override other controllers
        
        rospy.logwarn("Starting emergency recovery sequence")
        self.notify_server_emergency_start()
    
    def end_emergency_recovery(self):
        """End emergency recovery and return to normal operation"""
        self.in_emergency = False
        self.recovery_stage = "IDLE"
        self.recovery_start_time = None
        self.clear_direction = None
        self.override_pub.publish(False)  # Release override
        
        rospy.loginfo("Emergency recovery completed")
        self.notify_server_recovery_complete()
    
    def recovery_loop(self, event):
        """Main recovery control loop"""
        if not self.in_emergency:
            return
        
        try:
            # Check for recovery timeout
            if time.time() - self.recovery_start_time > self.recovery_timeout:
                rospy.logwarn("Recovery timeout reached, requesting human intervention")
                self.request_human_intervention()
                return
            
            # Execute recovery stage
            if self.recovery_stage == "STOPPING":
                self.execute_stop()
            elif self.recovery_stage == "BACKING":
                self.execute_backup()
            elif self.recovery_stage == "SCANNING":
                self.execute_scan()
            elif self.recovery_stage == "TURNING":
                self.execute_turn()
            elif self.recovery_stage == "RESUMING":
                self.execute_resume()
            
            # Publish recovery status
            status = {
                'stage': self.recovery_stage,
                'time_elapsed': time.time() - self.recovery_start_time,
                'clear_direction': self.clear_direction
            }
            self.recovery_status_pub.publish(json.dumps(status))
            
        except Exception as e:
            rospy.logerr(f"Error in recovery loop: {e}")
    
    def execute_stop(self):
        """Stop the robot immediately"""
        self.publish_wheels_cmd(0.0, 0.0)
        
        # Wait for 1 second to ensure complete stop
        if time.time() - self.recovery_start_time > 1.0:
            self.recovery_stage = "BACKING"
            rospy.loginfo("Recovery: Stopping complete, starting backup")
    
    def execute_backup(self):
        """Backup slowly to create space"""
        # Check if path behind is clear (simplified check)
        if hasattr(self, 'closest_object'):
            distance = self.closest_object.get('final_distance', 0)
            if distance > 0.5:  # Sufficient space to backup
                self.publish_wheels_cmd(-0.1, 0.0)  # Slow backward
            else:
                # Can't backup, go to scanning
                self.recovery_stage = "SCANNING"
                rospy.loginfo("Recovery: Cannot backup, starting scan")
                return
        
        # Backup for a short duration
        if time.time() - self.recovery_start_time > 3.0:
            self.recovery_stage = "SCANNING"
            rospy.loginfo("Recovery: Backup complete, starting scan")
    
    def execute_scan(self):
        """Scan left and right to find clear path"""
        scan_time = time.time() - self.recovery_start_time - 3.0  # Subtract previous stages
        
        if scan_time < self.scan_duration / 2:
            # Scan left
            self.publish_wheels_cmd(0.0, 0.5)
            self.check_clear_path("LEFT")
        elif scan_time < self.scan_duration:
            # Scan right
            self.publish_wheels_cmd(0.0, -0.5)
            self.check_clear_path("RIGHT")
        else:
            # Scanning complete
            self.publish_wheels_cmd(0.0, 0.0)
            if self.clear_direction:
                self.recovery_stage = "TURNING"
                rospy.loginfo(f"Recovery: Scan complete, turning {self.clear_direction}")
            else:
                rospy.logwarn("Recovery: No clear path found, requesting help")
                self.request_human_intervention()
    
    def check_clear_path(self, direction):
        """Check if current direction has a clear path"""
        if hasattr(self, 'closest_object'):
            distance = self.closest_object.get('final_distance', 0)
            if distance > 1.5:  # Clear path threshold
                self.clear_direction = direction
    
    def execute_turn(self):
        """Turn towards the clear direction"""
        if self.clear_direction == "LEFT":
            self.publish_wheels_cmd(0.0, 1.0)
        else:  # RIGHT
            self.publish_wheels_cmd(0.0, -1.0)
        
        # Turn for 2 seconds
        turn_time = time.time() - self.recovery_start_time - 6.0  # Subtract previous stages
        if turn_time > 2.0:
            self.recovery_stage = "RESUMING"
            rospy.loginfo("Recovery: Turn complete, resuming")
    
    def execute_resume(self):
        """Resume normal operation"""
        # Check if path ahead is clear
        if hasattr(self, 'closest_object'):
            distance = self.closest_object.get('final_distance', 0)
            if distance > 1.0:  # Safe to resume
                self.end_emergency_recovery()
                return
        
        # Move forward slowly
        self.publish_wheels_cmd(0.1, 0.0)
    
    def publish_wheels_cmd(self, linear, angular):
        """Publish wheel command"""
        # Convert to wheel speeds
        wheel_base = 0.1
        left_wheel = linear - (angular * wheel_base / 2.0)
        right_wheel = linear + (angular * wheel_base / 2.0)
        
        wheels_cmd = WheelsCmdStamped()
        wheels_cmd.header.stamp = rospy.Time.now()
        wheels_cmd.vel_left = left_wheel
        wheels_cmd.vel_right = right_wheel
        
        self.cmd_pub.publish(wheels_cmd)
    
    def request_human_intervention(self):
        """Request human intervention when recovery fails"""
        self.publish_wheels_cmd(0.0, 0.0)  # Stop completely
        
        try:
            data = {
                'bot_id': self.bot_id,
                'event': 'human_intervention_required',
                'reason': 'Emergency recovery failed',
                'timestamp': time.time(),
                'location': 'unknown'  # Could be enhanced with localization
            }
            requests.post(f"{self.server_url}/api/intervention", json=data, timeout=2.0)
            rospy.logwarn("Human intervention requested via server")
        except Exception as e:
            rospy.logerr(f"Failed to request human intervention: {e}")
        
        # Keep the robot stopped
        self.recovery_stage = "IDLE"
        self.in_emergency = False
    
    def notify_server_emergency_start(self):
        """Notify server that emergency recovery has started"""
        try:
            data = {
                'bot_id': self.bot_id,
                'event': 'emergency_recovery_start',
                'timestamp': time.time()
            }
            requests.post(f"{self.server_url}/api/emergency_recovery", json=data, timeout=1.0)
        except Exception as e:
            rospy.logwarn(f"Failed to notify server of recovery start: {e}")
    
    def notify_server_recovery_complete(self):
        """Notify server that recovery is complete"""
        try:
            data = {
                'bot_id': self.bot_id,
                'event': 'emergency_recovery_complete',
                'timestamp': time.time()
            }
            requests.post(f"{self.server_url}/api/emergency_recovery", json=data, timeout=1.0)
        except Exception as e:
            rospy.logwarn(f"Failed to notify server of recovery completion: {e}")

if __name__ == '__main__':
    try:
        recovery = EmergencyRecovery()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass