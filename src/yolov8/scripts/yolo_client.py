#!/usr/bin/env python3

import rospy
import json
import requests
import time
from std_msgs.msg import String, Bool
from threading import Thread, Lock

class YOLOClient:
    def __init__(self):
        rospy.init_node('yolo_client', anonymous=True)
        
        # Server configuration
        self.server_url = rospy.get_param('~server_url', 'http://localhost:5000')
        self.bot_id = rospy.get_param('~bot_id', 'duckiebot_01')
        self.api_key = rospy.get_param('~api_key', 'default_key')
        
        # Connection parameters
        self.heartbeat_interval = 5.0  # seconds
        self.command_poll_interval = 1.0  # seconds
        self.max_retries = 3
        
        # State variables
        self.connected = False
        self.last_heartbeat = time.time()
        self.command_lock = Lock()
        
        # Publishers for server commands
        self.server_cmd_pub = rospy.Publisher('/yolo/server_command', String, queue_size=10)
        self.connection_status_pub = rospy.Publisher('/yolo/connection_status', Bool, queue_size=10)
        
        # Start background threads
        self.heartbeat_thread = Thread(target=self.heartbeat_loop, daemon=True)
        self.command_thread = Thread(target=self.command_polling_loop, daemon=True)
        
        self.heartbeat_thread.start()
        self.command_thread.start()
        
        # Register with server
        self.register_bot()
        
        rospy.loginfo("YOLO Client initialized")
    
    def register_bot(self):
        """Register this bot with the server"""
        try:
            registration_data = {
                'bot_id': self.bot_id,
                'api_key': self.api_key,
                'capabilities': [
                    'object_detection',
                    'distance_calculation',
                    'intelligent_avoidance',
                    'emergency_recovery'
                ],
                'status': 'online',
                'timestamp': time.time()
            }
            
            response = requests.post(
                f"{self.server_url}/api/register",
                json=registration_data,
                timeout=5.0
            )
            
            if response.status_code == 200:
                self.connected = True
                rospy.loginfo("Successfully registered with server")
                self.connection_status_pub.publish(True)
            else:
                rospy.logwarn(f"Registration failed: {response.status_code}")
                self.connected = False
                self.connection_status_pub.publish(False)
                
        except requests.exceptions.RequestException as e:
            rospy.logwarn(f"Failed to register with server: {e}")
            self.connected = False
            self.connection_status_pub.publish(False)
    
    def heartbeat_loop(self):
        """Send periodic heartbeat to server"""
        while not rospy.is_shutdown():
            try:
                if time.time() - self.last_heartbeat >= self.heartbeat_interval:
                    self.send_heartbeat()
                    self.last_heartbeat = time.time()
                
                time.sleep(1.0)
                
            except Exception as e:
                rospy.logwarn(f"Error in heartbeat loop: {e}")
                time.sleep(5.0)
    
    def send_heartbeat(self):
        """Send heartbeat to server"""
        try:
            heartbeat_data = {
                'bot_id': self.bot_id,
                'status': 'alive',
                'timestamp': time.time()
            }
            
            response = requests.post(
                f"{self.server_url}/api/heartbeat",
                json=heartbeat_data,
                timeout=3.0
            )
            
            if response.status_code == 200:
                if not self.connected:
                    self.connected = True
                    self.connection_status_pub.publish(True)
                    rospy.loginfo("Reconnected to server")
            else:
                if self.connected:
                    self.connected = False
                    self.connection_status_pub.publish(False)
                    rospy.logwarn("Lost connection to server")
                    
        except requests.exceptions.RequestException as e:
            if self.connected:
                self.connected = False
                self.connection_status_pub.publish(False)
                rospy.logwarn(f"Heartbeat failed: {e}")
    
    def command_polling_loop(self):
        """Poll server for commands"""
        while not rospy.is_shutdown():
            try:
                if self.connected:
                    self.poll_for_commands()
                
                time.sleep(self.command_poll_interval)
                
            except Exception as e:
                rospy.logwarn(f"Error in command polling: {e}")
                time.sleep(5.0)
    
    def poll_for_commands(self):
        """Poll server for pending commands"""
        try:
            response = requests.get(
                f"{self.server_url}/api/commands/{self.bot_id}",
                timeout=2.0
            )
            
            if response.status_code == 200:
                commands = response.json()
                
                with self.command_lock:
                    for command in commands.get('commands', []):
                        self.process_server_command(command)
                        
                        # Acknowledge command execution
                        self.acknowledge_command(command.get('id'))
            
        except requests.exceptions.RequestException as e:
            rospy.logdebug(f"Command polling failed: {e}")
    
    def process_server_command(self, command):
        """Process a command received from server"""
        try:
            command_type = command.get('type')
            rospy.loginfo(f"Processing server command: {command_type}")
            
            # Publish command to appropriate topic
            self.server_cmd_pub.publish(json.dumps(command))
            
            # Handle specific command types
            if command_type == 'emergency_stop':
                rospy.logwarn("Emergency stop command received from server")
            elif command_type == 'resume_operation':
                rospy.loginfo("Resume operation command received from server")
            elif command_type == 'change_mode':
                mode = command.get('mode', 'normal')
                rospy.loginfo(f"Mode change command: {mode}")
            elif command_type == 'update_parameters':
                self.update_parameters(command.get('parameters', {}))
            
        except Exception as e:
            rospy.logerr(f"Error processing server command: {e}")
    
    def update_parameters(self, parameters):
        """Update system parameters from server"""
        try:
            for param_name, param_value in parameters.items():
                rospy.set_param(param_name, param_value)
                rospy.loginfo(f"Updated parameter {param_name} = {param_value}")
        except Exception as e:
            rospy.logerr(f"Error updating parameters: {e}")
    
    def acknowledge_command(self, command_id):
        """Acknowledge command execution to server"""
        try:
            ack_data = {
                'bot_id': self.bot_id,
                'command_id': command_id,
                'status': 'executed',
                'timestamp': time.time()
            }
            
            requests.post(
                f"{self.server_url}/api/command_ack",
                json=ack_data,
                timeout=2.0
            )
            
        except requests.exceptions.RequestException as e:
            rospy.logwarn(f"Failed to acknowledge command: {e}")
    
    def send_status_update(self, status_data):
        """Send status update to server"""
        try:
            if not self.connected:
                return False
            
            data = {
                'bot_id': self.bot_id,
                'timestamp': time.time(),
                **status_data
            }
            
            response = requests.post(
                f"{self.server_url}/api/status",
                json=data,
                timeout=2.0
            )
            
            return response.status_code == 200
            
        except requests.exceptions.RequestException as e:
            rospy.logdebug(f"Status update failed: {e}")
            return False
    
    def send_emergency_alert(self, alert_data):
        """Send emergency alert to server"""
        try:
            data = {
                'bot_id': self.bot_id,
                'alert_type': 'emergency',
                'timestamp': time.time(),
                **alert_data
            }
            
            # Try multiple times for emergency alerts
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        f"{self.server_url}/api/emergency_alert",
                        json=data,
                        timeout=3.0
                    )
                    
                    if response.status_code == 200:
                        rospy.loginfo("Emergency alert sent to server")
                        return True
                        
                except requests.exceptions.RequestException:
                    if attempt < self.max_retries - 1:
                        time.sleep(1.0)
                        continue
            
            rospy.logerr("Failed to send emergency alert after retries")
            return False
            
        except Exception as e:
            rospy.logerr(f"Error sending emergency alert: {e}")
            return False

if __name__ == '__main__':
    try:
        client = YOLOClient()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass