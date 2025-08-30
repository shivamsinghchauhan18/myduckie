#!/usr/bin/env python3

"""
Lane System Monitor for Advanced Lane Following System
Real-time monitoring and status reporting
"""

import rospy
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Bool, Float32, String
# Using standard ROS messages

class LaneSystemMonitor:
    def __init__(self):
        rospy.init_node('lane_system_monitor', anonymous=True)
        
        # Subscribers
        rospy.Subscriber('/lane_follower/lane_pose', Point, self.lane_pose_callback)
        rospy.Subscriber('/lane_follower/lane_found', Bool, self.lane_found_callback)
        rospy.Subscriber('/lane_follower/lane_center', Point, self.lane_center_callback)
        rospy.Subscriber('/lane_follower/obstacle_detected', Bool, self.obstacle_callback)
        rospy.Subscriber('/lane_follower/control_status', Bool, self.control_status_callback)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        
        # System state variables
        self.lane_found = False
        self.lane_pose = None
        self.lane_center = None
        self.obstacle_detected = False
        self.control_active = False
        self.current_velocity = None
        
        # Monitoring parameters
        self.monitor_interval = rospy.get_param('~monitor_interval', 2.0)
        
        # Status tracking
        self.last_lane_found_time = None
        self.last_control_time = None
        self.system_start_time = rospy.Time.now()
        
        # Performance counters
        self.lane_detection_count = 0
        self.control_command_count = 0
        self.obstacle_detection_count = 0
        
        rospy.loginfo("Lane System Monitor started - Monitoring all lane following components")
        
        # Status reporting timer
        self.status_timer = rospy.Timer(rospy.Duration(self.monitor_interval), self.report_system_status)
    
    def lane_pose_callback(self, msg):
        self.lane_pose = msg
        if msg.z > 0.5:
            self.lane_detection_count += 1
    
    def lane_found_callback(self, msg):
        if msg.data != self.lane_found:
            self.lane_found = msg.data
            if self.lane_found:
                rospy.loginfo("🛣️  LANE DETECTED!")
                self.last_lane_found_time = rospy.Time.now()
            else:
                rospy.logwarn("❌ Lane lost")
    
    def lane_center_callback(self, msg):
        self.lane_center = msg
    
    def obstacle_callback(self, msg):
        if msg.data != self.obstacle_detected:
            self.obstacle_detected = msg.data
            if self.obstacle_detected:
                rospy.logwarn("🚧 OBSTACLE DETECTED! Emergency protocols activated")
                self.obstacle_detection_count += 1
            else:
                rospy.loginfo("✅ Path clear - resuming normal operation")
    
    def control_status_callback(self, msg):
        self.control_active = msg.data
        if self.control_active:
            self.last_control_time = rospy.Time.now()
    
    def cmd_vel_callback(self, msg):
        self.current_velocity = msg
        self.control_command_count += 1
        
        # Log significant control actions
        if self.lane_found and self.lane_pose and self.lane_center:
            lateral_error = self.lane_pose.x
            heading_error = self.lane_pose.y
            
            # Log when making significant corrections
            if abs(lateral_error) > 0.2 or abs(heading_error) > 0.3:
                rospy.loginfo(f"🎮 Correcting: lateral={lateral_error:.3f}, heading={heading_error:.3f} "
                             f"→ cmd=({msg.linear.x:.2f}, {msg.angular.z:.2f})")
            elif self.lane_found:
                rospy.loginfo_throttle(5, f"🚗 Following: pos=({self.lane_center.x:.2f}) "
                                      f"vel=({msg.linear.x:.2f}, {msg.angular.z:.2f})")
    
    def report_system_status(self, event):
        """Report comprehensive system status"""
        current_time = rospy.Time.now()
        uptime = (current_time - self.system_start_time).to_sec()
        
        # System status indicators
        lane_status = "🛣️  ACTIVE" if self.lane_found else "❌ LOST"
        control_status = "🎮 ACTIVE" if self.control_active else "⏸️  INACTIVE"
        obstacle_status = "🚧 BLOCKED" if self.obstacle_detected else "✅ CLEAR"
        
        # Performance metrics
        lane_detection_rate = self.lane_detection_count / max(uptime, 1.0)
        control_rate = self.control_command_count / max(uptime, 1.0)
        
        # Current state information
        lateral_error = self.lane_pose.d if self.lane_pose else 0.0
        heading_error = self.lane_pose.phi if self.lane_pose else 0.0
        current_speed = self.current_velocity.linear.x if self.current_velocity else 0.0
        current_angular = self.current_velocity.angular.z if self.current_velocity else 0.0
        
        # System health assessment
        health_score = self.calculate_system_health()
        health_icon = "🟢" if health_score > 80 else "🟡" if health_score > 60 else "🔴"
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("🤖 LANE FOLLOWING SYSTEM STATUS")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"📊 SYSTEM HEALTH: {health_icon} {health_score:.1f}%")
        rospy.loginfo(f"🛣️  Lane Detection: {lane_status}")
        rospy.loginfo(f"🎮 Control System: {control_status}")
        rospy.loginfo(f"🚧 Obstacle Status: {obstacle_status}")
        
        if self.lane_found and self.lane_pose:
            rospy.loginfo(f"📍 CURRENT POSITION:")
            rospy.loginfo(f"   Lateral Error: {lateral_error:.4f} (0 = perfect center)")
            rospy.loginfo(f"   Heading Error: {heading_error:.4f} rad")
            rospy.loginfo(f"   In Lane: {self.lane_pose.z > 0.5}")
        
        if self.current_velocity:
            rospy.loginfo(f"🚗 CURRENT MOTION:")
            rospy.loginfo(f"   Speed: {current_speed:.3f} m/s")
            rospy.loginfo(f"   Turn Rate: {current_angular:.3f} rad/s")
        
        rospy.loginfo(f"📈 PERFORMANCE METRICS:")
        rospy.loginfo(f"   Lane Detection Rate: {lane_detection_rate:.1f} Hz")
        rospy.loginfo(f"   Control Command Rate: {control_rate:.1f} Hz")
        rospy.loginfo(f"   Obstacle Detections: {self.obstacle_detection_count}")
        rospy.loginfo(f"   System Uptime: {uptime:.1f} seconds")
        
        # System recommendations
        self.provide_system_recommendations(health_score, lateral_error, heading_error)
        
        rospy.loginfo("=" * 60)
    
    def calculate_system_health(self):
        """Calculate overall system health score"""
        health_score = 100.0
        
        # Deduct points for various issues
        if not self.lane_found:
            health_score -= 40  # Major issue
        
        if not self.control_active:
            health_score -= 30  # Significant issue
        
        if self.obstacle_detected:
            health_score -= 20  # Safety concern
        
        # Deduct points for poor lane tracking
        if self.lane_pose:
            if abs(self.lane_pose.x) > 0.3:  # Poor lateral positioning
                health_score -= 15
            if abs(self.lane_pose.y) > 0.4:  # Poor heading
                health_score -= 10
        
        # Check for recent activity
        current_time = rospy.Time.now()
        if self.last_lane_found_time:
            time_since_lane = (current_time - self.last_lane_found_time).to_sec()
            if time_since_lane > 10:  # No lane detection in 10 seconds
                health_score -= 25
        
        if self.last_control_time:
            time_since_control = (current_time - self.last_control_time).to_sec()
            if time_since_control > 5:  # No control commands in 5 seconds
                health_score -= 15
        
        return max(0.0, min(100.0, health_score))
    
    def provide_system_recommendations(self, health_score, lateral_error, heading_error):
        """Provide system optimization recommendations"""
        rospy.loginfo(f"💡 SYSTEM RECOMMENDATIONS:")
        
        if health_score < 50:
            rospy.loginfo("   ⚠️  CRITICAL: System health is poor - check all components")
        
        if not self.lane_found:
            rospy.loginfo("   🛣️  Check camera feed and lane detection parameters")
            rospy.loginfo("   🔧 Verify lighting conditions and lane visibility")
        
        if abs(lateral_error) > 0.2:
            rospy.loginfo("   🎯 Consider tuning lateral PID parameters")
            rospy.loginfo("   📐 Check camera calibration and mounting")
        
        if abs(heading_error) > 0.3:
            rospy.loginfo("   🧭 Consider tuning heading PID parameters")
            rospy.loginfo("   🔄 Check for systematic heading bias")
        
        if self.obstacle_detected:
            rospy.loginfo("   🚧 Obstacle avoidance active - check path ahead")
        
        if not self.control_active:
            rospy.loginfo("   🎮 Control system inactive - check controller node")
        
        # Performance-based recommendations
        current_time = rospy.Time.now()
        uptime = (current_time - self.system_start_time).to_sec()
        
        if uptime > 30:  # Only after system has been running for a while
            detection_rate = self.lane_detection_count / uptime
            if detection_rate < 5:  # Less than 5 Hz detection
                rospy.loginfo("   📊 Low detection rate - check processing performance")
            
            control_rate = self.control_command_count / uptime
            if control_rate < 10:  # Less than 10 Hz control
                rospy.loginfo("   🎮 Low control rate - check controller performance")
        
        if health_score > 90:
            rospy.loginfo("   ✅ System operating optimally!")
        elif health_score > 70:
            rospy.loginfo("   👍 System performing well with minor issues")
        elif health_score > 50:
            rospy.loginfo("   ⚠️  System functional but needs attention")
        else:
            rospy.loginfo("   🚨 System requires immediate attention")

if __name__ == '__main__':
    try:
        monitor = LaneSystemMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass