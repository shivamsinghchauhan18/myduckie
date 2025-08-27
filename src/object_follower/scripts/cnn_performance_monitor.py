#!/usr/bin/env python3

"""
CNN Performance Monitor for Phase 3
Monitors CNN detection performance, model switching, and computational metrics
"""

import rospy
from std_msgs.msg import Bool, Float32, String, Int32
from geometry_msgs.msg import Point
import numpy as np
import time
from collections import deque
import psutil
import os

class CNNPerformanceMonitor:
    def __init__(self):
        rospy.init_node('cnn_performance_monitor', anonymous=True)
        
        # Subscribers
        rospy.Subscriber('/object_follower/target_found', Bool, self.target_found_callback)
        rospy.Subscriber('/object_follower/detection_confidence', Float32, self.confidence_callback)
        rospy.Subscriber('/object_follower/cnn_detection_info', String, self.detection_info_callback)
        rospy.Subscriber('/object_follower/target_position', Point, self.position_callback)
        
        # Publishers
        self.performance_report_pub = rospy.Publisher('/object_follower/cnn_performance_report', String, queue_size=1)
        
        # Performance metrics
        self.detection_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100) 
        self.method_usage = {'cnn': 0, 'color': 0, 'hybrid': 0}
        self.processing_times = deque(maxlen=50)
        
        # System metrics
        self.cpu_usage_history = deque(maxlen=50)
        self.memory_usage_history = deque(maxlen=50)
        
        # Timing
        self.start_time = rospy.Time.now()
        self.last_detection_time = None
        self.detection_intervals = deque(maxlen=20)
        
        # Statistics
        self.total_frames = 0
        self.detection_frames = 0
        self.high_confidence_detections = 0
        
        # Timers
        rospy.Timer(rospy.Duration(5.0), self.quick_report)
        rospy.Timer(rospy.Duration(15.0), self.detailed_report)
        rospy.Timer(rospy.Duration(1.0), self.system_monitoring)
        
        rospy.loginfo("CNN Performance Monitor started")
    
    def target_found_callback(self, msg):
        self.total_frames += 1
        detection_time = rospy.Time.now()
        
        if msg.data:
            self.detection_frames += 1
            
            # Track detection intervals
            if self.last_detection_time:
                interval = (detection_time - self.last_detection_time).to_sec()
                self.detection_intervals.append(interval)
            
            self.last_detection_time = detection_time
        
        self.detection_history.append(msg.data)
    
    def confidence_callback(self, msg):
        confidence = msg.data
        self.confidence_history.append(confidence)
        
        if confidence > 0.8:
            self.high_confidence_detections += 1
    
    def detection_info_callback(self, msg):
        info = msg.data
        
        # Parse method used
        if 'cnn' in info.lower():
            self.method_usage['cnn'] += 1
        elif 'color' in info.lower():
            self.method_usage['color'] += 1
        elif 'hybrid' in info.lower():
            self.method_usage['hybrid'] += 1
    
    def position_callback(self, msg):
        # Could track position stability here
        pass
    
    def system_monitoring(self, event):
        """Monitor system resources"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage_history.append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.memory_usage_history.append(memory_percent)
            
        except Exception as e:
            rospy.logwarn_throttle(10, f"System monitoring error: {e}")
    
    def quick_report(self, event):
        """Quick status report every 5 seconds"""
        if not self.detection_history:
            return
        
        # Detection rate
        recent_detections = list(self.detection_history)[-20:]  # Last 20 frames
        detection_rate = sum(recent_detections) / len(recent_detections) * 100
        
        # Average confidence
        recent_confidence = list(self.confidence_history)[-20:] if self.confidence_history else [0]
        avg_confidence = np.mean(recent_confidence)
        
        # Status indicators
        detection_icon = "🎯" if detection_rate > 70 else "⚠️" if detection_rate > 30 else "❌"
        confidence_icon = "🟢" if avg_confidence > 0.7 else "🟡" if avg_confidence > 0.4 else "🔴"
        
        rospy.loginfo(f"{detection_icon} Detection: {detection_rate:.1f}% | {confidence_icon} Confidence: {avg_confidence:.2f}")
    
    def detailed_report(self, event):
        """Detailed performance report every 15 seconds"""
        if self.total_frames < 10:
            rospy.loginfo("📊 Gathering CNN performance data...")
            return
        
        elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
        
        # Detection metrics
        overall_detection_rate = (self.detection_frames / self.total_frames) * 100
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0
        high_conf_rate = (self.high_confidence_detections / max(1, self.detection_frames)) * 100
        
        # Method usage statistics
        total_method_usage = sum(self.method_usage.values())
        method_percentages = {k: (v/max(1, total_method_usage))*100 for k, v in self.method_usage.items()}
        
        # System performance
        avg_cpu = np.mean(self.cpu_usage_history) if self.cpu_usage_history else 0
        avg_memory = np.mean(self.memory_usage_history) if self.memory_usage_history else 0
        
        # Detection timing
        avg_interval = np.mean(self.detection_intervals) if self.detection_intervals else 0
        detection_fps = 1.0 / avg_interval if avg_interval > 0 else 0
        
        # Generate report
        rospy.loginfo("=" * 60)
        rospy.loginfo("🧠 CNN PERFORMANCE REPORT")
        rospy.loginfo("=" * 60)
        
        rospy.loginfo(f"🎯 DETECTION PERFORMANCE:")
        rospy.loginfo(f"   Overall Detection Rate: {overall_detection_rate:.1f}%")
        rospy.loginfo(f"   Average Confidence: {avg_confidence:.2f}")
        rospy.loginfo(f"   High Confidence Rate: {high_conf_rate:.1f}%")
        rospy.loginfo(f"   Detection FPS: {detection_fps:.1f} Hz")
        
        rospy.loginfo(f"🔧 METHOD USAGE:")
        rospy.loginfo(f"   CNN: {method_percentages['cnn']:.1f}%")
        rospy.loginfo(f"   Color: {method_percentages['color']:.1f}%")
        rospy.loginfo(f"   Hybrid: {method_percentages['hybrid']:.1f}%")
        
        rospy.loginfo(f"💻 SYSTEM RESOURCES:")
        rospy.loginfo(f"   Average CPU: {avg_cpu:.1f}%")
        rospy.loginfo(f"   Average Memory: {avg_memory:.1f}%")
        
        # Performance assessment
        overall_score = self.calculate_performance_score(
            overall_detection_rate, avg_confidence, high_conf_rate, avg_cpu
        )
        
        performance_grade = self.get_performance_grade(overall_score)
        rospy.loginfo(f"📈 OVERALL ASSESSMENT:")
        rospy.loginfo(f"   Performance Score: {overall_score:.1f}/100")
        rospy.loginfo(f"   Grade: {performance_grade}")
        
        rospy.loginfo("=" * 60)
        
        # Publish detailed report
        report = {
            'detection_rate': overall_detection_rate,
            'avg_confidence': avg_confidence,
            'high_conf_rate': high_conf_rate,
            'method_usage': method_percentages,
            'cpu_usage': avg_cpu,
            'memory_usage': avg_memory,
            'performance_score': overall_score,
            'grade': performance_grade
        }
        
        self.performance_report_pub.publish(String(str(report)))
    
    def calculate_performance_score(self, detection_rate, avg_confidence, high_conf_rate, cpu_usage):
        """Calculate overall performance score (0-100)"""
        # Weight factors
        detection_weight = 0.3
        confidence_weight = 0.3
        high_conf_weight = 0.2
        efficiency_weight = 0.2
        
        # Normalize scores
        detection_score = min(100, detection_rate)
        confidence_score = avg_confidence * 100
        high_conf_score = high_conf_rate
        efficiency_score = max(0, 100 - cpu_usage)  # Lower CPU = higher score
        
        # Calculate weighted average
        overall_score = (
            detection_score * detection_weight +
            confidence_score * confidence_weight +
            high_conf_score * high_conf_weight +
            efficiency_score * efficiency_weight
        )
        
        return overall_score
    
    def get_performance_grade(self, score):
        """Convert performance score to letter grade"""
        if score >= 90:
            return "A+ 🏆 EXCELLENT"
        elif score >= 80:
            return "A  🥇 VERY GOOD"
        elif score >= 70:
            return "B+ 🥈 GOOD"
        elif score >= 60:
            return "B  🥉 ACCEPTABLE"
        elif score >= 50:
            return "C  ⚠️  NEEDS IMPROVEMENT"
        else:
            return "F  ❌ POOR"

if __name__ == '__main__':
    try:
        monitor = CNNPerformanceMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass