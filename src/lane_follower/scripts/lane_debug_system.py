#!/usr/bin/env python3

"""
Lane Debug System for Advanced Lane Following
Comprehensive debugging and diagnostics
"""

import rospy
import subprocess
import time

def debug_lane_system():
    """Debug the Advanced Lane Following system to identify issues"""
    
    print("🛣️  ADVANCED LANE FOLLOWING SYSTEM DEBUGGER")
    print("=" * 60)
    
    # Check ROS topics
    print("\n📡 AVAILABLE TOPICS:")
    try:
        result = subprocess.run(['rostopic', 'list'], capture_output=True, text=True, timeout=5)
        topics = result.stdout.strip().split('\n')
        lane_topics = [topic for topic in sorted(topics) if 'lane_follower' in topic or 'camera' in topic or 'cmd_vel' in topic]
        
        for topic in lane_topics:
            print(f"  ✓ {topic}")
            
        if not lane_topics:
            print("  ❌ No lane following topics found!")
            
    except Exception as e:
        print(f"  ❌ Error getting topics: {e}")
    
    # Check active nodes
    print("\n🤖 ACTIVE NODES:")
    try:
        result = subprocess.run(['rosnode', 'list'], capture_output=True, text=True, timeout=5)
        nodes = result.stdout.strip().split('\n')
        lane_nodes = [node for node in sorted(nodes) if 'lane' in node.lower()]
        
        for node in lane_nodes:
            print(f"  ✓ {node}")
            
        if not lane_nodes:
            print("  ❌ No lane following nodes found!")
            
    except Exception as e:
        print(f"  ❌ Error getting nodes: {e}")
    
    # Test topic publishing rates
    print("\n📊 TOPIC RATES:")
    test_topics = [
        '/camera_node/image/compressed',
        '/lane_follower/lane_found', 
        '/lane_follower/lane_pose',
        '/lane_follower/lane_center',
        '/cmd_vel'
    ]
    
    for topic in test_topics:
        try:
            result = subprocess.run(['rostopic', 'hz', topic], 
                                  capture_output=True, text=True, timeout=3)
            if result.stdout:
                rate_line = [line for line in result.stdout.split('\n') if 'average rate' in line]
                if rate_line:
                    print(f"  ✓ {topic}: {rate_line[0].strip()}")
                else:
                    print(f"  ⚠️  {topic}: Publishing but no rate data")
            else:
                print(f"  ❌ {topic}: No data")
        except subprocess.TimeoutExpired:
            print(f"  ❌ {topic}: Timeout - likely no data")
        except Exception as e:
            print(f"  ❌ {topic}: Error - {e}")
    
    # Check recent topic messages
    print("\n📝 RECENT MESSAGES:")
    
    # Check lane_found
    try:
        result = subprocess.run(['rostopic', 'echo', '/lane_follower/lane_found', '-n', '1'], 
                              capture_output=True, text=True, timeout=2)
        if result.stdout:
            print(f"  🛣️  lane_found: {result.stdout.strip()}")
        else:
            print("  ❌ lane_found: No recent messages")
    except:
        print("  ❌ lane_found: Topic not available")
    
    # Check lane_pose
    try:
        result = subprocess.run(['rostopic', 'echo', '/lane_follower/lane_pose', '-n', '1'], 
                              capture_output=True, text=True, timeout=2)
        if result.stdout:
            print("  📍 lane_pose: Recent pose data available")
        else:
            print("  ❌ lane_pose: No recent messages")
    except:
        print("  ❌ lane_pose: Topic not available")
    
    # Check camera images
    try:
        result = subprocess.run(['rostopic', 'echo', '/camera_node/image/compressed', '-n', '1'], 
                              capture_output=True, text=True, timeout=2)
        if result.stdout:
            print("  📷 camera: Images being published")
        else:
            print("  ❌ camera: No images")
    except:
        print("  ❌ camera: No camera data")
    
    # System health check
    print("\n🏥 SYSTEM HEALTH CHECK:")
    
    # Check if all required nodes are running
    required_nodes = [
        'advanced_lane_detector',
        'enhanced_lane_controller', 
        'lane_performance_monitor',
        'lane_system_monitor'
    ]
    
    try:
        result = subprocess.run(['rosnode', 'list'], capture_output=True, text=True, timeout=5)
        active_nodes = result.stdout.strip().split('\n')
        
        for required_node in required_nodes:
            node_found = any(required_node in node for node in active_nodes)
            status = "✅ RUNNING" if node_found else "❌ NOT FOUND"
            print(f"  {required_node}: {status}")
            
    except Exception as e:
        print(f"  ❌ Error checking nodes: {e}")
    
    # Performance recommendations
    print("\n💡 DEBUGGING RECOMMENDATIONS:")
    print("  1. Ensure camera is connected and publishing images")
    print("  2. Check lighting conditions for lane visibility")
    print("  3. Verify lane markings are clear (yellow/white)")
    print("  4. Confirm robot is positioned on a lane")
    print("  5. Check ROS master is running: roscore")
    print("  6. Verify all nodes launched successfully")
    
    print("\n🔧 TROUBLESHOOTING COMMANDS:")
    print("  • View camera feed: rostopic echo /camera_node/image/compressed")
    print("  • Monitor lane detection: rostopic echo /lane_follower/lane_found")
    print("  • Check control commands: rostopic echo /cmd_vel")
    print("  • View debug images: rqt_image_view")
    print("  • Monitor system: rostopic echo /lane_follower/detection_info")

if __name__ == '__main__':
    debug_lane_system()