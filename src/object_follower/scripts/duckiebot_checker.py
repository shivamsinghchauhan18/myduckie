#!/usr/bin/env python3

"""
DuckieBot Deployment Checker
Verify all components are ready for DuckieBot deployment
"""

import os
import subprocess

def check_duckiebot_readiness():
    """Check if all components are ready for DuckieBot deployment"""
    
    print("🤖 DUCKIEBOT DEPLOYMENT READINESS CHECK")
    print("=" * 50)
    
    workspace_path = "/home/sumeettt/duckie_ws"
    scripts_path = f"{workspace_path}/src/object_follower/scripts"
    launch_path = f"{workspace_path}/src/object_follower/launch"
    
    # Check required scripts
    required_scripts = [
        "enhanced_object_detector.py",
        "enhanced_motor_controller.py", 
        "obstacle_detector.py",
        "performance_monitor.py",
        "system_monitor.py"
    ]
    
    print("\n📁 CHECKING REQUIRED SCRIPTS:")
    all_scripts_present = True
    for script in required_scripts:
        script_path = f"{scripts_path}/{script}"
        if os.path.exists(script_path):
            # Check if executable
            if os.access(script_path, os.X_OK):
                print(f"  ✅ {script} - Present and executable")
            else:
                print(f"  ⚠️  {script} - Present but not executable")
        else:
            print(f"  ❌ {script} - Missing")
            all_scripts_present = False
    
    # Check launch file
    print("\n🚀 CHECKING LAUNCH FILE:")
    launch_file = f"{launch_path}/duckiebot_follower.launch"
    if os.path.exists(launch_file):
        print(f"  ✅ duckiebot_follower.launch - Present")
        
        # Check launch file content
        with open(launch_file, 'r') as f:
            content = f.read()
            
        # Check for required nodes
        required_nodes = [
            "enhanced_object_detector",
            "enhanced_motor_controller", 
            "obstacle_detector",
            "performance_monitor",
            "image_republisher"
        ]
        
        print("\n🔍 CHECKING LAUNCH FILE CONTENT:")
        for node in required_nodes:
            if node in content:
                print(f"  ✅ {node} node configured")
            else:
                print(f"  ❌ {node} node missing")
        
        # Check DuckieBot-specific configurations
        duckiebot_configs = [
            "/camera_node/image/compressed",
            "enhanced_motor_controller.py",
            "compressed in:=/camera_node/image"
        ]
        
        print("\n⚙️  CHECKING DUCKIEBOT CONFIGURATIONS:")
        for config in duckiebot_configs:
            if config in content:
                print(f"  ✅ {config} - Configured")
            else:
                print(f"  ⚠️  {config} - May need attention")
                
    else:
        print(f"  ❌ duckiebot_follower.launch - Missing")
    
    # Check if workspace is built
    print("\n🔨 CHECKING WORKSPACE BUILD:")
    devel_path = f"{workspace_path}/devel"
    if os.path.exists(devel_path):
        print(f"  ✅ Workspace built (devel folder exists)")
        
        # Check if our package is in the devel space
        try:
            result = subprocess.run(['find', devel_path, '-name', '*object_follower*'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                print(f"  ✅ object_follower package found in devel space")
            else:
                print(f"  ⚠️  object_follower package may need rebuilding")
        except:
            print(f"  ⚠️  Could not verify package in devel space")
    else:
        print(f"  ❌ Workspace not built (run catkin_make)")
    
    # Check dependencies
    print("\n📦 CHECKING DEPENDENCIES:")
    try:
        result = subprocess.run(['rosmsg', 'show', 'duckietown_msgs/Twist2DStamped'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ duckietown_msgs available")
        else:
            print(f"  ❌ duckietown_msgs not available")
    except:
        print(f"  ❌ Cannot check duckietown_msgs")
    
    # Check image_transport
    try:
        result = subprocess.run(['rospack', 'find', 'image_transport'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ image_transport available")
        else:
            print(f"  ❌ image_transport not available")
    except:
        print(f"  ❌ Cannot check image_transport")
    
    print("\n" + "=" * 50)
    print("📋 DEPLOYMENT SUMMARY:")
    if all_scripts_present:
        print("✅ All required scripts are present")
        print("✅ System ready for DuckieBot deployment")
        print("\n🚀 DEPLOYMENT COMMANDS:")
        print(f"1. scp -r {workspace_path}/src/object_follower duckie@pinkduckie.local:~/catkin_ws/src/")
        print("2. ssh duckie@pinkduckie.local")
        print("3. cd ~/catkin_ws && catkin_make && source devel/setup.bash")
        print("4. roslaunch object_follower duckiebot_follower.launch")
    else:
        print("❌ Some components missing - fix issues before deployment")

if __name__ == '__main__':
    check_duckiebot_readiness()