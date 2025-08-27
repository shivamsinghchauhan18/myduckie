# DuckieBot Deployment Guide - Phase 2 Enhanced Object Follower

## Pre-Deployment Checklist

### ✅ **Local System Status**
- [x] Phase 2 enhanced system working locally
- [x] Enhanced object detector with Kalman filtering
- [x] Enhanced motor controller with DuckieBot messages
- [x] Performance monitoring
- [x] All scripts executable and tested

### ✅ **Required Components**
- [x] `enhanced_object_detector.py` - Kalman filter object tracking
- [x] `enhanced_motor_controller.py` - DuckieBot message support
- [x] `obstacle_detector.py` - Safety system
- [x] `performance_monitor.py` - Real-time metrics
- [x] `system_monitor.py` - System oversight
- [x] `duckiebot_follower.launch` - DuckieBot configuration

## Deployment Steps

### **Step 1: Copy Package to DuckieBot**
```bash
# From your local machine
cd /home/sumeettt/duckie_ws
scp -r src/object_follower duckie@pinkduckie.local:~/catkin_ws/src/
```

### **Step 2: SSH to DuckieBot**
```bash
ssh duckie@pinkduckie.local
```

### **Step 3: Build on DuckieBot**
```bash
# On DuckieBot
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### **Step 4: Verify DuckieBot Environment**
```bash
# Check camera
rostopic list | grep camera
rostopic hz /camera_node/image/compressed

# Check DuckieBot messages
rosmsg show duckietown_msgs/Twist2DStamped
rosmsg show duckietown_msgs/WheelsCmdStamped

# Check wheel commands topic
rostopic list | grep wheels
```

### **Step 5: Launch Enhanced System**
```bash
# Start the enhanced object follower
roslaunch object_follower duckiebot_follower.launch
```

## DuckieBot-Specific Configuration

### **Camera Configuration**
- **Input**: `/camera_node/image/compressed` (DuckieBot compressed camera)
- **Processing**: Automatic decompression to `/camera/image_raw`
- **Format**: 640x480 BGR8 images

### **Motor Control Topics**
- **Standard ROS**: `/cmd_vel` (Twist messages)
- **DuckieBot Car**: `/car_cmd_switch_node/cmd` (Twist2DStamped)
- **Wheel Commands**: `/duckiebot_driver/wheels_cmd` (WheelsCmdStamped)

### **Conservative Parameters for Hardware**
- **Max Speed**: 0.3 m/s (reduced for safety)
- **Target Distance**: 0.8m (closer following)
- **PID Gains**: Tuned for physical dynamics

## Testing Protocol

### **Phase 1: Basic Startup (5 minutes)**
1. **Camera Test**:
   ```bash
   # Verify camera feed
   rostopic echo /camera_node/image/compressed | head -5
   rqt_image_view  # View /camera/image_raw
   ```

2. **Node Status**:
   ```bash
   rosnode list | grep object_follower
   rostopic list | grep object_follower
   ```

### **Phase 2: Object Detection (10 minutes)**
1. **Show red object** to camera
2. **Monitor detection**:
   ```bash
   rostopic echo /object_follower/target_found
   rostopic echo /object_follower/target_position
   ```
3. **Check debug visualization**: `rqt_image_view` → `/object_follower/debug_image`

### **Phase 3: Motor Response (10 minutes)**
1. **Verify motor commands**:
   ```bash
   rostopic echo /cmd_vel
   rostopic echo /car_cmd_switch_node/cmd
   rostopic echo /duckiebot_driver/wheels_cmd
   ```
2. **Test following behavior** with red object
3. **Monitor performance**:
   ```bash
   rostopic echo /object_follower/performance
   ```

### **Phase 4: Safety Testing (5 minutes)**
1. **Test obstacle detection**:
   ```bash
   rostopic echo /object_follower/obstacle_detected
   ```
2. **Verify emergency stop** works
3. **Test recovery** after obstacle removal

## Performance Monitoring

### **Real-time Metrics**
```bash
# Watch system performance
rostopic echo /object_follower/performance

# Monitor system status  
rostopic echo /rosout | grep object_follower
```

### **Expected Performance**
- **Detection Rate**: >95% for red objects
- **Stability Score**: >70% for moving targets
- **Response Time**: <200ms from detection to motor command
- **Distance Control**: ±0.2m accuracy

## Troubleshooting

### **Common Issues**

1. **No Camera Feed**:
   - Check: `rostopic list | grep camera`
   - Fix: Restart camera node or check camera hardware

2. **No Object Detection**:
   - Check: Lighting conditions
   - Fix: Adjust color thresholds in enhanced_object_detector.py

3. **Robot Not Moving**:
   - Check: `rostopic echo /cmd_vel`
   - Fix: Verify wheel driver node is running

4. **Poor Following**:
   - Check: PID parameters may need tuning
   - Fix: Adjust gains in launch file

### **Emergency Commands**
```bash
# Stop all nodes
rosnode kill -a

# Emergency stop motors
rostopic pub /cmd_vel geometry_msgs/Twist "linear: {x: 0.0} angular: {z: 0.0}"
```

## Expected Behavior

### **Successful Deployment Indicators**
1. ✅ All 5-6 nodes running without errors
2. ✅ Camera feed visible in rqt_image_view
3. ✅ Red object detection working (target_found = True)
4. ✅ Robot follows red object smoothly
5. ✅ Performance score >70%
6. ✅ Emergency stop works when obstacle detected

### **Performance Targets**
- **Following Accuracy**: Robot stays within 1m of target
- **Response Time**: <1 second to start following new target
- **Stability**: Smooth motion without oscillation
- **Safety**: Immediate stop when obstacle detected

## Next Steps After Successful Deployment

1. **Fine-tune PID parameters** for your specific DuckieBot
2. **Test different colored objects** (blue, green, yellow)
3. **Experiment with different lighting conditions**
4. **Test obstacle avoidance scenarios**
5. **Ready for Phase 3**: CNN integration

---

**🎯 Your Phase 2 enhanced system is ready for DuckieBot deployment!**