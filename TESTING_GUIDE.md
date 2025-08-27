# Object Follower Testing Guide

## Initial Setup and Testing

### 1. Build the Package
```bash
cd /home/sumeettt/duckie_ws
catkin_make
source devel/setup.bash
```

### 2. Test Locally First
```bash
# Test the system tester
rosrun object_follower test_system.py

# In separate terminals, test individual nodes:
# Terminal 1: Motor controller
rosrun object_follower motor_controller.py

# Terminal 2: Run system test
rosrun object_follower test_system.py
```

### 3. Deploy to DuckieBot
```bash
# Connect to your DuckieBot
ssh duckie@pinkduckie.local

# Copy the package to the DuckieBot
scp -r /home/sumeettt/duckie_ws/src/object_follower duckie@pinkduckie.local:~/catkin_ws/src/

# On DuckieBot, build the package
ssh duckie@pinkduckie.local
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# Access the container
docker exec -it dt-duckiebot-interface /bin/bash
```

### 3. Test Individual Components

#### Test Camera Access
```bash
# Check if camera is working
rostopic list | grep camera
rostopic hz /camera_node/image/compressed
```

#### Test Object Detection
```bash
# Run only object detector
rosrun object_follower object_detector.py

# In another terminal, check topics
rostopic echo /object_follower/target_found
rostopic echo /object_follower/target_position
```

#### Test Motor Control
```bash
# Run motor controller (will stop since no target)
rosrun object_follower motor_controller.py

# Check wheel commands
rostopic echo /wheels_driver_node/wheels_cmd
```

#### Test Obstacle Detection
```bash
# Run obstacle detector
rosrun object_follower obstacle_detector.py

# Check obstacle detection
rostopic echo /object_follower/obstacle_detected
```

### 4. Full System Test
```bash
# For local testing (without camera)
roslaunch object_follower object_follower.launch

# For DuckieBot deployment
roslaunch object_follower duckiebot_follower.launch
```

## Testing Scenarios

### Phase 1: Basic Object Detection
1. **Red Object Test**: Use a red colored object (ball, cup, etc.)
2. **Distance Test**: Move object closer/farther to test distance estimation
3. **Tracking Test**: Move object left/right to test tracking

### Phase 2: Following Behavior
1. **Static Following**: Place object at target distance, robot should stay still
2. **Dynamic Following**: Move object slowly, robot should follow
3. **Distance Control**: Test if robot maintains proper distance

### Phase 3: Obstacle Avoidance
1. **Static Obstacle**: Place object in front, robot should stop
2. **Side Obstacles**: Test that side obstacles don't trigger stopping
3. **Emergency Stop**: Test immediate stopping when obstacle detected

## Debugging Commands

### Check Node Status
```bash
rosnode list
rosnode info /object_detector
```

### Monitor Topics
```bash
# List all topics
rostopic list

# Monitor key topics
rostopic echo /object_follower/target_found
rostopic echo /object_follower/obstacle_detected
rostopic echo /cmd_vel
```

### View Debug Images
```bash
# Use rqt_image_view to see processed images
rqt_image_view

# Select topics:
# /object_follower/debug_image
# /object_follower/obstacle_debug_image
```

### Parameter Tuning
```bash
# Adjust detection sensitivity
rosparam set /object_detector/lower_color "[0, 50, 50]"

# Adjust motor speeds
rosparam set /motor_controller/max_speed 0.2

# Adjust PID gains
rosparam set /motor_controller/kp_lateral 1.5
```

## Troubleshooting

### Common Issues
1. **No camera image**: Check camera node is running
2. **Robot doesn't move**: Check wheel driver node
3. **Poor detection**: Adjust color thresholds or lighting
4. **Erratic movement**: Tune PID parameters

### Safety Notes
- Always test in a safe, open area
- Keep emergency stop ready (Ctrl+C)
- Start with low speeds
- Monitor robot behavior closely

## Next Steps
1. Test basic color detection
2. Calibrate distance estimation
3. Tune PID controllers
4. Add CNN-based detection
5. Implement RL training