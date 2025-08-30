# MyAprilTags Package

This package provides AprilTag detection and stopping behavior for Duckietown robots. When an AprilTag is detected within a specified distance, the robot stops for 2 seconds and then resumes lane following.

## Features

- Real-time AprilTag detection using camera feed
- Distance-based stopping behavior
- Integration with lane following system
- Configurable stop duration and distance threshold

## Nodes

### apriltag_detector.py
- Detects AprilTags in camera images
- Estimates distance to tags
- Publishes detection results and stop signals

### apriltag_controller.py
- Manages stopping behavior when AprilTags are detected
- Controls stop duration (default: 2 seconds)
- Coordinates with lane following system

### apriltag_lane_integration.py
- Integrates AprilTag behavior with lane following
- Routes control commands appropriately
- Handles priority between AprilTag stops and lane following

## Topics

### Published
- `/apriltag_detections` (AprilTagDetectionArray): Detected AprilTags
- `/apriltag_stop_signal` (Bool): Signal to stop when tag is close
- `/lane_following_enable` (Bool): Enable/disable lane following
- `/wheels_cmd` (WheelsCmd): Wheel velocity commands
- `/cmd_vel` (Twist): Velocity commands

### Subscribed
- `/camera/image_raw` (Image): Camera feed for tag detection
- `/lane_following/wheels_cmd` (WheelsCmd): Lane following wheel commands
- `/lane_following/cmd_vel` (Twist): Lane following velocity commands

## Parameters

- `stop_distance_threshold`: Distance threshold for stopping (default: 0.3m)
- `tag_size`: Physical size of AprilTags (default: 0.065m)
- `stop_duration`: How long to stop when tag detected (default: 2.0s)

## Usage

### Build the package
```bash
catkin_make
source devel/setup.bash
```

### Run AprilTag system only
```bash
roslaunch myapriltags apriltag_system.launch
```

### Run with lane following integration
```bash
roslaunch myapriltags apriltag_with_lane_following.launch
```

### Or use the convenience script
```bash
./run_apriltag_system.sh
```

## Dependencies

- ROS Noetic
- OpenCV
- apriltag Python library (`pip install apriltag`)
- duckietown_msgs
- Standard ROS packages (sensor_msgs, geometry_msgs, etc.)

## Installation

1. Install apriltag library:
```bash
pip install apriltag
```

2. Build the workspace:
```bash
catkin_make
```

3. Source the workspace:
```bash
source devel/setup.bash
```

## Configuration

Edit the launch files to adjust parameters:
- Stop distance threshold
- Tag size
- Stop duration

The system is designed to work seamlessly with the existing lane following system, temporarily overriding lane following when AprilTags are detected nearby.