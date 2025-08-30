# YOLOv8 Object Detection and Intelligent Avoidance System

## Overview

This package provides advanced object detection using YOLOv8 with intelligent avoidance capabilities, distance calculation, emergency recovery, and server-client architecture for remote monitoring and control.

## Features

### Core Capabilities
- **Advanced Object Detection**: YOLOv8-based real-time object detection
- **Distance Calculation**: Multi-method distance estimation using object size, ground plane, and heuristics
- **Intelligent Avoidance**: Smart obstacle avoidance with priority-based decision making
- **Emergency Recovery**: Autonomous recovery system for critical situations
- **Safety Monitoring**: Comprehensive safety analysis and alerting
- **Server-Client Architecture**: Remote monitoring and control via API calls

### Safety Features
- **Safe Distance Maintenance**: Configurable safe distances for different object types
- **Emergency Stop**: Immediate stopping when critical distances are breached
- **Recovery Procedures**: Automated backup, scan, and path-finding
- **Human Intervention Requests**: Automatic help requests when recovery fails
- **Real-time Safety Monitoring**: Continuous safety assessment and alerting

## Architecture

### Node Structure
```
yolo_detector.py          # Main YOLOv8 detection and distance calculation
distance_calculator.py    # Advanced distance estimation and refinement
avoidance_controller.py   # Intelligent obstacle avoidance logic
emergency_recovery.py     # Emergency situation recovery system
safety_monitor.py         # Safety monitoring and alerting
yolo_client.py           # Server communication and API interface
```

### Communication Flow
```
Camera → YOLOv8 Detector → Distance Calculator → Avoidance Controller → Wheels
                ↓                    ↓                    ↓
         Safety Monitor ← Emergency Recovery ← Server Client
                ↓                    ↓                    ↓
            Alerts              Recovery Actions      Server API
```

## Installation

### Dependencies
```bash
# Install YOLOv8 and dependencies
pip3 install ultralytics opencv-python requests numpy

# ROS dependencies (should already be available)
sudo apt-get install ros-noetic-cv-bridge ros-noetic-sensor-msgs
```

### Build
```bash
# In your catkin workspace
catkin_make
source devel/setup.bash
```

## Usage

### Quick Start
```bash
# Launch complete system with default server
./run_yolo_system.sh

# Launch with custom server
./run_yolo_system.sh http://your-server:5000 your_bot_id
```

### Manual Launch
```bash
# Launch all components
roslaunch yolov8 yolo_system.launch server_url:=http://localhost:5000 bot_id:=duckiebot_01

# Launch individual components
rosrun yolov8 yolo_detector.py
rosrun yolov8 avoidance_controller.py
```

## Configuration

### Parameters

#### Detection Parameters
- `confidence_threshold`: Minimum detection confidence (default: 0.5)
- `safe_distance`: Safe distance threshold in meters (default: 0.8)
- `emergency_distance`: Emergency stop distance in meters (default: 0.3)

#### Camera Parameters
- `camera_height`: Camera height from ground in meters (default: 0.1)
- `camera_angle`: Camera tilt angle in radians (default: 0.0)
- `focal_length`: Camera focal length in pixels (default: 525.0)

#### Server Parameters
- `server_url`: Server API endpoint (default: http://localhost:5000)
- `bot_id`: Unique bot identifier (default: duckiebot_01)
- `api_key`: Authentication key for server (default: default_key)

### Object Size Database
The system uses known real-world object dimensions for distance calculation:
```python
known_widths = {
    'person': 0.6,      # meters
    'car': 1.8,         # meters
    'truck': 2.5,       # meters
    'bicycle': 0.6,     # meters
    'motorcycle': 0.8,  # meters
    'bus': 2.5,         # meters
    'stop sign': 0.3,   # meters
    'traffic light': 0.3 # meters
}
```

## Server API Integration

### Required Server Endpoints

#### Bot Registration
```
POST /api/register
{
    "bot_id": "duckiebot_01",
    "api_key": "your_key",
    "capabilities": ["object_detection", "avoidance", "recovery"],
    "status": "online"
}
```

#### Detection Updates
```
POST /api/detection_update
{
    "bot_id": "duckiebot_01",
    "detections": [...],
    "distances": [...],
    "safety_status": "SAFE"
}
```

#### Emergency Alerts
```
POST /api/emergency
{
    "bot_id": "duckiebot_01",
    "event": "emergency_stop",
    "timestamp": 1234567890
}
```

#### Command Polling
```
GET /api/commands/duckiebot_01
Response: {
    "commands": [
        {"id": "cmd_123", "type": "stop", "priority": "high"}
    ]
}
```

## Topics

### Published Topics
- `/yolo/detections` (String): Raw detection results
- `/yolo/distances` (String): Distance calculations
- `/yolo/safety_status` (String): Current safety status
- `/yolo/emergency` (Bool): Emergency state
- `/yolo/avoidance_status` (String): Avoidance controller status
- `/yolo/safety_alert` (String): Safety alerts and warnings
- `/wheels_driver_node/wheels_cmd` (WheelsCmdStamped): Motor commands

### Subscribed Topics
- `/camera/image/compressed` (CompressedImage): Camera input
- `/yolo/server_command` (String): Commands from server

## Safety System

### Safety Levels
1. **SAFE**: Normal operation, no obstacles detected
2. **CAUTION**: Objects detected at moderate distance
3. **WARNING**: Objects approaching safe distance threshold
4. **EMERGENCY**: Critical distance breached, immediate action required

### Emergency Recovery Sequence
1. **STOPPING**: Immediate brake application
2. **BACKING**: Slow backward movement to create space
3. **SCANNING**: Left-right scan to find clear path
4. **TURNING**: Turn toward clearest direction
5. **RESUMING**: Gradual forward movement
6. **INTERVENTION**: Request human help if recovery fails

### Threat Assessment
Objects are classified by threat level based on:
- Distance to robot
- Object type and size
- Position relative to robot path
- Movement patterns (if detectable)

## Monitoring and Debugging

### Log Levels
- **INFO**: Normal operation status
- **WARN**: Safety warnings and recoverable issues
- **ERROR**: Critical errors requiring attention

### Key Metrics
- Detection frequency and accuracy
- Distance calculation reliability
- Avoidance success rate
- Emergency recovery effectiveness
- Server communication status

### Debugging Tools
```bash
# Monitor detection output
rostopic echo /yolo/detections

# Check safety status
rostopic echo /yolo/safety_status

# View distance calculations
rostopic echo /yolo/refined_distances

# Monitor emergency state
rostopic echo /yolo/emergency
```

## Troubleshooting

### Common Issues

#### No Detections
- Check camera connection and topic
- Verify YOLOv8 model loading
- Confirm lighting conditions

#### Inaccurate Distances
- Calibrate camera parameters
- Update object size database
- Check camera mounting angle

#### Server Connection Issues
- Verify server URL and network connectivity
- Check API key authentication
- Monitor firewall settings

#### Emergency Recovery Failures
- Check for physical obstacles
- Verify wheel command topics
- Review recovery timeout settings

### Performance Optimization
- Use YOLOv8n (nano) model for speed
- Adjust detection confidence threshold
- Optimize image resolution
- Tune control loop frequencies

## Integration with Other Systems

### Lane Following Integration
The YOLOv8 system can work alongside lane following:
```bash
# Launch both systems
roslaunch lane_follower advanced_lane_following.launch &
roslaunch yolov8 yolo_system.launch
```

### AprilTag Integration
Compatible with AprilTag detection for enhanced navigation:
```bash
roslaunch myapriltags apriltag_with_lane_following.launch &
roslaunch yolov8 yolo_system.launch
```

## Development and Extension

### Adding New Object Types
1. Update `known_widths` dictionary in `yolo_detector.py`
2. Add threat level classification in `distance_calculator.py`
3. Update avoidance strategies in `avoidance_controller.py`

### Custom Recovery Behaviors
Extend `emergency_recovery.py` with new recovery sequences:
```python
def custom_recovery_sequence(self):
    # Implement custom recovery logic
    pass
```

### Server API Extensions
Add new endpoints and handlers in `yolo_client.py`:
```python
def handle_custom_command(self, command):
    # Process custom server commands
    pass
```

## License
MIT License - See LICENSE file for details

## Support
For issues and questions, please check the troubleshooting section or contact the development team.