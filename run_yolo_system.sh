#!/bin/bash

# YOLOv8 Object Detection and Avoidance System
# This script launches the complete YOLOv8 system with server-client architecture

echo "Starting YOLOv8 Object Detection and Avoidance System..."

# Check if server URL is provided
SERVER_URL=${1:-"http://localhost:5000"}
BOT_ID=${2:-"duckiebot_01"}

echo "Server URL: $SERVER_URL"
echo "Bot ID: $BOT_ID"

# Source ROS environment
source /opt/ros/noetic/setup.bash
source devel/setup.bash

# Check if YOLOv8 dependencies are installed
echo "Checking YOLOv8 dependencies..."
python3 -c "import ultralytics" 2>/dev/null || {
    echo "Installing YOLOv8 dependencies..."
    pip3 install ultralytics opencv-python requests numpy
}

# Build the workspace if needed
if [ ! -d "devel" ]; then
    echo "Building catkin workspace..."
    catkin_make
    source devel/setup.bash
fi

# Launch the YOLOv8 system
echo "Launching YOLOv8 system..."
roslaunch yolov8 yolo_system.launch server_url:=$SERVER_URL bot_id:=$BOT_ID

echo "YOLOv8 system stopped."