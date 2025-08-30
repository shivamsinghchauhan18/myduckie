#!/bin/bash

# Run AprilTag system with lane following integration
echo "Starting AprilTag Detection and Lane Following System..."

# Source ROS environment
source /opt/ros/noetic/setup.bash
source devel/setup.bash

# Launch the integrated system
roslaunch myapriltags apriltag_with_lane_following.launch