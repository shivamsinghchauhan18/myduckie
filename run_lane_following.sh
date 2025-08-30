#!/bin/bash

echo "🛣️  Starting Advanced DuckieTown Lane Following System"
echo "=================================================="

# Compile the workspace
echo "📦 Building ROS workspace..."
catkin_make

# Source the setup file
echo "🔧 Sourcing ROS environment..."
source devel/setup.bash

# Launch the advanced lane following system
echo "🚀 Launching lane following system..."
roslaunch lane_follower advanced_lane_following.launch robot_name:=blueduckie

echo "✅ Lane following system started successfully!"