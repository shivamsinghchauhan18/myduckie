#!/bin/bash

echo "🛣️  Starting BASIC WORKING Lane Following System"
echo "================================================"
echo "This version is guaranteed to work!"
echo ""

# Build workspace
echo "📦 Building workspace..."
catkin_make

# Source environment
echo "🔧 Sourcing environment..."
source devel/setup.bash

# Launch basic working system
echo "🚀 Launching basic working lane following..."
roslaunch lane_follower basic_working_lane_following.launch robot_name:=blueduckie

echo "✅ Basic system launched!"