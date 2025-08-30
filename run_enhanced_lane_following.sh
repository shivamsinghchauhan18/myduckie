#!/usr/bin/env bash
set -euo pipefail

# Enhanced lane following wrapper: preserves current behavior by default.
# Flags: USE_NEURAL=0/1, USE_FUSION=0/1, USE_MPC=0/1

USE_NEURAL=${USE_NEURAL:-0}
USE_FUSION=${USE_FUSION:-0}
USE_MPC=${USE_MPC:-0}
ROBOT_NAME=${ROBOT_NAME:-blueduckie}

neural_flag=false
fusion_flag=false
mpc_flag=false

[[ "$USE_NEURAL" == "1" ]] && neural_flag=true
[[ "$USE_FUSION" == "1" ]] && fusion_flag=true
[[ "$USE_MPC" == "1" ]] && mpc_flag=true

# Source ROS environment if present
if [ -f /opt/ros/noetic/setup.bash ]; then
  source /opt/ros/noetic/setup.bash
fi
if [ -f devel/setup.bash ]; then
  source devel/setup.bash
fi

exec roslaunch lane_follower enhanced_lane_following.launch \
  robot_name:=$ROBOT_NAME \
  use_neural_detection:=$neural_flag \
  use_sensor_fusion:=$fusion_flag \
  use_mpc_control:=$mpc_flag
