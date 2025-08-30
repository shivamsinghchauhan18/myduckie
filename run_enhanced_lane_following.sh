#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting ENHANCED DuckieTown Lane Following (safe-by-default)"
echo "==============================================================="
echo "🛡️  Baseline preserved | 🔀 Mux-enabled | 🧪 Shadow evaluation"
echo

# Flags: USE_NEURAL=0/1, USE_FUSION=0/1, USE_MPC=0/1, ROBOT_NAME
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

echo "🔍 Checking system dependencies..."

# PyTorch (optional for neural detector)
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} available')" 2>/dev/null || echo "ℹ️ PyTorch not found (neural may fallback)"

# SciPy (for current MPC)
python3 -c "from scipy.optimize import minimize; print('✅ SciPy optimization available')" 2>/dev/null || echo "⚠️ SciPy missing (MPC will not run)"

echo
echo "📦 Building ROS workspace..."
if command -v catkin_make >/dev/null 2>&1; then
  catkin_make
else
  echo "⚠️ catkin_make not found; assuming prebuilt devel"
fi

echo
echo "🔧 Sourcing ROS environment..."
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
unset ROS_IP
if [ -f /opt/ros/noetic/setup.bash ]; then
  source /opt/ros/noetic/setup.bash
fi
if [ -f devel/setup.bash ]; then
  source devel/setup.bash
fi

echo
echo "🎛️  System Configuration:"
echo "   🤖 Robot: $ROBOT_NAME"
echo "   🧠 Neural Lane Detection: ${neural_flag^^}"
echo "   🔄 Sensor Fusion: ${fusion_flag^^}"
echo "   🎯 MPC: ${mpc_flag^^}"
echo

LAUNCH_FILE="src/lane_follower/launch/enhanced_lane_following.launch"
if [ ! -f "$LAUNCH_FILE" ]; then
  echo "❌ Launch file not found: $LAUNCH_FILE"
  exit 1
fi

echo "🎬 Starting launch sequence..."
roslaunch lane_follower enhanced_lane_following.launch \
  robot_name:=$ROBOT_NAME \
  use_neural_detection:=$neural_flag \
  use_sensor_fusion:=$fusion_flag \
  use_mpc_control:=$mpc_flag

rc=$?
if [ $rc -eq 0 ]; then
  echo "\n✅ Enhanced Lane Following started successfully!"
else
  echo "\n❌ Launch failed. Try baseline: roslaunch lane_follower advanced_lane_following.launch"
  exit $rc
fi
