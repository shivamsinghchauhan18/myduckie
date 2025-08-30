#!/bin/bash

echo "🚀 Starting ULTRA-ADVANCED DuckieTown Lane Following System"
echo "============================================================"
echo "🧠 Neural Networks + 🎯 Model Predictive Control + 🔄 Sensor Fusion + 📚 Adaptive Learning"
echo ""

# Check for required dependencies
echo "🔍 Checking system dependencies..."

# Check for PyTorch
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} available')" 2>/dev/null || {
    echo "❌ PyTorch not found. Installing..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

# Check for scikit-learn
python3 -c "import sklearn; print(f'✅ scikit-learn {sklearn.__version__} available')" 2>/dev/null || {
    echo "❌ scikit-learn not found. Installing..."
    pip3 install scikit-learn
}

# Check for scipy optimization
python3 -c "from scipy.optimize import minimize; print('✅ SciPy optimization available')" 2>/dev/null || {
    echo "❌ SciPy optimization not found. Installing..."
    pip3 install scipy
}

echo ""
echo "📦 Building ROS workspace..."
catkin_make

echo ""
echo "🔧 Sourcing ROS environment..."
source devel/setup.bash

echo ""
echo "🎛️  System Configuration:"
echo "   🧠 Neural Lane Detection: ENABLED"
echo "   🎯 Model Predictive Control: ENABLED" 
echo "   🔄 Sensor Fusion (EKF): ENABLED"
echo "   📚 Adaptive Learning: ENABLED"
echo "   🚧 Advanced Obstacle Detection: ENABLED"
echo "   📊 Comprehensive Performance Monitoring: ENABLED"
echo ""

# Set environment variables for enhanced features
export RECORD_DATA=false  # Set to true to record data for analysis
export DISPLAY=${DISPLAY:-}  # For RViz visualization

echo "🚀 Launching ultra-advanced lane following system..."
echo "   This may take a moment to initialize all AI systems..."
echo ""

# Check for launch file preference
LAUNCH_FILE="ultra_advanced_lane_following_standalone.launch"
if [ "$1" = "--full" ]; then
    LAUNCH_FILE="ultra_advanced_lane_following.launch"
    echo "   🔗 Using full integration with AprilTags and YOLO"
else
    echo "   🎯 Using standalone version (core AI systems only)"
    echo "   💡 Use --full flag for complete integration with all packages"
fi

# Check if launch file exists
if [ ! -f "src/lane_follower/launch/$LAUNCH_FILE" ]; then
    echo "❌ Launch file not found: $LAUNCH_FILE"
    echo "   Please ensure the workspace is built correctly"
    exit 1
fi

# Launch the ultra-advanced system
echo "🎬 Starting launch sequence..."
roslaunch lane_follower $LAUNCH_FILE robot_name:=blueduckie \
    use_neural_detection:=true \
    use_mpc_control:=true \
    use_sensor_fusion:=true \
    use_adaptive_learning:=true

# Check launch result
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Ultra-Advanced Lane Following System started successfully!"
else
    echo ""
    echo "❌ Launch failed. Check the error messages above."
    echo "💡 Try running with basic features first:"
    echo "   roslaunch lane_follower advanced_lane_following.launch"
    exit 1
fi

echo ""
echo "✅ Ultra-Advanced Lane Following System started successfully!"
echo ""
echo "🎮 System Features Active:"
echo "   • Deep Learning Lane Detection with CNN"
echo "   • Model Predictive Control with Optimization"
echo "   • Extended Kalman Filter Sensor Fusion"
echo "   • Machine Learning Parameter Adaptation"
echo "   • Multi-Modal Obstacle Detection"
echo "   • Real-Time Performance Analytics"
echo ""
echo "📊 Monitor system performance with:"
echo "   rostopic echo /lane_follower/learning_status"
echo "   rostopic echo /lane_follower/fusion_confidence"
echo "   rostopic echo /lane_follower/performance"
echo ""
echo "🛠️  For debugging, run:"
echo "   python3 src/lane_follower/scripts/lane_debug_system.py"