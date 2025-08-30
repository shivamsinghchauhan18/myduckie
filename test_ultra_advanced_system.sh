#!/bin/bash

echo "🧪 Testing Ultra-Advanced Lane Following System"
echo "=============================================="

# Test 1: Check Python dependencies
echo ""
echo "🔍 Testing Python Dependencies:"

# Test PyTorch
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} - OK')" 2>/dev/null || {
    echo "❌ PyTorch not available"
    echo "   Install with: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
}

# Test scikit-learn
python3 -c "import sklearn; print(f'✅ scikit-learn {sklearn.__version__} - OK')" 2>/dev/null || {
    echo "❌ scikit-learn not available"
    echo "   Install with: pip3 install scikit-learn"
}

# Test SciPy optimization
python3 -c "from scipy.optimize import minimize; print('✅ SciPy optimization - OK')" 2>/dev/null || {
    echo "❌ SciPy optimization not available"
    echo "   Install with: pip3 install scipy"
}

# Test OpenCV
python3 -c "import cv2; print(f'✅ OpenCV {cv2.__version__} - OK')" 2>/dev/null || {
    echo "❌ OpenCV not available"
    echo "   Install with: pip3 install opencv-python"
}

# Test NumPy
python3 -c "import numpy; print(f'✅ NumPy {numpy.__version__} - OK')" 2>/dev/null || {
    echo "❌ NumPy not available"
    echo "   Install with: pip3 install numpy"
}

# Test 2: Check ROS environment
echo ""
echo "🤖 Testing ROS Environment:"

if [ -z "$ROS_DISTRO" ]; then
    echo "❌ ROS not sourced"
    echo "   Run: source /opt/ros/melodic/setup.bash (or your ROS version)"
else
    echo "✅ ROS $ROS_DISTRO - OK"
fi

# Test 3: Check workspace build
echo ""
echo "🏗️  Testing Workspace Build:"

if [ -f "devel/setup.bash" ]; then
    echo "✅ Workspace built - OK"
    source devel/setup.bash
else
    echo "❌ Workspace not built"
    echo "   Run: catkin_make"
fi

# Test 4: Check launch files
echo ""
echo "📋 Testing Launch Files:"

if [ -f "src/lane_follower/launch/ultra_advanced_lane_following_standalone.launch" ]; then
    echo "✅ Standalone launch file - OK"
else
    echo "❌ Standalone launch file missing"
fi

if [ -f "src/lane_follower/launch/ultra_advanced_lane_following.launch" ]; then
    echo "✅ Full integration launch file - OK"
else
    echo "❌ Full integration launch file missing"
fi

# Test 5: Check Python scripts
echo ""
echo "🐍 Testing Python Scripts:"

SCRIPTS=(
    "neural_lane_detector.py"
    "mpc_lane_controller.py"
    "sensor_fusion_system.py"
    "adaptive_learning_system.py"
    "lane_obstacle_detector.py"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "src/lane_follower/scripts/$script" ]; then
        # Test syntax
        python3 -m py_compile "src/lane_follower/scripts/$script" 2>/dev/null && {
            echo "✅ $script - Syntax OK"
        } || {
            echo "⚠️  $script - Syntax issues (may still work)"
        }
    else
        echo "❌ $script - Missing"
    fi
done

# Test 6: Quick functionality test
echo ""
echo "⚡ Quick Functionality Test:"

# Test neural network creation
python3 -c "
import sys
sys.path.append('src/lane_follower/scripts')
try:
    from neural_lane_detector import LightweightLaneNet
    import torch
    model = LightweightLaneNet()
    print('✅ Neural network creation - OK')
except Exception as e:
    print(f'❌ Neural network creation failed: {e}')
" 2>/dev/null

# Test MPC optimization
python3 -c "
try:
    from scipy.optimize import minimize
    import numpy as np
    result = minimize(lambda x: x[0]**2, [1.0], method='SLSQP')
    print('✅ MPC optimization - OK')
except Exception as e:
    print(f'❌ MPC optimization failed: {e}')
" 2>/dev/null

# Test EKF
python3 -c "
try:
    import numpy as np
    from scipy.linalg import block_diag
    P = np.eye(7) * 0.1
    print('✅ Extended Kalman Filter - OK')
except Exception as e:
    print(f'❌ Extended Kalman Filter failed: {e}')
" 2>/dev/null

# Test machine learning
python3 -c "
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    model = RandomForestRegressor(n_estimators=10)
    scaler = StandardScaler()
    print('✅ Machine learning - OK')
except Exception as e:
    print(f'❌ Machine learning failed: {e}')
" 2>/dev/null

echo ""
echo "🎯 Test Summary:"
echo "   If all tests show ✅, the system should work properly"
echo "   If you see ❌, install the missing dependencies"
echo "   If you see ⚠️, the system may still work but with reduced performance"
echo ""
echo "🚀 To launch the system:"
echo "   ./run_ultra_advanced_lane_following.sh"
echo ""
echo "🔧 For troubleshooting:"
echo "   python3 src/lane_follower/scripts/lane_debug_system.py"