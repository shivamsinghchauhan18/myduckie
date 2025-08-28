#!/bin/bash

echo "🎯 DUCKIEBOT CAMERA STARTUP SCRIPT"
echo "=================================="

# Check if we're on DuckieBot host or in container
if [ -f /.dockerenv ]; then
    echo "📦 Inside Docker container - checking camera status..."
    
    # Check if camera topics exist
    echo "📡 Checking camera topics..."
    rostopic list | grep camera
    
    # Check if camera node is running
    echo "🔍 Checking camera nodes..."
    rosnode list | grep camera
    
    # Try to get camera info
    echo "📷 Testing camera data..."
    timeout 5 rostopic echo /camera_node/image/compressed --noarr | head -2
    
    echo ""
    echo "💡 If no camera data, exit container and run:"
    echo "   dts duckiebot demo camera --duckiebot_name pinkduckie"
    
else
    echo "🏠 On DuckieBot host - starting camera..."
    
    # Start camera driver
    echo "📷 Starting camera driver..."
    dts duckiebot demo camera --duckiebot_name pinkduckie &
    
    # Wait a bit
    sleep 5
    
    echo "✅ Camera should now be running!"
    echo "📦 Enter container to test:"
    echo "   docker exec -it dt-duckiebot-interface /bin/bash"
fi