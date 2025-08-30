#!/bin/bash

echo "🦆 Setting up YOLOv8 for Duckietown 🦆"
echo "======================================"

# Check Python and pip
echo "Checking Python environment..."
python3 --version || { echo "Python3 not found!"; exit 1; }

# Install YOLOv8 dependencies
echo "Installing YOLOv8 dependencies..."
pip3 install ultralytics opencv-python requests numpy torch torchvision

# Create models directory
echo "Creating models directory..."
mkdir -p ~/.yolo_models

# Download default COCO model
echo "Downloading YOLOv8 nano model..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Make scripts executable
echo "Making scripts executable..."
chmod +x src/yolov8/scripts/*.py

# Build workspace
echo "Building catkin workspace..."
if [ -f "devel/setup.bash" ]; then
    source devel/setup.bash
fi
catkin_make

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Quick Start Options:"
echo ""
echo "1️⃣  Use pre-trained COCO model (recommended for quick start):"
echo "   ./run_yolo_system.sh"
echo ""
echo "2️⃣  List available models:"
echo "   python3 src/yolov8/scripts/model_manager.py list"
echo ""
echo "3️⃣  Get model recommendations:"
echo "   python3 src/yolov8/scripts/model_manager.py recommend"
echo ""
echo "4️⃣  Launch with specific model:"
echo "   roslaunch yolov8 yolo_system.launch model_path:=yolov8s.pt"
echo ""
echo "🎯 For Duckietown-specific objects, consider training a custom model!"
echo "   See: src/yolov8/scripts/train_duckietown_model.py"