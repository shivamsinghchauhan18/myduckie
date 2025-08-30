#!/usr/bin/env python3

import os
import requests
import rospy
from ultralytics import YOLO

class YOLOModelManager:
    def __init__(self):
        self.models_dir = os.path.expanduser('~/.yolo_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.available_models = {
            'coco_nano': {
                'file': 'yolov8n.pt',
                'description': 'Pre-trained COCO dataset (fastest)',
                'classes': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign']
            },
            'coco_small': {
                'file': 'yolov8s.pt', 
                'description': 'Pre-trained COCO dataset (balanced)',
                'classes': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign']
            },
            'duckietown_custom': {
                'file': 'models/yolov8_duckietown.pt',
                'description': 'Custom trained for Duckietown (TRAINED ✅)',
                'classes': ['duckiebot', 'duckie', 'cone', 'stop_sign', 'person', 'building', 'tree', 'road_sign', 'barrier', 'intersection_sign']
            }
        }
    
    def download_model(self, model_name):
        """Download a model if not already present"""
        if model_name not in self.available_models:
            print(f"Unknown model: {model_name}")
            return False
        
        model_info = self.available_models[model_name]
        model_path = os.path.join(self.models_dir, model_info['file'])
        
        if os.path.exists(model_path):
            print(f"Model {model_name} already exists at {model_path}")
            return True
        
        if model_name.startswith('coco'):
            # YOLO will auto-download COCO models
            try:
                model = YOLO(model_info['file'])
                print(f"Downloaded {model_name} successfully")
                return True
            except Exception as e:
                print(f"Failed to download {model_name}: {e}")
                return False
        
        elif model_name == 'duckietown_custom':
            print("Custom Duckietown model needs to be trained first.")
            print("Run: python3 train_duckietown_model.py")
            return False
        
        return False
    
    def list_models(self):
        """List available models"""
        print("\nAvailable YOLO Models:")
        print("-" * 50)
        
        for name, info in self.available_models.items():
            model_path = os.path.join(self.models_dir, info['file'])
            status = "✓ Available" if os.path.exists(model_path) or name.startswith('coco') else "✗ Not available"
            
            print(f"{name:20} | {status:15} | {info['description']}")
            print(f"{'':20} | {'':15} | Classes: {len(info['classes'])} objects")
            print()
    
    def get_model_path(self, model_name):
        """Get the path to a specific model"""
        if model_name not in self.available_models:
            return None
        
        model_info = self.available_models[model_name]
        
        if model_name.startswith('coco'):
            return model_info['file']  # YOLO will handle download
        else:
            model_path = os.path.join(self.models_dir, model_info['file'])
            return model_path if os.path.exists(model_path) else None
    
    def test_model(self, model_name, test_image_path=None):
        """Test a model with a sample image"""
        model_path = self.get_model_path(model_name)
        if not model_path:
            print(f"Model {model_name} not available")
            return False
        
        try:
            model = YOLO(model_path)
            
            if test_image_path and os.path.exists(test_image_path):
                results = model(test_image_path)
                print(f"Model {model_name} tested successfully on {test_image_path}")
                
                # Print detections
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        print(f"Detected {len(boxes)} objects:")
                        for box in boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = model.names[class_id]
                            print(f"  - {class_name}: {confidence:.2f}")
            else:
                print(f"Model {model_name} loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"Failed to test model {model_name}: {e}")
            return False
    
    def recommend_model(self):
        """Recommend the best model for Duckietown"""
        print("\nModel Recommendations for Duckietown:")
        print("-" * 40)
        
        print("🚀 Quick Start (Recommended):")
        print("   Model: coco_nano")
        print("   Pros: Ready to use, detects people/vehicles")
        print("   Cons: May miss Duckietown-specific objects")
        print()
        
        print("⚡ Best Performance:")
        print("   Model: duckietown_custom")
        print("   Pros: Optimized for Duckietown objects")
        print("   Cons: Requires training data and time")
        print()
        
        print("🎯 Balanced Option:")
        print("   Model: coco_small")
        print("   Pros: Better accuracy than nano, still fast")
        print("   Cons: Slightly slower than nano")

def main():
    manager = YOLOModelManager()
    
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 model_manager.py [list|download|test|recommend] [model_name]")
        return
    
    command = sys.argv[1]
    
    if command == 'list':
        manager.list_models()
    
    elif command == 'download':
        if len(sys.argv) < 3:
            print("Please specify model name")
            manager.list_models()
            return
        model_name = sys.argv[2]
        manager.download_model(model_name)
    
    elif command == 'test':
        if len(sys.argv) < 3:
            print("Please specify model name")
            return
        model_name = sys.argv[2]
        test_image = sys.argv[3] if len(sys.argv) > 3 else None
        manager.test_model(model_name, test_image)
    
    elif command == 'recommend':
        manager.recommend_model()
    
    else:
        print("Unknown command. Use: list, download, test, or recommend")

if __name__ == '__main__':
    main()