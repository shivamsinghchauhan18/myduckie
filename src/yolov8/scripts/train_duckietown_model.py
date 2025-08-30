#!/usr/bin/env python3

import os
from ultralytics import YOLO
import yaml

class DuckietownYOLOTrainer:
    def __init__(self):
        self.duckietown_classes = [
            'duckiebot',
            'duckie',
            'cone',
            'stop_sign',
            'person',
            'building',
            'tree',
            'road_sign',
            'barrier',
            'intersection_sign'
        ]
        
    def create_dataset_config(self, dataset_path):
        """Create dataset configuration for Duckietown training"""
        config = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.duckietown_classes),
            'names': {i: name for i, name in enumerate(self.duckietown_classes)}
        }
        
        config_path = os.path.join(dataset_path, 'duckietown.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    def train_model(self, dataset_path, epochs=100, img_size=640):
        """Train YOLOv8 model on Duckietown dataset"""
        # Create dataset config
        config_path = self.create_dataset_config(dataset_path)
        
        # Load pre-trained model for transfer learning
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=img_size,
            batch=16,
            device='cpu',  # Change to 'cuda' if GPU available
            project='duckietown_training',
            name='yolov8_duckietown',
            save_period=10,
            patience=20,
            lr0=0.01,
            augment=True
        )
        
        return results
    
    def validate_model(self, model_path, dataset_path):
        """Validate trained model"""
        model = YOLO(model_path)
        config_path = self.create_dataset_config(dataset_path)
        
        results = model.val(
            data=config_path,
            imgsz=640,
            batch=16,
            device='cpu'
        )
        
        return results

if __name__ == '__main__':
    trainer = DuckietownYOLOTrainer()
    
    # Example usage (adjust paths as needed)
    dataset_path = '/path/to/duckietown_dataset'
    
    if os.path.exists(dataset_path):
        print("Training Duckietown YOLOv8 model...")
        results = trainer.train_model(dataset_path)
        print("Training completed!")
    else:
        print(f"Dataset path {dataset_path} not found.")
        print("Please prepare your Duckietown dataset first.")