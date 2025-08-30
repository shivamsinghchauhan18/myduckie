#!/usr/bin/env python3

"""
Adaptive Learning System for Lane Following
Uses machine learning to automatically tune parameters and improve performance
"""

import rospy
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Bool, Float32, Float32MultiArray
from collections import deque
import threading
import time

class PerformanceMetrics:
    """Track and calculate performance metrics"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        
        # Performance history
        self.lateral_errors = deque(maxlen=window_size)
        self.heading_errors = deque(maxlen=window_size)
        self.velocities = deque(maxlen=window_size)
        self.control_smoothness = deque(maxlen=window_size)
        self.lane_detection_rates = deque(maxlen=window_size)
        
        # Environmental conditions
        self.lighting_conditions = deque(maxlen=window_size)
        self.curve_conditions = deque(maxlen=window_size)
        self.speed_conditions = deque(maxlen=window_size)
        
    def update(self, lateral_error, heading_error, velocity, control_change, 
               lane_detected, lighting, curvature):
        """Update performance metrics"""
        self.lateral_errors.append(abs(lateral_error))
        self.heading_errors.append(abs(heading_error))
        self.velocities.append(velocity)
        self.control_smoothness.append(control_change)
        self.lane_detection_rates.append(1.0 if lane_detected else 0.0)
        self.lighting_conditions.append(lighting)
        self.curve_conditions.append(abs(curvature))
        self.speed_conditions.append(velocity)
    
    def get_performance_score(self):
        """Calculate overall performance score"""
        if len(self.lateral_errors) < 10:
            return 0.5  # Default score
            
        # Component scores
        lateral_score = max(0, 1.0 - np.mean(self.lateral_errors) * 2)
        heading_score = max(0, 1.0 - np.mean(self.heading_errors) * 1.5)
        smoothness_score = max(0, 1.0 - np.mean(self.control_smoothness) * 5)
        detection_score = np.mean(self.lane_detection_rates)
        
        # Weighted overall score
        overall_score = (0.4 * lateral_score + 0.3 * heading_score + 
                        0.2 * smoothness_score + 0.1 * detection_score)
        
        return overall_score
    
    def get_feature_vector(self):
        """Get current environmental features for learning"""
        if len(self.lateral_errors) < 5:
            return np.zeros(10)  # Default feature vector
            
        features = [
            np.mean(self.lighting_conditions),      # Average lighting
            np.std(self.lighting_conditions),       # Lighting variability
            np.mean(self.curve_conditions),         # Average curvature
            np.std(self.curve_conditions),          # Curvature variability
            np.mean(self.speed_conditions),         # Average speed
            np.std(self.speed_conditions),          # Speed variability
            np.mean(self.lateral_errors),           # Current lateral performance
            np.mean(self.heading_errors),           # Current heading performance
            np.mean(self.control_smoothness),       # Current smoothness
            np.mean(self.lane_detection_rates)      # Current detection rate
        ]
        
        return np.array(features)

class ParameterOptimizer:
    """Optimize control parameters using machine learning"""
    
    def __init__(self):
        # Parameter ranges for optimization
        self.parameter_ranges = {
            'kp_lateral': (0.5, 5.0),
            'ki_lateral': (0.0, 0.5),
            'kd_lateral': (0.0, 2.0),
            'kp_heading': (0.5, 3.0),
            'ki_heading': (0.0, 0.2),
            'kd_heading': (0.0, 1.0),
            'target_speed': (0.1, 0.4),
            'smoothing_factor': (0.5, 0.95)
        }
        
        # Machine learning models for each parameter
        self.models = {}
        self.scalers = {}
        
        # Training data
        self.training_features = []
        self.training_targets = {}
        
        # Initialize models
        self.initialize_models()
        
        # Load existing models if available
        self.load_models()
    
    def initialize_models(self):
        """Initialize ML models for each parameter"""
        for param_name in self.parameter_ranges.keys():
            self.models[param_name] = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            self.scalers[param_name] = StandardScaler()
            self.training_targets[param_name] = []
    
    def add_training_sample(self, features, parameters, performance_score):
        """Add training sample for learning"""
        self.training_features.append(features)
        
        for param_name, param_value in parameters.items():
            if param_name in self.training_targets:
                # Weight the parameter by performance score
                weighted_param = param_value * performance_score
                self.training_targets[param_name].append(weighted_param)
    
    def train_models(self):
        """Train ML models with collected data"""
        if len(self.training_features) < 20:  # Need minimum samples
            return False
            
        X = np.array(self.training_features)
        
        for param_name in self.parameter_ranges.keys():
            if len(self.training_targets[param_name]) == len(self.training_features):
                y = np.array(self.training_targets[param_name])
                
                # Scale features
                X_scaled = self.scalers[param_name].fit_transform(X)
                
                # Train model
                self.models[param_name].fit(X_scaled, y)
                
                rospy.loginfo(f"Trained model for {param_name}")
        
        return True
    
    def predict_optimal_parameters(self, features):
        """Predict optimal parameters for current conditions"""
        optimal_params = {}
        
        for param_name in self.parameter_ranges.keys():
            try:
                # Scale features
                features_scaled = self.scalers[param_name].transform([features])
                
                # Predict parameter value
                predicted_value = self.models[param_name].predict(features_scaled)[0]
                
                # Clip to valid range
                min_val, max_val = self.parameter_ranges[param_name]
                optimal_params[param_name] = np.clip(predicted_value, min_val, max_val)
                
            except Exception as e:
                # Use default value if prediction fails
                min_val, max_val = self.parameter_ranges[param_name]
                optimal_params[param_name] = (min_val + max_val) / 2
                
        return optimal_params
    
    def save_models(self, filepath):
        """Save trained models to file"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'training_features': self.training_features,
                'training_targets': self.training_targets
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            rospy.loginfo(f"Saved adaptive models to {filepath}")
            
        except Exception as e:
            rospy.logerr(f"Failed to save models: {str(e)}")
    
    def load_models(self, filepath=None):
        """Load trained models from file"""
        if filepath is None:
            filepath = os.path.expanduser("~/lane_following_models.pkl")
            
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.models = model_data['models']
                self.scalers = model_data['scalers']
                self.training_features = model_data['training_features']
                self.training_targets = model_data['training_targets']
                
                rospy.loginfo(f"Loaded adaptive models from {filepath}")
                return True
                
        except Exception as e:
            rospy.logerr(f"Failed to load models: {str(e)}")
            
        return False

class AdaptiveLearningSystem:
    def __init__(self):
        rospy.init_node('adaptive_learning_system', anonymous=True)
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        
        # Parameter optimization
        self.parameter_optimizer = ParameterOptimizer()
        
        # Publishers
        self.optimal_params_pub = rospy.Publisher('/lane_follower/optimal_parameters', 
                                                 Float32MultiArray, queue_size=1)
        self.learning_status_pub = rospy.Publisher('/lane_follower/learning_status', 
                                                  Float32, queue_size=1)
        self.adaptation_info_pub = rospy.Publisher('/lane_follower/adaptation_info', 
                                                  Float32MultiArray, queue_size=1)
        
        # Subscribers
        self.setup_subscribers()
        
        # Current state
        self.current_lateral_error = 0.0
        self.current_heading_error = 0.0
        self.current_velocity = 0.0
        self.current_control_change = 0.0
        self.lane_detected = False
        self.current_lighting = 0.5  # Normalized lighting condition
        self.current_curvature = 0.0
        
        # Current parameters (will be updated by parameter server)
        self.current_parameters = {
            'kp_lateral': 2.0,
            'ki_lateral': 0.1,
            'kd_lateral': 0.5,
            'kp_heading': 1.5,
            'ki_heading': 0.05,
            'kd_heading': 0.3,
            'target_speed': 0.25,
            'smoothing_factor': 0.8
        }
        
        # Learning control
        self.learning_enabled = True
        self.adaptation_interval = 30.0  # Adapt every 30 seconds
        self.last_adaptation_time = rospy.Time.now()
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.parameter_history = deque(maxlen=100)
        
        # Adaptation timer
        self.adaptation_timer = rospy.Timer(rospy.Duration(self.adaptation_interval), 
                                          self.adaptation_loop)
        
        # Save models periodically
        self.save_timer = rospy.Timer(rospy.Duration(300.0), self.save_models_callback)  # Every 5 minutes
        
        rospy.loginfo("Adaptive Learning System initialized - Ready for parameter optimization")
    
    def setup_subscribers(self):
        """Setup subscribers for learning inputs"""
        # Lane following performance - use same topics as other components
        self.lane_pose_sub = rospy.Subscriber('/lane_follower/lane_pose', 
                                             Point, self.lane_pose_callback)
        self.lane_found_sub = rospy.Subscriber('/lane_follower/lane_found', 
                                              Bool, self.lane_found_callback)
        self.curvature_sub = rospy.Subscriber('/lane_follower/lane_curvature', 
                                             Float32, self.curvature_callback)
        
        # Control performance
        self.control_effort_sub = rospy.Subscriber('/lane_follower/control_effort', 
                                                  Float32, self.control_effort_callback)
        
        # Environmental conditions
        self.lighting_sub = rospy.Subscriber('/lane_follower/lighting_condition', 
                                           Float32, self.lighting_callback)
        
        # Performance feedback
        self.performance_sub = rospy.Subscriber('/lane_follower/performance', 
                                               Float32, self.performance_callback)
    
    def lane_pose_callback(self, msg):
        """Update lane pose for learning"""
        self.current_lateral_error = msg.x
        self.current_heading_error = msg.y
        
        # Update performance metrics
        self.update_performance_metrics()
    
    def lane_found_callback(self, msg):
        self.lane_detected = msg.data
    
    def curvature_callback(self, msg):
        self.current_curvature = msg.data
    
    def control_effort_callback(self, msg):
        self.current_control_change = msg.data
    
    def lighting_callback(self, msg):
        self.current_lighting = msg.data
    
    def performance_callback(self, msg):
        """Receive overall performance score"""
        self.performance_history.append(msg.data)
        self.parameter_history.append(self.current_parameters.copy())
        
        # Add training sample
        features = self.performance_metrics.get_feature_vector()
        self.parameter_optimizer.add_training_sample(
            features, self.current_parameters, msg.data
        )
    
    def update_performance_metrics(self):
        """Update performance metrics with current data"""
        self.performance_metrics.update(
            self.current_lateral_error,
            self.current_heading_error,
            self.current_velocity,
            self.current_control_change,
            self.lane_detected,
            self.current_lighting,
            self.current_curvature
        )
    
    def adaptation_loop(self, event):
        """Main adaptation loop"""
        if not self.learning_enabled:
            return
            
        try:
            # Train models with recent data
            if self.parameter_optimizer.train_models():
                rospy.loginfo("Updated parameter optimization models")
            
            # Get current environmental features
            features = self.performance_metrics.get_feature_vector()
            
            # Predict optimal parameters
            optimal_params = self.parameter_optimizer.predict_optimal_parameters(features)
            
            # Gradually adapt parameters (don't change too quickly)
            adapted_params = self.gradual_parameter_adaptation(optimal_params)
            
            # Update current parameters
            self.current_parameters = adapted_params
            
            # Publish optimal parameters
            self.publish_optimal_parameters(adapted_params)
            
            # Publish learning status
            performance_score = self.performance_metrics.get_performance_score()
            self.learning_status_pub.publish(Float32(performance_score))
            
            # Log adaptation
            rospy.loginfo(f"Adapted parameters - Performance: {performance_score:.3f}")
            
        except Exception as e:
            rospy.logerr(f"Adaptation error: {str(e)}")
    
    def gradual_parameter_adaptation(self, optimal_params):
        """Gradually adapt parameters to avoid sudden changes"""
        adapted_params = {}
        adaptation_rate = 0.1  # 10% adaptation per cycle
        
        for param_name, optimal_value in optimal_params.items():
            current_value = self.current_parameters.get(param_name, optimal_value)
            
            # Gradual change
            change = (optimal_value - current_value) * adaptation_rate
            adapted_params[param_name] = current_value + change
            
            # Ensure within valid range
            min_val, max_val = self.parameter_optimizer.parameter_ranges[param_name]
            adapted_params[param_name] = np.clip(adapted_params[param_name], min_val, max_val)
        
        return adapted_params
    
    def publish_optimal_parameters(self, parameters):
        """Publish optimal parameters for other nodes"""
        param_array = Float32MultiArray()
        
        # Pack parameters in known order
        param_order = ['kp_lateral', 'ki_lateral', 'kd_lateral', 'kp_heading', 
                      'ki_heading', 'kd_heading', 'target_speed', 'smoothing_factor']
        
        param_array.data = [parameters[param] for param in param_order]
        self.optimal_params_pub.publish(param_array)
        
        # Publish adaptation info
        features = self.performance_metrics.get_feature_vector()
        adaptation_info = Float32MultiArray()
        adaptation_info.data = list(features) + list(param_array.data)
        self.adaptation_info_pub.publish(adaptation_info)
    
    def save_models_callback(self, event):
        """Periodically save learned models"""
        filepath = os.path.expanduser("~/lane_following_models.pkl")
        self.parameter_optimizer.save_models(filepath)
    
    def get_learning_statistics(self):
        """Get learning system statistics"""
        stats = {
            'training_samples': len(self.parameter_optimizer.training_features),
            'performance_history_length': len(self.performance_history),
            'current_performance': self.performance_metrics.get_performance_score(),
            'learning_enabled': self.learning_enabled
        }
        
        if self.performance_history:
            stats['avg_recent_performance'] = np.mean(list(self.performance_history)[-20:])
            stats['performance_trend'] = np.mean(list(self.performance_history)[-10:]) - np.mean(list(self.performance_history)[-20:-10:])
        
        return stats

if __name__ == '__main__':
    try:
        learning_system = AdaptiveLearningSystem()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass