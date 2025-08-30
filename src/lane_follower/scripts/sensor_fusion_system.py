#!/usr/bin/env python3

"""
Advanced Sensor Fusion System for Lane Following
Combines camera, IMU, wheel odometry, and AprilTags for robust localization
"""

import rospy
import numpy as np
from scipy.linalg import block_diag
from geometry_msgs.msg import Point, PointStamped, Twist, PoseStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, Header, Float32MultiArray
from nav_msgs.msg import Odometry
try:
    import tf2_ros
    import tf2_geometry_msgs
    TF2_AVAILABLE = True
except ImportError:
    rospy.logwarn("TF2 not available. Sensor fusion will work without coordinate transformations.")
    TF2_AVAILABLE = False
from collections import deque
import threading
import time

class ExtendedKalmanFilter:
    """Extended Kalman Filter for sensor fusion"""
    
    def __init__(self, state_dim, obs_dim):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # State: [x, y, theta, v, lateral_offset, heading_error, curvature]
        self.x = np.zeros(state_dim)  # State estimate
        self.P = np.eye(state_dim) * 0.1  # Covariance matrix
        
        # Process noise
        self.Q = np.eye(state_dim) * 0.01
        
        # Measurement noise (will be updated based on sensor)
        self.R = np.eye(obs_dim) * 0.1
        
        # Time step
        self.dt = 0.05  # 20 Hz
        
    def predict(self, control_input):
        """Prediction step"""
        # State transition model (bicycle model + lane tracking)
        F = self.get_state_transition_matrix()
        
        # Control input matrix
        B = self.get_control_matrix()
        
        # Predict state
        self.x = F @ self.x + B @ control_input
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, measurement, measurement_function, measurement_jacobian, R_sensor):
        """Update step with measurement"""
        # Predicted measurement
        h = measurement_function(self.x)
        
        # Innovation
        y = measurement - h
        
        # Measurement Jacobian
        H = measurement_jacobian(self.x)
        
        # Innovation covariance
        S = H @ self.P @ H.T + R_sensor
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P
    
    def get_state_transition_matrix(self):
        """Get state transition matrix F"""
        F = np.eye(self.state_dim)
        
        # Position updates
        F[0, 3] = self.dt * np.cos(self.x[2])  # x += v * cos(theta) * dt
        F[1, 3] = self.dt * np.sin(self.x[2])  # y += v * sin(theta) * dt
        
        return F
    
    def get_control_matrix(self):
        """Get control matrix B"""
        B = np.zeros((self.state_dim, 2))  # 2 control inputs: [v_cmd, steering]
        
        B[3, 0] = 1.0  # Velocity control
        B[2, 1] = self.dt  # Steering affects heading
        
        return B

class SensorFusionSystem:
    def __init__(self):
        rospy.init_node('sensor_fusion_system', anonymous=True)
        
        # Extended Kalman Filter
        state_dim = 7  # [x, y, theta, v, lateral_offset, heading_error, curvature]
        obs_dim = 3   # Variable based on sensor
        self.ekf = ExtendedKalmanFilter(state_dim, obs_dim)
        
        # Publishers
        self.fused_pose_pub = rospy.Publisher('/lane_follower/fused_pose', PoseStamped, queue_size=1)
        self.fused_lane_pose_pub = rospy.Publisher('/lane_follower/fused_lane_pose', PointStamped, queue_size=1)
        self.fusion_confidence_pub = rospy.Publisher('/lane_follower/fusion_confidence', Float32, queue_size=1)
        self.fusion_debug_pub = rospy.Publisher('/lane_follower/fusion_debug', Float32MultiArray, queue_size=1)
        
        # Subscribers for different sensors
        self.setup_subscribers()
        
        # Sensor data storage
        self.camera_data = None
        self.imu_data = None
        self.odometry_data = None
        self.apriltag_data = None
        
        # Sensor timestamps for synchronization
        self.last_camera_time = None
        self.last_imu_time = None
        self.last_odom_time = None
        self.last_apriltag_time = None
        
        # Sensor reliability tracking
        self.sensor_health = {
            'camera': deque(maxlen=50),
            'imu': deque(maxlen=50),
            'odometry': deque(maxlen=50),
            'apriltag': deque(maxlen=50)
        }
        
        # Fusion parameters
        self.max_sensor_delay = 0.2  # Maximum acceptable sensor delay (seconds)
        self.min_confidence_threshold = 0.3
        
        # TF2 for coordinate transformations (if available)
        if TF2_AVAILABLE:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        else:
            self.tf_buffer = None
            self.tf_listener = None
        
        # Fusion timer
        self.fusion_timer = rospy.Timer(rospy.Duration(0.05), self.fusion_loop)  # 20 Hz
        
        # Threading
        self._processing = False
        
        rospy.loginfo("Advanced Sensor Fusion System initialized")
    
    def setup_subscribers(self):
        """Setup subscribers for all sensor inputs"""
        # Camera-based lane detection
        self.neural_pose_sub = rospy.Subscriber('/lane_follower/neural_lane_pose', 
                                               PointStamped, self.camera_callback)
        self.lane_confidence_sub = rospy.Subscriber('/lane_follower/lane_confidence', 
                                                   Float32, self.camera_confidence_callback)
        
        # IMU data
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        
        # Wheel odometry
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odometry_callback)
        
        # AprilTag detections
        self.apriltag_sub = rospy.Subscriber('/apriltag_detector/detections', 
                                           Float32MultiArray, self.apriltag_callback)
        
        # Control commands (for prediction) - try both message types
        try:
            from duckietown_msgs.msg import Twist2DStamped
            self.cmd_sub = rospy.Subscriber('/car_cmd_switch_node/cmd', 
                                           Twist2DStamped, self.control_callback_2d)
        except ImportError:
            self.cmd_sub = rospy.Subscriber('/cmd_vel', 
                                           Twist, self.control_callback)
        
        self.current_control = np.array([0.0, 0.0])  # [v, omega]
    
    def camera_callback(self, msg):
        """Process camera-based lane detection"""
        self.camera_data = {
            'lateral_offset': msg.point.x,
            'heading_error': msg.point.y,
            'lane_found': msg.point.z > 0.5,
            'timestamp': msg.header.stamp,
            'confidence': 0.8  # Default confidence, updated by confidence callback
        }
        self.last_camera_time = rospy.Time.now()
        
        # Update sensor health
        self.sensor_health['camera'].append(1.0 if self.camera_data['lane_found'] else 0.0)
    
    def camera_confidence_callback(self, msg):
        """Update camera confidence"""
        if self.camera_data:
            self.camera_data['confidence'] = msg.data
    
    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = {
            'angular_velocity': msg.angular_velocity.z,
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y],
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'timestamp': msg.header.stamp
        }
        self.last_imu_time = rospy.Time.now()
        
        # Update sensor health (IMU is usually reliable)
        self.sensor_health['imu'].append(1.0)
    
    def odometry_callback(self, msg):
        """Process wheel odometry"""
        self.odometry_data = {
            'position': [msg.pose.pose.position.x, msg.pose.pose.position.y],
            'velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y],
            'angular_velocity': msg.twist.twist.angular.z,
            'timestamp': msg.header.stamp
        }
        self.last_odom_time = rospy.Time.now()
        
        # Update sensor health
        velocity_magnitude = np.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        self.sensor_health['odometry'].append(1.0 if velocity_magnitude > 0.01 else 0.5)
    
    def apriltag_callback(self, msg):
        """Process AprilTag detections"""
        if len(msg.data) >= 6:  # At least one detection with pose
            self.apriltag_data = {
                'detections': np.array(msg.data).reshape(-1, 6),  # [id, x, y, z, roll, pitch, yaw]
                'timestamp': rospy.Time.now()
            }
            self.last_apriltag_time = rospy.Time.now()
            
            # Update sensor health
            self.sensor_health['apriltag'].append(1.0)
        else:
            self.sensor_health['apriltag'].append(0.0)
    
    def control_callback(self, msg):
        """Update current control input for prediction"""
        self.current_control = np.array([msg.linear.x, msg.angular.z])
    
    def control_callback_2d(self, msg):
        """Update current control input for prediction (Twist2DStamped)"""
        self.current_control = np.array([msg.v, msg.omega])
    
    def fusion_loop(self, event):
        """Main sensor fusion loop"""
        if self._processing:
            return
            
        self._processing = True
        
        try:
            # Prediction step
            self.ekf.predict(self.current_control)
            
            # Update with available sensor measurements
            self.fuse_camera_measurement()
            self.fuse_imu_measurement()
            self.fuse_odometry_measurement()
            self.fuse_apriltag_measurement()
            
            # Publish fused results
            self.publish_fused_state()
            
            # Publish debug information
            self.publish_debug_info()
            
        except Exception as e:
            rospy.logerr(f"Sensor fusion error: {str(e)}")
        finally:
            self._processing = False
    
    def fuse_camera_measurement(self):
        """Fuse camera-based lane detection"""
        if not self.is_sensor_data_valid('camera'):
            return
            
        # Measurement: [lateral_offset, heading_error]
        z = np.array([
            self.camera_data['lateral_offset'],
            self.camera_data['heading_error']
        ])
        
        # Measurement function (direct observation of lane pose)
        def h_camera(x):
            return np.array([x[4], x[5]])  # lateral_offset, heading_error from state
        
        # Measurement Jacobian
        def H_camera(x):
            H = np.zeros((2, self.ekf.state_dim))
            H[0, 4] = 1.0  # lateral_offset
            H[1, 5] = 1.0  # heading_error
            return H
        
        # Measurement noise (based on confidence)
        confidence = self.camera_data['confidence']
        R_camera = np.eye(2) * (0.1 / max(confidence, 0.1))  # Lower noise for higher confidence
        
        # Update EKF
        self.ekf.update(z, h_camera, H_camera, R_camera)
    
    def fuse_imu_measurement(self):
        """Fuse IMU measurements"""
        if not self.is_sensor_data_valid('imu'):
            return
            
        # Measurement: [angular_velocity]
        z = np.array([self.imu_data['angular_velocity']])
        
        # Measurement function
        def h_imu(x):
            return np.array([x[3] * np.tan(self.current_control[1]) / 0.1])  # v * tan(delta) / wheelbase
        
        # Measurement Jacobian
        def H_imu(x):
            H = np.zeros((1, self.ekf.state_dim))
            H[0, 3] = np.tan(self.current_control[1]) / 0.1  # d/dv
            return H
        
        # Measurement noise
        R_imu = np.array([[0.05]])  # IMU is usually quite accurate
        
        # Update EKF
        self.ekf.update(z, h_imu, H_imu, R_imu)
    
    def fuse_odometry_measurement(self):
        """Fuse wheel odometry"""
        if not self.is_sensor_data_valid('odometry'):
            return
            
        # Measurement: [velocity]
        velocity_magnitude = np.sqrt(
            self.odometry_data['velocity'][0]**2 + self.odometry_data['velocity'][1]**2
        )
        z = np.array([velocity_magnitude])
        
        # Measurement function
        def h_odom(x):
            return np.array([x[3]])  # Direct velocity measurement
        
        # Measurement Jacobian
        def H_odom(x):
            H = np.zeros((1, self.ekf.state_dim))
            H[0, 3] = 1.0  # d/dv
            return H
        
        # Measurement noise
        R_odom = np.array([[0.02]])  # Wheel odometry is quite accurate for velocity
        
        # Update EKF
        self.ekf.update(z, h_odom, H_odom, R_odom)
    
    def fuse_apriltag_measurement(self):
        """Fuse AprilTag-based localization"""
        if not self.is_sensor_data_valid('apriltag'):
            return
            
        # Use AprilTag detections for absolute position correction
        detections = self.apriltag_data['detections']
        
        for detection in detections:
            tag_id, x, y, z, roll, pitch, yaw = detection
            
            # Known AprilTag positions (would be loaded from map)
            known_positions = self.get_apriltag_positions()
            
            if tag_id in known_positions:
                # Calculate expected position based on tag detection
                tag_world_pos = known_positions[tag_id]
                
                # Measurement: [x_position, y_position]
                z = np.array([tag_world_pos[0] - x, tag_world_pos[1] - y])
                
                # Measurement function
                def h_apriltag(x_state):
                    return np.array([x_state[0], x_state[1]])  # Direct position measurement
                
                # Measurement Jacobian
                def H_apriltag(x_state):
                    H = np.zeros((2, self.ekf.state_dim))
                    H[0, 0] = 1.0  # d/dx
                    H[1, 1] = 1.0  # d/dy
                    return H
                
                # Measurement noise (AprilTags can be noisy at distance)
                distance = np.sqrt(x**2 + y**2 + z**2)
                noise_factor = max(0.1, distance * 0.05)  # Noise increases with distance
                R_apriltag = np.eye(2) * noise_factor
                
                # Update EKF
                self.ekf.update(z, h_apriltag, H_apriltag, R_apriltag)
    
    def is_sensor_data_valid(self, sensor_name):
        """Check if sensor data is valid and recent"""
        current_time = rospy.Time.now()
        
        if sensor_name == 'camera':
            return (self.camera_data is not None and 
                   self.last_camera_time is not None and
                   (current_time - self.last_camera_time).to_sec() < self.max_sensor_delay)
        elif sensor_name == 'imu':
            return (self.imu_data is not None and 
                   self.last_imu_time is not None and
                   (current_time - self.last_imu_time).to_sec() < self.max_sensor_delay)
        elif sensor_name == 'odometry':
            return (self.odometry_data is not None and 
                   self.last_odom_time is not None and
                   (current_time - self.last_odom_time).to_sec() < self.max_sensor_delay)
        elif sensor_name == 'apriltag':
            return (self.apriltag_data is not None and 
                   self.last_apriltag_time is not None and
                   (current_time - self.last_apriltag_time).to_sec() < self.max_sensor_delay)
        
        return False
    
    def get_apriltag_positions(self):
        """Get known AprilTag positions (would be loaded from map file)"""
        # Example positions - in practice, load from configuration
        return {
            0: [0.0, 0.0],    # Tag 0 at origin
            1: [1.0, 0.0],    # Tag 1 at 1m along x-axis
            2: [0.0, 1.0],    # Tag 2 at 1m along y-axis
            # Add more tags as needed
        }
    
    def calculate_fusion_confidence(self):
        """Calculate overall fusion confidence"""
        confidences = []
        
        # Camera confidence
        if self.is_sensor_data_valid('camera'):
            camera_health = np.mean(self.sensor_health['camera']) if self.sensor_health['camera'] else 0
            confidences.append(camera_health * 0.4)  # 40% weight
        
        # IMU confidence
        if self.is_sensor_data_valid('imu'):
            imu_health = np.mean(self.sensor_health['imu']) if self.sensor_health['imu'] else 0
            confidences.append(imu_health * 0.2)  # 20% weight
        
        # Odometry confidence
        if self.is_sensor_data_valid('odometry'):
            odom_health = np.mean(self.sensor_health['odometry']) if self.sensor_health['odometry'] else 0
            confidences.append(odom_health * 0.2)  # 20% weight
        
        # AprilTag confidence
        if self.is_sensor_data_valid('apriltag'):
            apriltag_health = np.mean(self.sensor_health['apriltag']) if self.sensor_health['apriltag'] else 0
            confidences.append(apriltag_health * 0.2)  # 20% weight
        
        return sum(confidences) if confidences else 0.0
    
    def publish_fused_state(self):
        """Publish fused state estimate"""
        # Publish full pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        
        pose_msg.pose.position.x = self.ekf.x[0]
        pose_msg.pose.position.y = self.ekf.x[1]
        pose_msg.pose.position.z = 0.0
        
        # Convert heading to quaternion
        theta = self.ekf.x[2]
        pose_msg.pose.orientation.z = np.sin(theta / 2)
        pose_msg.pose.orientation.w = np.cos(theta / 2)
        
        self.fused_pose_pub.publish(pose_msg)
        
        # Publish lane pose
        lane_pose_msg = PointStamped()
        lane_pose_msg.header = pose_msg.header
        lane_pose_msg.point.x = self.ekf.x[4]  # lateral_offset
        lane_pose_msg.point.y = self.ekf.x[5]  # heading_error
        lane_pose_msg.point.z = 1.0  # Always assume lane found with fusion
        
        self.fused_lane_pose_pub.publish(lane_pose_msg)
        
        # Publish confidence
        confidence = self.calculate_fusion_confidence()
        self.fusion_confidence_pub.publish(Float32(confidence))
    
    def publish_debug_info(self):
        """Publish debug information"""
        debug_data = list(self.ekf.x) + list(np.diag(self.ekf.P))  # State + uncertainties
        debug_msg = Float32MultiArray()
        debug_msg.data = debug_data
        self.fusion_debug_pub.publish(debug_msg)

if __name__ == '__main__':
    try:
        fusion_system = SensorFusionSystem()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass