#!/usr/bin/env python3

"""
Model Predictive Controller (MPC) for Advanced Lane Following
State-of-the-art control using optimization-based approach
"""

import rospy
import numpy as np
from scipy.optimize import minimize
from geometry_msgs.msg import Point, PointStamped, Twist
from std_msgs.msg import Bool, Float32, Header, Float32MultiArray
from duckietown_msgs.msg import Twist2DStamped
import threading
import time
from collections import deque

class VehicleModel:
    """Bicycle model for DuckieBot dynamics"""
    def __init__(self):
        # Vehicle parameters
        self.wheelbase = 0.1  # Distance between front and rear axles (m)
        self.max_speed = 0.4  # Maximum speed (m/s)
        self.max_steering = 1.0  # Maximum steering angle (rad)
        self.dt = 0.1  # Time step (s)
        
    def predict_state(self, state, control, dt=None):
        """Predict next state using bicycle model"""
        if dt is None:
            dt = self.dt
            
        x, y, theta, v = state
        v_cmd, delta = control
        
        # Bicycle model equations
        x_next = x + v * np.cos(theta) * dt
        y_next = y + v * np.sin(theta) * dt
        theta_next = theta + (v / self.wheelbase) * np.tan(delta) * dt
        v_next = v_cmd  # Assume perfect velocity tracking
        
        return np.array([x_next, y_next, theta_next, v_next])
    
    def predict_trajectory(self, initial_state, control_sequence):
        """Predict trajectory over control horizon"""
        trajectory = [initial_state]
        state = initial_state.copy()
        
        for control in control_sequence:
            state = self.predict_state(state, control)
            trajectory.append(state.copy())
            
        return np.array(trajectory)

class MPCLaneController:
    def __init__(self):
        rospy.init_node('mpc_lane_controller', anonymous=True)
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
        self.mpc_debug_pub = rospy.Publisher('/lane_follower/mpc_debug', Float32MultiArray, queue_size=1)
        self.predicted_path_pub = rospy.Publisher('/lane_follower/predicted_path', Float32MultiArray, queue_size=1)
        self.control_effort_pub = rospy.Publisher('/lane_follower/control_effort', Float32, queue_size=1)
        
        # Subscribers - Use SAME topics as enhanced controller
        self.lane_pose_sub = rospy.Subscriber('/lane_follower/lane_pose', Point, self.lane_pose_callback)
        self.lane_found_sub = rospy.Subscriber('/lane_follower/lane_found', Bool, self.lane_found_callback)
        self.lane_center_sub = rospy.Subscriber('/lane_follower/lane_center', Point, self.lane_center_callback)
        self.lane_curvature_sub = rospy.Subscriber('/lane_follower/lane_curvature', Float32, self.curvature_callback)
        self.lane_coefficients_sub = rospy.Subscriber('/lane_follower/lane_coefficients', Float32MultiArray, self.coefficients_callback)
        
        # Vehicle model
        self.vehicle_model = VehicleModel()
        
        # MPC parameters
        self.horizon = rospy.get_param('~mpc_horizon', 10)  # Prediction horizon
        self.dt = rospy.get_param('~mpc_dt', 0.1)  # Time step
        self.target_speed = rospy.get_param('~target_speed', 0.25)
        
        # State: [x, y, theta, v]
        self.state_dim = 4
        self.control_dim = 2  # [v_cmd, steering_angle]
        
        # Current state
        self.current_state = np.array([0.0, 0.0, 0.0, 0.0])  # x, y, theta, v
        self.lane_found = False
        self.lane_pose = None
        self.lane_curvature = 0.0
        self.lane_coefficients = None
        
        # Smoothing for stable control
        self.previous_steering = 0.0
        self.steering_filter_alpha = 0.7  # Smoothing factor
        
        # MPC weights (tunable parameters)
        self.Q = np.diag([10.0, 1.0, 5.0, 1.0])  # State weights [x, y, theta, v]
        self.R = np.diag([1.0, 5.0])  # Control weights [v_cmd, steering]
        self.Q_terminal = self.Q * 2  # Terminal state weights
        
        # Control constraints
        self.v_min, self.v_max = 0.0, self.vehicle_model.max_speed
        self.delta_min, self.delta_max = -self.vehicle_model.max_steering, self.vehicle_model.max_steering
        
        # Reference trajectory generation
        self.reference_generator = ReferenceTrajectoryGenerator()
        
        # Performance tracking
        self.solve_times = deque(maxlen=50)
        self.control_history = deque(maxlen=100)
        
        # Control loop
        self.control_timer = rospy.Timer(rospy.Duration(self.dt), self.mpc_control_loop)
        
        # Threading
        self._solving = False
        
        rospy.loginfo("MPC Lane Controller initialized - Advanced optimization-based control")
    
    def lane_pose_callback(self, msg):
        """Update lane pose from detector"""
        self.lane_pose = msg
        
        # Update current state estimate
        lateral_error = msg.x
        heading_error = msg.y
        
        # Simple state estimation (in practice, use Kalman filter)
        self.current_state[0] = lateral_error  # Lateral position
        self.current_state[2] = heading_error  # Heading angle
        
        # Debug logging
        rospy.loginfo_throttle(3, f"MPC received lane pose: lat={lateral_error:.3f}, head={heading_error:.3f}")
    
    def lane_center_callback(self, msg):
        """Update lane center"""
        self.lane_center = msg
    
    def lane_found_callback(self, msg):
        self.lane_found = msg.data
    
    def curvature_callback(self, msg):
        self.lane_curvature = msg.data
    
    def coefficients_callback(self, msg):
        """Receive lane polynomial coefficients"""
        if len(msg.data) >= 6:  # Left and right lane coefficients
            self.lane_coefficients = np.array(msg.data).reshape(2, 3)  # 2 lanes, 3 coefficients each
    
    def mpc_control_loop(self, event):
        """Main MPC control loop"""
        if not self.lane_found:
            rospy.logwarn_throttle(5, "MPC: No lane found, stopping")
            self.publish_safe_stop()
            return
            
        if self.lane_pose is None:
            rospy.logwarn_throttle(5, "MPC: No lane pose data, stopping")
            self.publish_safe_stop()
            return
            
        if self._solving:
            rospy.logwarn_throttle(1, "MPC solver still running, skipping iteration")
            return
        
        # Solve MPC in separate thread for real-time performance
        thread = threading.Thread(target=self._solve_mpc_async)
        thread.daemon = True
        thread.start()
    
    def _solve_mpc_async(self):
        """Solve MPC optimization problem asynchronously"""
        self._solving = True
        
        try:
            start_time = time.time()
            
            # Generate reference trajectory
            reference_trajectory = self.reference_generator.generate_reference(
                self.current_state, self.lane_coefficients, self.lane_curvature, self.horizon, self.dt
            )
            
            # Solve MPC optimization
            optimal_controls = self.solve_mpc_optimization(self.current_state, reference_trajectory)
            
            if optimal_controls is not None:
                # Apply first control action (receding horizon principle)
                v_cmd, steering_cmd = optimal_controls[0]
                
                # Publish control commands
                self.publish_mpc_commands(v_cmd, steering_cmd)
                rospy.loginfo_throttle(3, f"MPC: Published commands v={v_cmd:.3f}, ω={steering_cmd:.3f}")
            else:
                rospy.logwarn_throttle(2, "MPC: No optimal controls found, publishing safe stop")
                self.publish_safe_stop()
                
                # Update state prediction
                self.current_state = self.vehicle_model.predict_state(
                    self.current_state, [v_cmd, steering_cmd]
                )
                
                # Publish debug information
                self.publish_mpc_debug(optimal_controls, reference_trajectory)
            
            # Track performance
            solve_time = time.time() - start_time
            self.solve_times.append(solve_time)
            
            if len(self.solve_times) % 20 == 0:
                avg_solve_time = np.mean(self.solve_times)
                rospy.loginfo(f"MPC avg solve time: {avg_solve_time*1000:.1f}ms")
            
        except Exception as e:
            rospy.logerr(f"MPC solver error: {str(e)}")
            self.publish_safe_stop()
        finally:
            self._solving = False
    
    def solve_mpc_optimization(self, initial_state, reference_trajectory):
        """Solve MPC optimization problem - SIMPLIFIED VERSION"""
        try:
            # FALLBACK: Use simple PID-like control instead of complex optimization
            # This ensures the robot actually moves while we debug the full MPC
            
            lateral_error = initial_state[0]  # x position error
            heading_error = initial_state[2]  # theta error
            
            # Gentler control law - TUNED FOR STABILITY
            Kp_lateral = 0.8  # Reduced from 2.0
            Kp_heading = 0.6  # Reduced from 1.5
            
            # Calculate control commands
            steering_cmd = -Kp_lateral * lateral_error - Kp_heading * heading_error
            
            # More conservative steering limits
            max_steering = 0.4  # Reduced from 1.0 rad
            steering_cmd = np.clip(steering_cmd, -max_steering, max_steering)
            
            # Apply smoothing to prevent oscillation
            steering_cmd = (self.steering_filter_alpha * steering_cmd + 
                          (1 - self.steering_filter_alpha) * self.previous_steering)
            self.previous_steering = steering_cmd
            
            # Smoother speed control
            if abs(lateral_error) > 0.4 or abs(heading_error) > 0.6:
                speed_cmd = 0.12  # Slower for large errors
            elif abs(lateral_error) > 0.2 or abs(heading_error) > 0.3:
                speed_cmd = 0.16  # Medium speed for medium errors
            else:
                speed_cmd = self.target_speed  # Full speed when centered
            
            # Create control sequence (repeat first command for horizon)
            controls = np.zeros((self.horizon, self.control_dim))
            for i in range(self.horizon):
                controls[i, 0] = speed_cmd
                controls[i, 1] = steering_cmd
            
            rospy.loginfo_throttle(2, f"MPC Fallback: lat_err={lateral_error:.3f}, head_err={heading_error:.3f}, steer={steering_cmd:.3f}")
            return controls
            
        except Exception as e:
            rospy.logerr(f"MPC fallback error: {str(e)}")
            return None
    
    def _mpc_cost_function(self, decision_vars, initial_state, reference_trajectory):
        """MPC cost function to minimize"""
        # Reshape decision variables to control sequence
        controls = decision_vars.reshape(self.horizon, self.control_dim)
        
        # Predict trajectory
        predicted_trajectory = self.vehicle_model.predict_trajectory(initial_state, controls)
        
        total_cost = 0.0
        
        # Stage costs
        for k in range(self.horizon):
            # State error
            state_error = predicted_trajectory[k] - reference_trajectory[k]
            state_cost = state_error.T @ self.Q @ state_error
            
            # Control effort
            control_cost = controls[k].T @ self.R @ controls[k]
            
            total_cost += state_cost + control_cost
        
        # Terminal cost
        terminal_error = predicted_trajectory[-1] - reference_trajectory[-1]
        terminal_cost = terminal_error.T @ self.Q_terminal @ terminal_error
        total_cost += terminal_cost
        
        # Smoothness penalty (minimize control changes)
        for k in range(1, self.horizon):
            control_change = controls[k] - controls[k-1]
            smoothness_cost = 0.1 * np.sum(control_change**2)
            total_cost += smoothness_cost
        
        return total_cost
    
    def publish_mpc_commands(self, v_cmd, steering_cmd):
        """Publish MPC control commands"""
        # Create Twist2DStamped message for DuckieBot
        twist_msg = Twist2DStamped()
        twist_msg.header = Header()
        twist_msg.header.stamp = rospy.Time.now()
        twist_msg.v = float(v_cmd)
        twist_msg.omega = float(steering_cmd)
        
        self.cmd_vel_pub.publish(twist_msg)
        
        # Debug logging
        rospy.loginfo_throttle(2, f"MPC Publishing: v={v_cmd:.3f}, omega={steering_cmd:.3f}")
        
        # Track control effort
        control_effort = np.sqrt(v_cmd**2 + steering_cmd**2)
        self.control_effort_pub.publish(Float32(control_effort))
        
        # Store in history
        self.control_history.append([v_cmd, steering_cmd])
    
    def publish_mpc_debug(self, optimal_controls, reference_trajectory):
        """Publish MPC debug information"""
        # Flatten control sequence for publishing
        controls_flat = optimal_controls.flatten()
        debug_msg = Float32MultiArray()
        debug_msg.data = controls_flat.tolist()
        self.mpc_debug_pub.publish(debug_msg)
        
        # Publish predicted path
        predicted_trajectory = self.vehicle_model.predict_trajectory(
            self.current_state, optimal_controls
        )
        path_flat = predicted_trajectory.flatten()
        path_msg = Float32MultiArray()
        path_msg.data = path_flat.tolist()
        self.predicted_path_pub.publish(path_msg)
    
    def publish_safe_stop(self):
        """Publish safe stop command"""
        twist_msg = Twist2DStamped()
        twist_msg.header = Header()
        twist_msg.header.stamp = rospy.Time.now()
        twist_msg.v = 0.0
        twist_msg.omega = 0.0
        self.cmd_vel_pub.publish(twist_msg)

class ReferenceTrajectoryGenerator:
    """Generate reference trajectory for MPC"""
    
    def generate_reference(self, current_state, lane_coefficients, curvature, horizon, dt):
        """Generate reference trajectory to follow lane center"""
        reference = np.zeros((horizon + 1, 4))  # +1 for initial state
        
        if lane_coefficients is None:
            # No lane information, maintain current state
            for k in range(horizon + 1):
                reference[k] = current_state.copy()
                reference[k, 3] = 0.2  # Target speed
            return reference
        
        # Extract left and right lane coefficients
        left_coeffs = lane_coefficients[0]
        right_coeffs = lane_coefficients[1]
        
        # Generate reference trajectory along lane center
        for k in range(horizon + 1):
            # Time into the future
            t = k * dt
            
            # Predict position along lane
            y_ahead = t * 0.2  # Assume moving forward at 0.2 m/s
            
            # Calculate lane center at this position
            left_x = np.polyval(left_coeffs, y_ahead)
            right_x = np.polyval(right_coeffs, y_ahead)
            center_x = (left_x + right_x) / 2
            
            # Reference state
            reference[k, 0] = 0.0  # Lateral position (aim for center)
            reference[k, 1] = y_ahead  # Longitudinal position
            reference[k, 2] = 0.0  # Heading (aim straight)
            
            # Adaptive speed based on curvature
            if abs(curvature) > 100:  # High curvature (tight turn)
                reference[k, 3] = 0.15
            elif abs(curvature) > 50:  # Medium curvature
                reference[k, 3] = 0.2
            else:  # Straight or gentle curve
                reference[k, 3] = 0.25
        
        return reference

if __name__ == '__main__':
    try:
        controller = MPCLaneController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass