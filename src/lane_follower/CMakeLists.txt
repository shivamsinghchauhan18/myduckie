cmake_minimum_required(VERSION 3.0.2)
project(lane_follower)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  tf2
  tf2_ros
  visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS 
    rospy 
    std_msgs 
    sensor_msgs 
    geometry_msgs 
    cv_bridge 
    image_transport 
    tf2 
    tf2_ros 
    visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Make Python scripts executable
catkin_install_python(PROGRAMS
  scripts/advanced_lane_detector.py
  scripts/enhanced_lane_controller.py
  scripts/lane_performance_monitor.py
  scripts/lane_system_monitor.py
  scripts/lane_debug_system.py
  scripts/neural_lane_detector.py
  scripts/mpc_lane_controller.py
  scripts/sensor_fusion_system.py
  scripts/adaptive_learning_system.py
  scripts/lane_obstacle_detector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)