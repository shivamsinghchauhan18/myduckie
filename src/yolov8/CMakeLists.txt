cmake_minimum_required(VERSION 3.0.2)
project(yolov8)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  duckietown_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs cv_bridge duckietown_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/yolo_detector.py
  scripts/distance_calculator.py
  scripts/avoidance_controller.py
  scripts/emergency_recovery.py
  scripts/yolo_client.py
  scripts/safety_monitor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)