cmake_minimum_required(VERSION 3.0.2)
project(myapriltags)

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
  scripts/apriltag_detector.py
  scripts/apriltag_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)