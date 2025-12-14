cmake_minimum_required(VERSION 3.5)
project(dynamic_rviz_config)
find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package(
  CATKIN_DEPENDS urdf xacro
)

catkin_install_python(PROGRAMS
  scripts/rviz_generate.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)