cmake_minimum_required(VERSION 3.10)
project(nomad_planner)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_core
  pluginlib
  costmap_2d
  geometry_msgs
  nav_msgs
  nomad_planner_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES nomad_planner
  CATKIN_DEPENDS roscpp nav_core pluginlib costmap_2d geometry_msgs nav_msgs nomad_planner_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(nomad_planner
  src/nomad_planner.cpp
)

target_compile_features(nomad_planner PUBLIC cxx_std_17)

target_link_libraries(nomad_planner
  ${catkin_LIBRARIES}
)

install(TARGETS nomad_planner
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(FILES nomad_planner_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
