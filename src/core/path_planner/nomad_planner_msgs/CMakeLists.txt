cmake_minimum_required(VERSION 3.0.2)
project(nomad_planner_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  geometry_msgs
  nav_msgs
)

add_service_files(
  FILES
  MakePlan.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  nav_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime geometry_msgs nav_msgs
)
