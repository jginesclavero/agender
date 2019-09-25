cmake_minimum_required(VERSION 2.8.3)
project(agender_ros_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

add_message_files(
 FILES
 PersonDescriptors.msg
 PeopleDescriptors.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS std_msgs message_runtime
)
