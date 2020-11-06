cmake_minimum_required(VERSION 3.0.2)
project(points_process_test)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_ros
  sensor_msgs
)

catkin_package(
INCLUDE_DIRS include
CATKIN_DEPENDS roscpp sensor_msgs pcl_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})

add_executable(${PROJECT_NAME}_node src/points_process_test_node.cpp src/points_process_test_core.cpp)


target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

