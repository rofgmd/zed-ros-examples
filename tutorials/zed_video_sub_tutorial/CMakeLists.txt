cmake_minimum_required(VERSION 3.5)

project(zed_video_sub_tutorial)

## Generate symbols for IDE indexer
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# if CMAKE_BUILD_TYPE is not specified, take 'Release' as default
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_BUILD_TYPE)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS 
    roscpp 
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    cv_bridge  # Required for converting ROS images to OpenCV
)

## Declare a catkin package
catkin_package()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS} 
  ${catkin_INCLUDE_DIRS}
)

## Script
install(PROGRAMS
  scripts/zed2i_sub.py
  scripts/zed2i_sub_tensorRT.py
  scripts/publish_svo.py
  scripts/control_car.py
  scripts/go_line.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## Build 
add_executable(zed_video_sub src/zed_video_sub_tutorial.cpp)
add_executable(zed2i_video_sub src/zed2i_video_sub_tutorial.cpp)

target_link_libraries(zed_video_sub ${catkin_LIBRARIES})
target_link_libraries(zed2i_video_sub ${catkin_LIBRARIES})