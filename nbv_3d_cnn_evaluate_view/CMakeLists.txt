cmake_minimum_required(VERSION 3.0.2)
project(nbv_3d_cnn_evaluate_view)

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  geometry_msgs
  roscpp
  rmonica_voxelgrid_common
  nbv_3d_cnn_evaluate_view_msgs
  eigen_conversions
  actionlib
  message_serialization
  cv_bridge
  pcl_conversions
  render_robot_urdf
  nbv_3d_cnn_msgs
  nbv_3d_cnn_real_image_msgs
)

find_package(OpenCL REQUIRED)
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=110)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)

find_package(Eigen3 REQUIRED)

find_package(PCL REQUIRED)

#find_package(yaml-cpp REQUIRED)

################################################
## Declare ROS messages, services and actions ##
################################################

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES nbv_3d_cnn_evaluate_view
#  CATKIN_DEPENDS nbv_3d_cnn_view_filter_msgs roscpp
  DEPENDS Eigen3
)

###########
## Build ##
###########

add_custom_command(
  COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_LIST_DIR}/cmake/evaluate_view_opencl.cmake"
  MAIN_DEPENDENCY "${CMAKE_CURRENT_LIST_DIR}/src/evaluate_view_opencl.cl"
  DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/opencl_cpp_common.h"
  OUTPUT "${CMAKE_CURRENT_LIST_DIR}/src/evaluate_view_opencl.cl.cpp"
  VERBATIM
  )

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${YAML_INCLUDE_DIRS}
)

add_executable(nbv_3d_cnn_evaluate_view
  src/evaluate_view.cpp
  src/evaluate_view_filter.cpp
  src/autocomplete_predict.cpp
  src/image_predict.cpp
  src/projection_image_predict.cpp
  src/evaluate_view_opencl.cpp
  src/evaluate_view_opencl.cl.cpp
)

add_dependencies(nbv_3d_cnn_evaluate_view ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(nbv_3d_cnn_evaluate_view
  ${catkin_LIBRARIES}
  ${OpenCL_LIBRARIES}
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

add_executable(test_evaluate_view
  src/evaluate_view_test.cpp
)

add_dependencies(test_evaluate_view ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(test_evaluate_view
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

#############
## Install ##
#############

#############
## Testing ##
#############

