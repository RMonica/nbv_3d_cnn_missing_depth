cmake_minimum_required(VERSION 3.0.2)
project(nbv_3d_cnn_real_image)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  cv_bridge
  eigen_conversions
  nbv_3d_cnn_real_image_msgs
  nbv_3d_cnn_evaluate_view_msgs
  roscpp
  rospy
  sensor_msgs
  visualization_msgs
  rmonica_voxelgrid_common
  pcl_conversions
  message_serialization
)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

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
#  LIBRARIES nbv_3d_cnn_real_image
#  CATKIN_DEPENDS actionlib actionlib_msgs cv_bridge eigen_conversions nbv_3d_cnn_real_image_msgs roscpp rospy sensor_msgs visualization_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(generate_virtual_views
  src/generate_virtual_views.cpp
  )

add_dependencies(generate_virtual_views ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(generate_virtual_views
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(cnn_real_image_projection
  src/cnn_real_image_projection.cpp
  )

add_dependencies(cnn_real_image_projection ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(cnn_real_image_projection
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(cnn_real_image_evaluate
  src/cnn_real_image_evaluate.cpp
  )

add_dependencies(cnn_real_image_evaluate ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(cnn_real_image_evaluate
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(cnn_real_image_eval_gain
  src/cnn_real_image_eval_gain.cpp
  )

add_dependencies(cnn_real_image_eval_gain ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(cnn_real_image_eval_gain
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

#############
## Install ##
#############

#############
## Testing ##
#############

