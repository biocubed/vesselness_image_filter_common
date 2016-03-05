cmake_minimum_required(VERSION 2.8.3)
project(vesselness_image_filter_common)
find_package(OpenCV REQUIRED)


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  message_generation
)
find_package(catkin REQUIRED dynamic_reconfigure)

add_message_files(
    FILES
    vesselness_params.msg
    )

generate_messages(
    DEPENDENCIES
    std_msgs
)

#
generate_dynamic_reconfigure_options(
  cfg/vesselness_params_.cfg
  #...
)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES vesselness_image_filter_common_lib
    CATKIN_DEPENDS dynamic_reconfigure image_transport cv_bridge
    
)

SET(LIB_TYPE STATIC) 


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake

)


add_library(vesselness_image_filter_common_lib
   src/vesselness_image_filter_common.cpp
 )

add_dependencies(vesselness_image_filter_common_lib ${PROJECT_NAME}_msgs_generate_messages_cpp)
add_dependencies(vesselness_image_filter_common_lib ${PROJECT_NAME}_gencfg)


target_link_libraries(vesselness_image_filter_common_lib
   ${catkin_LIBRARIES}
 )
