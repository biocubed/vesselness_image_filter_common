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

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES vesselness_image_filter_common_lib
    # CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs
    #   DEPENDS system_lib
)

#
generate_dynamic_reconfigure_options(
  cfg/vesselness_params_config.cfg
  #...
)



#SET(CUDA_NVCC_FLAGS "-arch=sm_13" CACHE STRING "nvcc flags" FORCE)
#SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
SET(LIB_TYPE STATIC) 


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)


#CUDA_ADD_LIBRARY(vesselness_image_filter_gpu_kernels src/gpu_src/vesselness_image_filter_kernels.cu)
#CUDA_ADD_LIBRARY(vesselness_image_gpu src/gpu_src/vesselness_image_filter_gpu.cu)


add_library(vesselness_image_filter_common_lib
   src/vesselness_image_filter_common.cpp
 )

 add_dependencies(vesselness_image_filter_common_lib ${PROJECT_NAME}_gencfg)

#cuda_add_library( 

#add_library(vesselness_image_filter
#   src/${PROJECT_NAME}/src/vesselness_image_filter.cpp
# )



#ADD_EXECUTABLE(vesselness_image_filter_cpu_node src/cpu_src/vesselness_image_filter_cpu_node.cpp)

#ADD_EXECUTABLE(vesselness_image_filter_viewer_node src/vesselness_image_viewer_node.cpp)

#ADD_EXECUTABLE(vesselness_image_filter_gpu_node src/gpu_src/vesselness_image_filter_gpu_node.cpp)

#ADD_EXECUTABLE(vesselness_image_filter_cpu src/cpu_src/vesselness_image_filter_gpu.cpp)
#ADD_DEPENDENCIES(beginner_tutorials_node T)

#TARGET_LINK_LIBRARIES(
#   #vesselness_image_filter
#   ${catkin_LIBRARIES}
#  #${PCL_LIBRARIES}
#)


#target_link_libraries(vesselness_image_filter_cpu_node
#   vesselness_image_base
#   vesselness_image_cpu
#   ${catkin_LIBRARIES}
# )

#target_link_libraries(vesselness_image_filter_gpu_node
#   vesselness_image_base
#  vesselness_image_gpu
#  vesselness_image_filter_gpu_kernels
#  ${CUDA_LIBRARIES}
#  ${OpenCV_LIBRARIES}
#  ${catkin_LIBRARIES}
# )

#target_link_libraries(vesselness_image_cpu
#  vesselness_image_base
#   ${catkin_LIBRARIES}
# )

#target_link_libraries(vesselness_image_gpu
# vesselness_image_gpu
# vesselness_image_filter_gpu_kernels
#  vesselness_image_base
#   ${CUDA_LIBRARIES}
#   ${catkin_LIBRARIES}
# )

target_link_libraries(vesselness_image_filter_common_lib
   ${catkin_LIBRARIES}
 )

#target_link_libraries(vesselness_image_filter_viewer_node
#   ${catkin_LIBRARIES}
# )
