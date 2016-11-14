cmake_minimum_required(VERSION 2.8.3)
project(vesselness_image_filter)
find_package(OpenCV REQUIRED)


find_package(CUDA)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  message_generation
  cwru_opencv_common
  dynamic_reconfigure
)


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
    LIBRARIES vesselness_lib
    CATKIN_DEPENDS dynamic_reconfigure image_transport cv_bridge
)

SET(LIB_TYPE STATIC) 

if(CUDA_FOUND)

SET(CUDA_NVCC_FLAGS "-arch=compute_30" CACHE STRING "nvcc flags" FORCE)

endif()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${catkin_INCLUDES}
  ${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake
)


add_library(vesselness_image_filter_common_lib
   src/vesselness_image_filter_common.cpp
 )




target_link_libraries(vesselness_image_filter_common_lib
   ${catkin_LIBRARIES}
 )

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)




add_library(vesselness_image_filter_cpu_lib
   src/cpu/vesselness_image_filter_cpu.cpp
 )

add_library(vesselness_lib
   src/cpu/vesselness_lib.cpp
 )

add_library(vesselness_image_filter_bw_cpu_lib
   src/cpu/vesselness_image_filter_cpu_bw.cpp
 )

if(CUDA_FOUND)

#CUDA_ADD_LIBRARY(vesselness_image_filter_gpu_kernels_lib src_cu/vesselness_image_filter_kernels.cu)
CUDA_ADD_LIBRARY(vesselness_image_filter_gpu_lib src/gpu/cu/vesselness_image_filter_gpu.cu)

CUDA_ADD_LIBRARY(vesselness_viewer_gpu_lib src/gpu/cu/vesselness_viewer_kernel.cu)


endif()

ADD_EXECUTABLE(vesselness_image_filter_cpu_node src/cpu/vesselness_image_filter_cpu_node.cpp)

ADD_EXECUTABLE(vesselness_image_filter_cpu_bw_node src/cpu/vesselness_image_filter_cpu_bw_node.cpp)

ADD_EXECUTABLE(vesselness_image_filter_cpu_viewer_node src/cpu/vesselness_image_viewer_node.cpp)


if(CUDA_FOUND)

CUDA_ADD_LIBRARY(vesselness_viewer_gpu_lib src_cu/vesselness_viewer_kernel.cu)
CUDA_ADD_EXECUTABLE(vesselness_image_filter_gpu_node src/vesselness_image_filter_gpu_node.cpp)
CUDA_ADD_EXECUTABLE(vesselness_viewer_gpu_node src/vesselness_image_viewer_gpu_node.cpp)

endif()


add_dependencies(vesselness_image_filter_common_lib ${PROJECT_NAME}_gencfg)
add_dependencies(vesselness_image_filter_common_lib vesselness_image_filter_gencfg)

add_dependencies(vesselness_image_filter_cpu_node vesselness_image_filter_gencfg)
add_dependencies(vesselness_image_filter_cpu_node vesselness_image_filter_gencpp)

add_dependencies(vesselness_lib vesselness_image_filter_gencpp)
add_dependencies(vesselness_lib ${PROJECT_NAME}_gencfg)

add_dependencies(vesselness_lib ${PROJECT_NAME}_gencfg)

add_dependencies(vesselness_image_filter_common_lib ${PROJECT_NAME}_gencfg)

target_link_libraries(vesselness_image_filter_cpu_lib
  vesselness_image_filter_common_lib
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
 )

target_link_libraries(vesselness_image_filter_cpu_node
   vesselness_image_filter_common_lib
   vesselness_image_filter_cpu_lib
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
 )


target_link_libraries(vesselness_image_filter_bw_cpu_lib
    vesselness_image_filter_common_lib
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    )

target_link_libraries(vesselness_image_filter_cpu_bw_node
   vesselness_image_filter_common_lib
   vesselness_image_filter_bw_cpu_lib
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
 )

target_link_libraries(vesselness_lib
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
 )

target_link_libraries(vesselness_image_filter_cpu_viewer_node
   vesselness_lib
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
 )


if(CUDA_FOUND)

target_link_libraries(vesselness_viewer_gpu_node
   vesselness_viewer_gpu_lib
   ${CUDA_LIBRARIES}
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
)

target_link_libraries(vesselness_viewer_gpu_lib
   ${CUDA_LIBRARIES}
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
)

#target_link_libraries(vesselness_image_filter_gpu_kernels_lib
#   ${CUDA_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   ${catkin_LIBRARIES}
#)

target_link_libraries(vesselness_image_filter_gpu_lib
   #vesselness_image_filter_gpu_kernels_lib
   ${CUDA_LIBRARIES}
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
)

target_link_libraries(vesselness_image_filter_gpu_node
   vesselness_image_filter_common_lib
   #vesselness_image_filter_gpu_kernels_lib
   vesselness_image_filter_gpu_lib
   ${CUDA_LIBRARIES}
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
)


endif()