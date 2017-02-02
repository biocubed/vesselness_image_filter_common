cmake_minimum_required(VERSION 2.8.3)
project(vesselness_image_filter)

# Find system packages...
find_package(OpenCV REQUIRED)
find_package(CUDA)


# Find catkin packages...
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  message_generation
  # cwru_opencv_common
  dynamic_reconfigure
)

# Add messages
add_message_files(
  FILES
  vesselness_params.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

generate_dynamic_reconfigure_options(
  cfg/vesselness_params_.cfg
  #...
)

# package definition...
catkin_package(
    INCLUDE_DIRS include
    LIBRARIES vesselness_lib
    CATKIN_DEPENDS dynamic_reconfigure image_transport cv_bridge
)

# Validate that cuda is available and built into opencv.
if(CUDA_FOUND)
  message("...Found CUDA ${CUDA_VERSION}")
  if(OpenCV_CUDA_VERSION) 
    message("...Opencv compiled with cuda version ${OpenCV_CUDA_VERSION}")
  endif()
endif()


# If necessary set the nvidia build flags for the GPU architecture.
if(CUDA_FOUND AND OpenCV_CUDA_VERSION)
  SET(CUDA_NVCC_FLAGS "-arch=compute_30" CACHE STRING "nvcc flags" FORCE)
endif()


# Include directory path
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${catkin_INCLUDES}
  ${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake
)

# Base library for the filtering applications 
# provides an abstract class.
add_library(vesselness_image_filter_common_lib
  src/vesselness_image_filter_common.cpp
)

target_link_libraries(vesselness_image_filter_common_lib
  ${catkin_LIBRARIES}
)

add_dependencies(vesselness_image_filter_common_lib ${PROJECT_NAME}_gencfg)

# Directional CPU processing filter.
add_library(vesselness_image_filter_cpu_lib
  src/cpu/vesselness_image_filter_cpu.cpp
)

target_link_libraries(vesselness_image_filter_cpu_lib
  vesselness_image_filter_common_lib
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_dependencies(vesselness_image_filter_cpu_lib ${PROJECT_NAME}_gencfg)

# B & W output vesselness filter
add_library(vesselness_image_filter_bw_cpu_lib
  src/cpu/vesselness_image_filter_cpu_bw.cpp
)

target_link_libraries(vesselness_image_filter_bw_cpu_lib
  vesselness_image_filter_common_lib
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_dependencies(vesselness_image_filter_bw_cpu_lib ${PROJECT_NAME}_gencfg)

# Exported library of vesselness functions (CPU).
add_library(vesselness_lib
  src/cpu/vesselness_lib.cpp
)

target_link_libraries(vesselness_lib
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_dependencies(vesselness_lib ${PROJECT_NAME}_gencfg)


# Add the cuda libraries.
if(CUDA_FOUND AND OpenCV_CUDA_VERSION)

  # The dirtectional GPU filter
  cuda_add_library(vesselness_image_filter_gpu_lib
    src/gpu/cu/vesselness_image_filter_gpu.cu
  )

  target_link_libraries(vesselness_image_filter_gpu_lib
    vesselness_image_filter_common_lib
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
  )

  add_dependencies(vesselness_image_filter_gpu_lib ${PROJECT_NAME}_gencfg)


  # GPU Viewer Library
  cuda_add_library(vesselness_viewer_gpu_lib
    src/gpu/cu/vesselness_viewer_kernel.cu
  )

  target_link_libraries(vesselness_viewer_gpu_lib
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
  )

endif()

# Finished the libraries: add the executables.

# cpu based filter executable:
add_executable(vesselness_image_filter_cpu_node
  src/cpu/vesselness_image_filter_cpu_node.cpp
)

target_link_libraries(vesselness_image_filter_cpu_node
  vesselness_image_filter_common_lib
  vesselness_image_filter_cpu_lib
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

# B & W filter executable.
add_executable(vesselness_image_filter_cpu_bw_node
  src/cpu/vesselness_image_filter_cpu_bw_node.cpp
)

target_link_libraries(vesselness_image_filter_cpu_bw_node
  vesselness_image_filter_common_lib
  vesselness_image_filter_bw_cpu_lib
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

# CPU viewer executable:
add_executable(vesselness_image_viewer_cpu_node
  src/cpu/vesselness_image_viewer_node.cpp
)

target_link_libraries(vesselness_image_viewer_cpu_node
  vesselness_lib
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

# only add the cuda executables if necessary:
if(CUDA_FOUND AND OpenCV_CUDA_VERSION)
  
  # GPU filter:
  cuda_add_executable(vesselness_image_filter_gpu_node
    src/gpu/cpp/vesselness_image_filter_gpu_node.cpp
  )

  target_link_libraries(vesselness_image_filter_gpu_node
    vesselness_image_filter_common_lib
    vesselness_image_filter_gpu_lib
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
  )   

  # GPU viewer:
  cuda_add_executable(vesselness_image_viewer_gpu_node
    src/gpu/cpp/vesselness_image_viewer_gpu_node.cpp
  )

  target_link_libraries(vesselness_image_viewer_gpu_node
    vesselness_viewer_gpu_lib
    vesselness_image_filter_common_lib
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
  )

endif()

# installation path:
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)