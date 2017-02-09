## Synopsis

This library instantiates a directional vesselness filter that enhances threadlike structures. This filter is used in the conference paper ["Automatic initialization and dynamic tracking of surgical suture threads"](http://ieeexplore.ieee.org/document/7139853/) by Jackson et al. This package includes both a CPU and GPU implementation.

## Code Example

The basic Robot OS call for the filters are: 
```sh
rosrun vesselness_image_filter vesselness_image_filter_gpu_node     #GPU based vesselness filter
rosrun vesselness_image_filter vesselness_image_filter_cpu_node      #CPU based vesselness filter
rosrun vesselness_image_filter vesselness_image_filter_cpu_bw_node  #CPU based vesselness filter
```
By defaults the filters subscribe to the ros topic: "image_in" and publish to the ros topic, "image_thin".

The viewers can be called as follows: 
```sh
rosrun vesselness_image_filter vesselness_image_viewer_gpu_node    #GPU based vesselness viewer
rosrun vesselness_image_filter vesselness_image_viewer_cpu_node    #CPU based vesselness viewer
```
By defaults the viewers subscribe to the ros topic: "image_in" and display the image using OpenCV.

The filter has several parameters that can be configured using rqt_reconfigure.

## Installation

The CPU based filter and viewer will operate with ROS and OpenCV. If GPU functionality is desired, OpenCV must be recompiled with CUDA support on a CUDA capable GPU.

## License

This package is licensed under the BSD-3-Clause. 
Please direct all inquiries to rcj33@case.edu.