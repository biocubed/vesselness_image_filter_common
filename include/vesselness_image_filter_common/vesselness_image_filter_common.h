/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Case Western Reserve University
 *    Russell C Jackson <rcj33@case.edu>
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Case Western Reserve University, nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */


#ifndef VESSELNESS_IMAGE_FILTER_COMMON_VESSELNESS_IMAGE_FILTER_COMMON_H
#define VESSELNESS_IMAGE_FILTER_COMMON_VESSELNESS_IMAGE_FILTER_COMMON_H

#include <vector>
#include <stdio.h>
#include <iostream>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dynamic_reconfigure/server.h>
#include <vesselness_image_filter/vesselness_params_Config.h>
/**
 * This file introduces the abstract base class for the vesselness_image_filter nodes.
 * The Base Class is VesselnessNodeBase.
 * The base class gets derived into two forms: a CPU and a GPU based algorithm for image processing.
 * The base class is meant to handle the ROS communication functionality as well as initialize the 
 * members and methods required by both classes.
 */


/**
 * The struct gaussParam stores the information needed to construct a simple gaussian filter kernel.
 * The members are the variance (float) and the length of a side (int). The length of the side should always be odd.
 * and at least 3.
 * @todo 
 * Make this struct a class.
 */
struct gaussParam
{
  /**
   * @brief the kernel variance
   */
  float variance;

  /**
   * @brief the kernel side length 
   *
   * This should be odd and greater than or equal to 3.
   */
  int side;

  /**
   * @brief element wise costructor
   *
   * constructs a gaussParam object from the variance and the side length
   * @param variance_ : the new variance 
   * @param side_ : the length of a side (odd and > 2.)
   */
  gaussParam(float variance_, int side_):
    variance(variance_),
    side(side_)
  {
    if (side < 3)
    {
      side = 3;
    }
    if (side%2 == 0)
    {
      side++;
    }
  }

  /**
   * @brief copy costructor
   *
   * constructs a gaussParam object from another object.
   * @param src__ : the gaussParam to be copied.
   */
  gaussParam(const gaussParam &src_):
    variance(src_.variance),
    side(src_.side)
  {
  }

  /**
   * @brief assignment operator
   *
   * Assigns an gaussParam to match the values of another gaussParam
   * @param src_ : the gaussParam to be copied
   */
  gaussParam& operator = (const gaussParam& src_)
  {
    variance = src_.variance;
    side     = src_.side;
    return *this;
  }
};


/**
 * 
 * 
 * The struct segmentThinParam stores all of the variables for the vesselness segmentation
 * This includes the pre (and post) gaussian filter parameters as well as the beta,c parameters
 */
struct segmentThinParam
{
  /**
   * @brief the initial filter settings
   */
  gaussParam hessProcess;

  /**
   * @brief the post processing settings
   */
  gaussParam postProcess;

  /**
   * @brief the beta parameter:
   */
  float betaParam;

  /**
   * @brief the c parameter
   */
  float cParam;

  /**
   * @brief copy costructor
   *
   * constructs a segmentThinParam object from another segmentThinParam object.
   * @param src__ : the segmentThinParam object to be copied.
   */
  segmentThinParam(const segmentThinParam & src_):
    hessProcess(src_.hessProcess),
    postProcess(src_.postProcess),
    betaParam(src_.betaParam),
    cParam(src_.cParam)
  {
  }

  /**
   * @brief element wise costructor
   *
   * constructs a segmentThinParam object from the elements
   * @param hessProcess_ : the new hessProcess
   * @param postProcess_ : the new postProcess
   * @param betaParam_ : the new betaParam
   * @param cParam_ : the new cParam
   */
  segmentThinParam(gaussParam hessProcess_, gaussParam postProcess_, float betaParam_, float cParam_):
    hessProcess(hessProcess_),
    postProcess(postProcess_),
    betaParam(betaParam_),
    cParam(cParam_)
  {
  }

  /**
   * @brief assignment operator
   *
   * Assigns an segmentThinParam to match the values of another segmentThinParam
   * @param src_ : the segmentThinParam to be copied
   */
  segmentThinParam& operator= (const segmentThinParam& src_)
  {
    hessProcess = src_.hessProcess;
    postProcess = src_.postProcess;
    betaParam   = src_.betaParam;
    cParam      = src_.cParam;
    return *this;
  }
};


/**
 * @brief gaussFnc is an inline gaussian pdf computation
 *
 * @param var : the variance
 * @param x   : the x position
 * @param y   : the y position
 *
 * @returns : The pdf value of the gaussian function
 */
inline float gaussFnc(float var, float x, float y)
{
  return 1/(3.1415*2*var)*exp(-x*x/(2*var)-y*y/(2*var));
}


/**
 * @brief This is an abstract base class for the vesselness filter. 
 *
 * The VesselnessNodeBase abstract class defines members which are common to both the CPU and the GPU instantiation of
 * the vessel_image_filter object. The base class defines all of the public members which are common accross instantiation.
 */
class VesselnessNodeBase
{
private:
  /**
   * @brief The internal node handle.
   * 
   * @TODO, make this a reference.
   */
  ros::NodeHandle nh_;

  /**
   * @brief The settings subscriber
   * 
   * @TODO, check & remove if obsolete
   */
  ros::Subscriber settings_sub_;

  /**
   * @brief image transport member
   */
  image_transport::ImageTransport it_;

  /**
   * @brief image publisher of the segmented image.
   */
  image_transport::Publisher image_pub_;

  /**
   * @brief image subscriber for the input image.
   */
  image_transport::Subscriber image_sub_;

  /**
   * @brief dynamic reconfigure server
   */
  dynamic_reconfigure::Server<vesselness_image_filter::vesselness_params_Config> srv_;

  /**
   * @brief dynamic reconfigure callback type
   */
  dynamic_reconfigure::Server<vesselness_image_filter::vesselness_params_Config>::CallbackType f_;

  /**
   * @brief dynamic reconfigure callback function
   *
   * @param new configuration
   * @param integer
   */
  void paramCallback(vesselness_image_filter::vesselness_params_Config &, uint32_t);

protected:
  /**
   * @brief cv::Mat container for the output image
   */
  cv::Mat outputImage_;

  /**
   * @brief allocated filter image size
   */
  cv::Size imgAllocSize_;

  /**
   * @brief the filter parameter settings.
   */
  segmentThinParam filterParameters_;

  /**
   * @brief the number of output channels.
   */
  int outputChannels_;

  /**
   * @brief an indicator that the kernels are allocated
   */
  bool kernelReady_;

public:
  /**
   * @brief The default conststructor
   *
   * @param subscription path,
   * @param publisher path.
   *
   * After initializing the publisher and subscriber,
   * the memory allocation functions are called.
   */
  explicit VesselnessNodeBase(const char*, const char*);

  /**
   * @brief An unused default destructor.
   */
  ~VesselnessNodeBase()
  {};

  /**
   * @brief This unimplemented function sets the Mat Mask used for masked segmentation
   *
   * @param the mask image
   *
   * @todo implement this feature.
   */
  // virtual void setImageMask(const Mat &) = 0;

  /**
   * @brief The incoming image topic callback (for new images)
   */
  void  imgTopicCallback(const sensor_msgs::ImageConstPtr&);

  /**
   * @brief This functions performs the image segmentation
   *
   * @param input image
   * @param Output image
   * 
   * The instatiation determines both the method and architecture of the segmentation.
   */
  virtual void segmentImage(const cv::Mat&, cv::Mat &) = 0;

  /**
   * @brief Allocate the image memory
   * 
   * @param image size to allocate
   * 
   * @return the allocated image size.
   *
   * The memory allocation is different on the GPU and the CPU.
   */
  virtual cv::Size allocateMem(const cv::Size&) = 0;

  /**
   * @brief deallocate the image memory
   * 
   * The memory deallocation is different on the GPU and the CPU.
   */
  virtual void deallocateMem() = 0;

  /**
   * @brief initialize the gaussian filter kernels.
   */
  virtual void initKernels() = 0;


  /**
   * @brief set the number of output channels
   *
   * @param the new number of output channels
   */
  void setOutputChannels(int outputChannels_);

  /**
   * @brief set the parameter server.
   */
  void setParamServer();
};


#endif  // VESSELNESS_IMAGE_FILTER_COMMON_VESSELNESS_IMAGE_FILTER_COMMON_H
