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

#ifndef VESSELNESS_IMAGE_FILTER_CPU_VESSELNESS_FILTER_NODE_CPU_BW_H
#define VESSELNESS_IMAGE_FILTER_CPU_VESSELNESS_FILTER_NODE_CPU_BW_H

#include <vesselness_image_filter_common/vesselness_image_filter_common.h>


/**
 * @brief convert a bw segmented vesselness image to a displayable format.
 *
 * @param the input image
 * @param the output image
 */
void convertSegmentImageCPUBW(const cv::Mat &src, cv::Mat &dst);

/**
 * @brief The BW vesselness node class which extends the vesselness base class.
 */
class VesselnessNodeCPUBW: public VesselnessNodeBase
{
private:
  /**
   * @brief the grey input image.
   */
  cv::Mat grayImage_;

  /**
   * @brief the grey floating point input image.
   */
  cv::Mat grayFloat_;

  /**
   * @brief the XX grey image.
   */
  cv::Mat grayImage_xx_;

  /**
   * @brief the XY grey image.
   */
  cv::Mat grayImage_xy_;

  /**
   * @brief the YY grey image.
   */
  cv::Mat grayImage_yy_;

  /**
   * @brief the preoutput image.
   */
  cv::Mat preOutput_;

  /**
   * @brief the XX gaussian kernel.
   */
  cv::Mat gaussKernel_XX_;

  /**
   * @brief the XY gaussian kernel.
   */
  cv::Mat gaussKernel_XY_;

  /**
   * @brief the YY gaussian kernel.
   */
  cv::Mat gaussKernel_YY_;

  /**
   * @brief the image filtering mask.
   */
  cv::Mat imageMask_;

  /**
   * @brief initialize the required gaussian kernels.
   */
  void  initKernels() override;

  /**
   * @brief The allocate memory function. 
   *
   * required for class instantiation.
   *
   * @param size of matrices to allocate.
   */
  cv::Size allocateMem(const cv::Size &sizeIn);

  /**
   * @brief deallocates the class memory
   *
   * required for class instantiation but is blank.
   */
  void deallocateMem()
  {};

  /**
   * @brief segments the image using a BW output.
   *
   * @param input image
   * @param output image
   */
  void segmentImage(const cv::Mat &, cv::Mat &);

  /**
   * @brief updates the image processing kernels.
   *
   * Not yet implemented
   *
   * @param the new kernel parameters.
   */
  void updateKernels(const segmentThinParam &);

public:
  /**
   * @brief The explicit constructor
   *
   * @param the subscribed image topic.
   * @param the published image topic.
   */
  explicit VesselnessNodeCPUBW(const char*, const char *);

  /**
   * @brief the deconstructor.
   */
  ~VesselnessNodeCPUBW()
  {};
};

#endif  // VESSELNESS_IMAGE_FILTER_CPU_VESSELNESS_FILTER_NODE_CPU_BW_H
