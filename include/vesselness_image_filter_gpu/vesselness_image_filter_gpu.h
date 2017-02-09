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
 *     disclaimer in the documentation and/or other cv::Materials provided
 *     with the distribution.
 *   * Neither the name of Case Western Reserve Univeristy, nor the names of its
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

#ifndef VESSELNESS_IMAGE_FILTER_GPU_VESSELNESS_IMAGE_FILTER_GPU_H
#define VESSELNESS_IMAGE_FILTER_GPU_VESSELNESS_IMAGE_FILTER_GPU_H

#include <vesselness_image_filter_gpu/vesselness_image_filter_kernels.h>
#include <vesselness_image_filter_common/vesselness_image_filter_common.h>




/**
 * @brief The vesselness node class which extends the vesselness base class.
 *
 * This class is based on GPU processing.
 */
class VesselnessNodeGPU: public VesselnessNodeBase
{
private:
  /**
   * @brief the GPU side input image.
   */
  cv::cuda::GpuMat inputG_;

  /**
   * @brief the GPU side gray image.
   */
  cv::cuda::GpuMat inputGrayG_;

  /**
   * @brief the GPU side floating point image.
   */
  cv::cuda::GpuMat inputFloat255G_;

  /**
   * @brief the GPU side image of 1's
   */
  cv::cuda::GpuMat ones_;

  /**
   * @brief the GPU side float image (scaled to 0-1).
   */
  cv::cuda::GpuMat inputFloat1G_;

  /**
   * @brief the GPU side convolved Hessian XX Mat.
   */
  cv::cuda::GpuMat cXX_;

  /**
   * @brief the GPU side convolved Hessian XY Mat.
   */
  cv::cuda::GpuMat cXY_;

  /**
   * @brief the GPU side convolved Hessian YY Mat.
   */
  cv::cuda::GpuMat cYY_;

  /**
   * @brief the GPU side preOutput image.
   */
  cv::cuda::GpuMat preOutput_;

  /**
   * @brief the GPU side output image.
   */
  cv::cuda::GpuMat outputG_;

  /**
   * @brief the GPU side gaussian Mat.
   */
  cv::cuda::GpuMat gaussG_;


  /**
   * @brief the CPU side page locked memory
   */
  cv::cuda::HostMem dstMatMem_;

  /**
   * @brief the output matrix.
   */
  cv::Mat dstMats_;

  /**
   * @brief the CPU side hessian XX Mat.
   */
  cv::Mat tempCPU_XX_;

  /**
   * @brief the CPU side hessian XY Mat.
   */
  cv::Mat tempCPU_XY_;

  /**
   * @brief the CPU side hessian YY Mat.
   */
  cv::Mat tempCPU_YY_;


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
   * Deallocates the GPU memory and the page-locked memory.
   * This is needed to prevent memory leaks.
   */
  void deallocateMem();


  /**
   * @brief segments the image with a 2 channel output.
   *
   * @param input image
   * @param output image
   */
  void segmentImage(const cv::Mat &src, cv::Mat &dst);

public:
  /**
   * @brief initialize the required gaussian kernels.
   */
  void  initKernels() override;


  /**
   * @brief The explicit constructor
   *
   * @param the subscribed image topic.
   * @param the published image topic.
   */
  explicit VesselnessNodeGPU(const char* subscriptionChar, const char* publicationChar);

  /**
   * @brief the deconstructor.
   *
   * This function cleans up the class memory
   */
  ~VesselnessNodeGPU();
};


#endif  // VESSELNESS_IMAGE_FILTER_GPU_VESSELNESS_IMAGE_FILTER_GPU_H
