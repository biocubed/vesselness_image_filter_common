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



#ifndef VESSELNESS_IMAGE_FILTER_GPU_VESSELNESS_IMAGE_FILTER_KERNELS_H
#define VESSELNESS_IMAGE_FILTER_GPU_VESSELNESS_IMAGE_FILTER_KERNELS_H

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>



/**
 * @brief generates the gaussian derivative XX kernel.
 *
 * @param pointer to the kernel
 * @param kernel variance
 * @param offset to kernel center.
 */
__global__ void genGaussHessKernel_XX(cv::cuda::PtrStepSzf output, float var, int offset);

/**
 * @brief generates the gaussian derivative XY kernel.
 *
 * @param pointer to the kernel
 * @param kernel variance
 * @param offset to kernel center.
 */
__global__ void genGaussHessKernel_XY(cv::cuda::PtrStepSzf output, float var, int offset);

/**
 * @brief generates the gaussian derivative YY kernel.
 *
 * @param pointer to the kernel
 * @param kernel variance
 * @param offset to kernel center.
 */
__global__ void genGaussHessKernel_YY(cv::cuda::PtrStepSzf output, float var, int offset);


/**
 * @brief Computes the vesselness of the image.
 *
 * @param const pointer to the XX hessian
 * @param const pointer to the XY hessian
 * @param const pointer to the YY hessian
 *
 * @param pointer to the output image
 * @param betaParam eigen ratio sensitivity
 * @param cParam eigen mag sensitivity. 
 *
 * This function uses XX, XY, and YY as a 2x2 symmetric matrix and compute its eigenvalues.
 * The ratio and magnitude of these eigenvalues are used to compute the vesselness (2 channel 32 bit floating image).
 * the vesselness has a normal direction between 0 and pi. (channel 0)
 * the vesselness has a magnitude (channel 1)
 */
__global__ void computeVesselness(
  const cv::cuda::PtrStepSzf XX, const cv::cuda::PtrStepSzf XY, const cv::cuda::PtrStepSzf YY,
  cv::cuda::PtrStepSz<float2> output, float betaParam, float cParam);

/**
 * @brief blurs a vesselness image by summing the vesselness vectors scaled with a gaussian kernel.
 *
 * @param pointer to the source image (to be blurred)
 * @param pointer to the destination matrix (blurred)
 * @param pointer to the gaussian kernel matrix.
 * @param gaussian kernel offset.
 */
__global__ void gaussAngBlur(const cv::cuda::PtrStepSz<float2> srcMat, cv::cuda::PtrStepSz<float2> dstMat,
  cv::cuda::PtrStepSzf gMat, int gaussOff);

#endif  // VESSELNESS_IMAGE_FILTER_GPU_VESSELNESS_IMAGE_FILTER_KERNELS_H
