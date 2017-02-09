/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016 Case Western Reserve University
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



#include "vesselness_image_filter_gpu/vesselness_viewer_kernels.h"

void convertSegmentImageGPU(const cv::Mat &src, cv::Mat &dst)
{
  // Initialize the stream:
  cv::cuda::Stream streamInfo;

  // allocate the pagelocked memory:
  cv::cuda::HostMem srcMatMem(src.size(), CV_32FC2);
  cv::cuda::HostMem dstMatMem(src.size(), CV_8UC3);

  cv::Mat srcMatLocked = srcMatMem.createMatHeader();

  src.copyTo(srcMatLocked);

  cv::cuda::GpuMat srcG;
  srcG.upload(srcMatLocked, streamInfo);
  cv::cuda::GpuMat scaledSrc, scaledSrcU;

  // scale the image from [0-3.141] x [0-1] to [0-255] x [0 - 255].
  cv::cuda::multiply(srcG, cv::Scalar(179/3.14159, 255.0), scaledSrc, 1, CV_32FC2, streamInfo);

  // convert the image from a floating point image
  // to an 8 bit unsigned char image.
  scaledSrc.convertTo(scaledSrcU, CV_8UC2, streamInfo);

  // split the 2 channel image into a vector of images.
  // add a third channel of 1s.
  // combine the images into 1 3 channel image.
  std::vector< cv::cuda::GpuMat > splitList;
  cv::cuda::split(scaledSrcU, splitList, streamInfo);
  cv::cuda::GpuMat onesG(src.rows, src.cols, CV_8UC1);
  splitList.push_back(onesG);
  splitList[2].setTo(127, streamInfo);

  cv::cuda::GpuMat outputG, hsvG;
  cv::cuda::merge(splitList, hsvG, streamInfo);

  // convert the image to a color image from an HSV image.
  cv::cuda::cvtColor(hsvG, outputG, CV_HSV2BGR, 0, streamInfo);

  // download the image to the host memory.
  outputG.download(dstMatMem, streamInfo);

  streamInfo.waitForCompletion();

  cv::Mat tempDst = dstMatMem.createMatHeader();
  dst = tempDst.clone();

  // clean up the memory
  srcMatMem.release();
  srcG.release();
  scaledSrc.release();
  scaledSrcU.release();
  onesG.release();
  for (int i = 0; i < splitList.size(); i++)
  {
    splitList[i].release();
  }
  outputG.release();
  hsvG.release();
  dstMatMem.release();
  // All done
}
