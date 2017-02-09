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


#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vesselness_image_filter_cpu/vesselness_filter_node_cpu_bw.h>


// constructor.
VesselnessNodeCPUBW::VesselnessNodeCPUBW(const char* subscriptionChar, const char* publicationChar):
  VesselnessNodeBase(subscriptionChar, publicationChar)
{
  // initialize the kernels
  outputChannels_ = 1;
  initKernels();
  setParamServer();
}

// initializes the hessian kernels.
void VesselnessNodeCPUBW::initKernels()
{
  double var(filterParameters_.hessProcess.variance);

  // Allocate the matrices
  gaussKernel_XX_ = cv::Mat(filterParameters_.hessProcess.side, filterParameters_.hessProcess.side, CV_32F);
  gaussKernel_XY_ = cv::Mat(filterParameters_.hessProcess.side, filterParameters_.hessProcess.side, CV_32F);
  gaussKernel_YY_ = cv::Mat(filterParameters_.hessProcess.side, filterParameters_.hessProcess.side, CV_32F);

  int kSizeEnd = static_cast<int> (filterParameters_.hessProcess.side-1)/2;

  for (int ix = -kSizeEnd; ix < kSizeEnd+1; ix++)
  {
    for (int iy = -kSizeEnd; iy < kSizeEnd+1; iy++)
    {
      float ixD = static_cast<float>  (ix);
      float iyD = static_cast<float>  (iy);

      gaussKernel_XX_.at<float>(iy+kSizeEnd, ix+kSizeEnd) =
        (ixD*ixD)/(var*var)*gaussFnc(var, ixD, iyD)-1/(var)*gaussFnc(var, ixD, iyD);

      gaussKernel_YY_.at<float>(iy+kSizeEnd, ix+kSizeEnd) =
        (iyD*iyD)/(var*var)*gaussFnc(var, ixD, iyD)-1/(var)*gaussFnc(var, ixD, iyD);

      gaussKernel_XY_.at<float>(iy+kSizeEnd, ix+kSizeEnd) = (iyD*ixD)/(var*var)*gaussFnc(var, ixD, iyD);
    }
  }
}


// image segmentation:
void  VesselnessNodeCPUBW::segmentImage(const cv::Mat& src, cv::Mat& dst)
{
  float betaParam(filterParameters_.betaParam);
  float cParam(filterParameters_.cParam);

  // Actual process segmentation code:
  cv::cvtColor(src, grayImage_, CV_BGR2GRAY);
  grayImage_.convertTo(grayFloat_, CV_32FC1, 1.0, 0.0);

  // @TODO Replace the C style pointer casting with the reinterpret_cast

  float *grayFloatPtr = reinterpret_cast<float*> (grayFloat_.data);
  grayFloat_ /= 255.0;


  // Gaussian Blur filtering (XX,XY,YY);
  cv::filter2D(grayFloat_, grayImage_xx_, -1, gaussKernel_XX_);
  cv::filter2D(grayFloat_, grayImage_xy_, -1, gaussKernel_XY_);
  cv::filter2D(grayFloat_, grayImage_yy_, -1, gaussKernel_YY_);

  std::cout << "Blurred images" << std::endl;

  // Compute the number of total pixels
  int pixCount = grayImage_xx_.rows*grayImage_xx_.cols;


  // pull out the image data pointers
  float *gradPtr_xx = reinterpret_cast<float*> (grayImage_xx_.data);
  float *gradPtr_yx = reinterpret_cast<float*> (grayImage_xy_.data);
  float *gradPtr_xy = reinterpret_cast<float*> (grayImage_xy_.data);
  float *gradPtr_yy = reinterpret_cast<float*> (grayImage_yy_.data);

  preOutput_.create(grayImage_xx_.rows, grayImage_xx_.cols, CV_32FC1);
  char* preOutputImagePtr = reinterpret_cast<char*> (preOutput_.data);

  int preOutputImageStep0 =  preOutput_.step[0];
  int preOutputImageStep1 =  preOutput_.step[1];


  char* inputMaskPtr = reinterpret_cast<char*> (imageMask_.data);

  int inputMaskStep0 =  imageMask_.step[0];
  int inputMaskStep1 =  imageMask_.step[1];


  // evaluate the hessian eigen vectors
  // use that information to generate the vesselness
  for (int i = 0; i < pixCount; i++)
  {
    int xPos =  i%grayImage_xx_.cols;
    int yPos =  static_cast<int> (floor(static_cast<float> (i)/static_cast<float> (grayImage_.cols)));

    // construct the output pointer
    float* prePointer =  reinterpret_cast<float*>
      (preOutputImagePtr+ preOutputImageStep0*yPos + preOutputImageStep1*xPos);

    // If the mask is valid, use it to select points
    if (imageMask_.rows == imageMask_.rows && imageMask_.cols == preOutput_.cols)
    {
      char* maskVal = (inputMaskPtr+ inputMaskStep0*yPos + inputMaskStep1*xPos);

      if (maskVal[0] == 0)
      {
        prePointer[0] = 0.0;
        continue;
      }
    }  // if(inputMask.rows == preOutput_.rows && inputMask.cols == preOutput_.cols)

    float vMag(0.0);
    float v_y(0.0);
    float v_x(1.0);
    float a2(0.0);

    float det(gradPtr_xx[i]*gradPtr_yy[i]-gradPtr_yx[i]*gradPtr_yx[i]);
    float b(-gradPtr_xx[i]-gradPtr_yy[i]);
    float c(det);
    float descriminant(sqrt(b*b-4*c));

    float eig0;
    float eig1;
    float r_Beta;


    // Check if the eigenvalue is repeated.
    if (descriminant > 0.000000001)
    {
      eig0 = (-b+descriminant)/(2);
      eig1 = (-b-descriminant)/(2);

      r_Beta = eig0/eig1;

      // find the dominant eigenvector:
      if (abs(r_Beta) > 1.0)  // indicates that eig0 is larger.
      {
        r_Beta = 1/r_Beta;
      }
    }  // if(descriminant > 0.000000001)
    else
    {
      eig0 = eig1 = -b/2;
      r_Beta = 1.0;
      v_y = 0.00;
      v_x = 1.0;
    }

    // In this formulation, the image peak is 1.0.
    vMag = exp(-r_Beta*r_Beta/(betaParam))*(1-exp(-(eig0*eig0+eig1*eig1)/(cParam)));

    // error handling
    if (!(vMag <= 1) || !(vMag >= 0))
    {
      float test = 1;
    }
    prePointer[0] = vMag;
  }
    // Finally blur the final image using a gaussian.
    dst.create(src.size(), src.type());
    cv::Size kernelSize(filterParameters_.postProcess.side, filterParameters_.postProcess.side);
    cv::GaussianBlur(preOutput_, dst, kernelSize,
      filterParameters_.postProcess.variance, filterParameters_.postProcess.variance);

    return;
}

cv::Size VesselnessNodeCPUBW::allocateMem(const cv::Size &sizeIn)
{
  imgAllocSize_ = sizeIn;
  outputImage_.create(imgAllocSize_, CV_32FC1);
  return imgAllocSize_;
}
