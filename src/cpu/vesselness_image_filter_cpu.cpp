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

#include <vesselness_image_filter_common/vesselness_image_filter_common.h>
#include <vesselness_image_filter_cpu/vesselness_filter_node_cpu.h>


/**
 * @brief blurs an image using an input kernel spec
 *
 * The angle blurring allows the direction to be flipped.
 *
 * @param input image to be blurred
 * @param blurred output image
 * @param kernel parameter
 */
void angleMagBlur(const cv::Mat &src, cv::Mat &dst, const gaussParam inputParam);

// This function is only required to avoid class abstraction
void VesselnessNodeCPU::deallocateMem()
{
}

// initialize the Node and define the kernels accordingly.
VesselnessNodeCPU::VesselnessNodeCPU(const char* subscriptionChar, const char* publicationChar):
  VesselnessNodeBase(subscriptionChar, publicationChar)
{
  outputChannels_ = 2;
  initKernels();
  setParamServer();
}

void VesselnessNodeCPU::initKernels()
{
  double var(filterParameters_.hessProcess.variance);

  // Allocate the matrices
  gaussKernel_XX_ = cv::Mat(filterParameters_.hessProcess.side, filterParameters_.hessProcess.side, CV_32F);
  gaussKernel_XY_ = cv::Mat(filterParameters_.hessProcess.side, filterParameters_.hessProcess.side, CV_32F);
  gaussKernel_YY_ = cv::Mat(filterParameters_.hessProcess.side, filterParameters_.hessProcess.side, CV_32F);

  int kSizeEnd = static_cast<int> ((filterParameters_.hessProcess.side-1)/2);

  // Populate the matrix values:
  for (int ix = -kSizeEnd; ix < kSizeEnd+1; ix++)
  {
    for (int iy = -kSizeEnd; iy < kSizeEnd+1; iy++)
    {
      float ixD = static_cast<float> (ix);
      float iyD = static_cast<float> (iy);

      // assign the three kernels their respective values:

      // gaussian kernel XX derivative
      gaussKernel_XX_.at<float>(iy+kSizeEnd, ix+kSizeEnd) =
        (ixD*ixD)/(var*var)*gaussFnc(var, ixD, iyD)-1/(var)*gaussFnc(var, ixD, iyD);

      // gaussian kernel XY derivative
      gaussKernel_YY_.at<float>(iy+kSizeEnd, ix+kSizeEnd) =
        (iyD*iyD)/(var*var)*gaussFnc(var, ixD, iyD)-1/(var)*gaussFnc(var, ixD, iyD);

      // gaussian kernel YY derivative
      gaussKernel_XY_.at<float>(iy+kSizeEnd, ix+kSizeEnd) =
        (iyD*ixD)/(var*var)*gaussFnc(var, ixD, iyD);
    }
  }
}

void  VesselnessNodeCPU::segmentImage(const cv::Mat& src, cv::Mat& dst)
{
  float betaParam(filterParameters_.betaParam);
  float cParam(filterParameters_.cParam);

  // copnvert the image to a gray scale float 32 with a range of 0-1.
  cv::cvtColor(src, grayImage_, CV_BGR2GRAY);
  grayImage_.convertTo(grayFloat_, CV_32FC1, 1.0, 0.0);
  grayFloat_ /= 255.0;


  // for faster access, pointers are used.
  float *grayFloatPtr = reinterpret_cast<float*> (grayFloat_.data);

  // Gaussian Blur filtering (XX,XY,YY);
  cv::filter2D(grayFloat_, grayImage_xx_, -1, gaussKernel_XX_);
  cv::filter2D(grayFloat_, grayImage_xy_, -1, gaussKernel_XY_);
  cv::filter2D(grayFloat_, grayImage_yy_, -1, gaussKernel_YY_);

  std::cout << "Blurred images" << std::endl;

  // Compute the number of total pixels
  int pixCount = grayImage_xx_.rows*grayImage_xx_.cols;


  // pull out the image data pointers for faster data access.
  float *gradPtr_xx = reinterpret_cast<float*>  (grayImage_xx_.data);
  float *gradPtr_yx = reinterpret_cast<float*>  (grayImage_xy_.data);
  float *gradPtr_xy = reinterpret_cast<float*>  (grayImage_xy_.data);
  float *gradPtr_yy = reinterpret_cast<float*>  (grayImage_yy_.data);

  preOutput_.create(grayImage_xx_.rows, grayImage_xx_.cols, CV_32FC2);

  char* preOutputImagePtr = reinterpret_cast<char*> (preOutput_.data);

  int preOutputImageStep0 =  preOutput_.step[0];
  int preOutputImageStep1 =  preOutput_.step[1];


  char* inputMaskPtr = reinterpret_cast<char*> (imageMask_.data);

  int inputMaskStep0 =  imageMask_.step[0];
  int inputMaskStep1 =  imageMask_.step[1];

  // For each image pixel Hessian, evaluate its eigen vectors, then look at the cost
  for (int i =0 ; i < pixCount; i++)
  {
    // Get the image index.
    int xPos =  i%grayImage_xx_.cols;
    int yPos =  static_cast<int> (floor(static_cast<float> (i)/static_cast<float> (grayImage_.cols)));

    // construct the preoutput pointer for the current index
    float* prePointer =  reinterpret_cast<float*>
      (preOutputImagePtr + preOutputImageStep0*yPos + preOutputImageStep1*xPos);

    // If the mask is valid, use it to select points
    if (imageMask_.rows == imageMask_.rows && imageMask_.cols == preOutput_.cols)
    {
      char* maskVal = (inputMaskPtr+ inputMaskStep0*yPos + inputMaskStep1*xPos);

      if (maskVal[0] == 0)
      {
        prePointer[0] = 0.0;
        prePointer[1] = 0.0;
        continue;
      }
    }  // if(inputMask.rows == preOutput_.rows && inputMask.cols == preOutput_.cols)

    // Identify the eigenvectors using the quadratic equation
    float vMag(0.0);
    float v_y(0.0);
    float v_x(1.0);
    float a2(0.0);

    float det(gradPtr_xx[i]*gradPtr_yy[i]-gradPtr_yx[i]*gradPtr_yx[i]);
    float b(-gradPtr_xx[i]-gradPtr_yy[i]);
    float c(det);
    float descriminant = sqrt(b*b-4*c);

    float eig0;
    float eig1;
    float r_Beta;

    // verify that the descriminent is > epsilon
    // If it isn't, then there is a single repeated eigenvector.
    if (descriminant > 0.000000001)
    {
      // compute the eigen values.
      eig0 = (-b+descriminant)/(2);
      eig1 = (-b-descriminant)/(2);

      r_Beta = eig0/eig1;

      // find the dominant eigenvector:
      if (abs(r_Beta) > 1.0)
      {
        // eig0 is larger.
        r_Beta = 1/r_Beta;
        v_y = (eig0-gradPtr_xx[i])*v_x/(gradPtr_xy[i]);
      }
      else
      {
        // eig1 is larger.
        v_y = (eig1-gradPtr_xx[i])*v_x/(gradPtr_xy[i]);
      }
    }  // if(descriminant > 0.000000001)
    else
    {
      // the eigenvalue is repeated.
      eig0 = eig1 = -b/2;
      r_Beta = 1.0;
      v_y = 0.00;
      v_x = 1.0;
    }

    // In this formulation, the image peak is 1.0;
    vMag = exp(-r_Beta*r_Beta/(betaParam))*(1-exp(-(eig0*eig0+eig1*eig1)/(cParam)));

    // include the eigenVector direction:
    float a = atan2(v_y, v_x);

    // the output of atan2 is [-pi, pi].
    // ensure that a is between 0 and pi rads
    if (a > 0.00)
    {
      a2 = (a);
    }
    else
    {
      a2 = (a+3.1415);
    }

    // error catch if vMag is undefined...
    if (!(vMag <= 1) || !(vMag >= 0))
    {
      float test = 1;
      std::cout << "Bad number here\n";
    }
    // HSV inspired space assignment
    prePointer[0] = a2;  // vesselness normal direction
    prePointer[1] = vMag;  // vesselness magnitude
  }

  // Once all is said and done, blur the final image using a gaussian.
  angleMagBlur(preOutput_, dst, this->filterParameters_.postProcess);

  return;
}


void angleMagBlur(const cv::Mat &src, cv::Mat &dst, const gaussParam inputParam)
{
  // reallocate the dst matrix
  dst.create(src.size(), src.type());


  // Construct a square gaussian kernel
  cv::Mat gaussKernelA = cv::getGaussianKernel(inputParam.side, inputParam.variance, CV_32F);
  cv::Mat gaussKernel = gaussKernelA*gaussKernelA.t();
  int gaussOffset = floor(static_cast<float> (inputParam.side)/2);


  int imagePixCount = src.rows*src.cols;

  int gaussPixCount = gaussKernel.rows*gaussKernel.cols;


  char * gPtr = reinterpret_cast<char*> (gaussKernel.data);
  int  gStep0 = gaussKernel.step[0];
  int  gStep1 = gaussKernel.step[1];

  char * srcPtr = reinterpret_cast<char*> (src.data);
  int  srcStep0 = src.step[0];
  int  srcStep1 = src.step[1];

  char * dstPtr =  reinterpret_cast<char*> (dst.data);
  int  dstStep0 = dst.step[0];
  int  dstStep1 = dst.step[1];

  // compute the kernel convolution.
  for (int i = 0; i < imagePixCount; i++)
  {
    int dstXPos =  i%src.cols;
    int dstYPos =  static_cast<int> (floor(static_cast<double> (i)/static_cast<double> (src.cols)));

    float* dstPointer =  reinterpret_cast<float*> (dstPtr+ dstStep0*dstYPos + dstStep1*dstXPos);

    float val = 0.0;
    cv::Point2f dirPt(0, 0);

    // when iterating through the gaussian kernel,
    // create the vesselness vector.
    // then sum the gaussian scaled vectors in order to estimate the output vector
    for (int j = 0; j < gaussPixCount; j++)
    {
      int gXPos = j%gaussKernel.cols;
      int gYPos = static_cast<int> (floor(static_cast<double> (j)/static_cast<double> (gaussKernel.cols)));

      float* gPointer =  reinterpret_cast<float*> (gPtr+ gStep0*gYPos + gStep1*gXPos);

      int srcXPos = dstXPos-gaussOffset+gXPos;
      int srcYPos = dstYPos-gaussOffset+gYPos;

      // constant corner assumption:
      if (srcXPos < 0) srcXPos = 0;
      if (srcYPos < 0) srcYPos = 0;
      if (srcXPos >= src.cols) srcXPos = src.cols-1;
      if (srcYPos >= src.rows) srcYPos = src.rows-1;

      float* srcPointer =  reinterpret_cast<float*> (srcPtr+ srcStep0*srcYPos + srcStep1*srcXPos);

      val += srcPointer[1]*gPointer[0];

      cv::Point2f newDir(srcPointer[1]*gPointer[0]*cos(srcPointer[0]), srcPointer[1]*gPointer[0]*sin(srcPointer[0]));

    // find the angle between the two vectors and flip if necessary
    float dotResult = newDir.dot(dirPt)/(norm(newDir)*norm(dirPt));

    // this makes the sum constructive
    if (dotResult < 0.0) dirPt -= newDir;
    else dirPt += newDir;
  }
  dstPointer[1]  = val;
  float newAngle = atan2(dirPt.y, dirPt.x);

  // Constrain the angle to be between 0 an pi rads.
  if (newAngle < 0.0) dstPointer[0] = (newAngle+3.1415);
  else dstPointer[0] = (newAngle);
  }
  return;
}


// empty destructor function
VesselnessNodeCPU::~VesselnessNodeCPU()
{
}


cv::Size VesselnessNodeCPU::allocateMem(const cv::Size& sizeIn)
{
    imgAllocSize_ = sizeIn;
    outputImage_.create(imgAllocSize_, CV_32FC2);
    return imgAllocSize_;
}
