/*  Copyright (c) 2014 Case Western Reserve University
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
 *   * Neither the name of Case Western Reserve Univeristy, Inc. nor the names of its
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
#include <vesselness_image_filter_cpu/vesselness_lib.h>


// Converts a simgle image into a displayable RGB format.
void convertSegmentImageCPU(const cv::Mat&src, cv::Mat&dst)
{
  cv::Mat temp1 = src.mul(cv::Scalar((179.0/255)/3.14159, 1.0));
  cv::Mat temp2, temp3;
  cv::convertScaleAbs(temp1, temp2, 255.0);
  temp3.create(src.rows, src.cols, CV_8UC3);

  cv::Mat tempHalf = cv::Mat::ones(src.rows, src.cols, CV_8UC1)*127;

  cv::Mat in[] = {temp2, tempHalf};

  // forming an array of matrices is an efficient operation,
  // because the matrix data is not copied, only the headers
  // rgba[0] -> bgr[2], rgba[1] -> bgr[1],
  // rgba[2] -> bgr[0], rgba[3] -> alpha[0]
  int from_to[] = {0, 0, 1, 1, 2, 2};

  cv::mixChannels(in, 2, &temp3, 1, from_to, 3);
  cv::cvtColor(temp3, dst, CV_HSV2BGR);
}


void convertSegmentImageCPUBW(const cv::Mat&src, cv::Mat&dst)
{
  double maxVal;
  cv::minMaxLoc(src, NULL, &maxVal, NULL, NULL);
  cv::convertScaleAbs(src, dst, (255.0/maxVal));
}

void findOutputCutoff(const cv::Mat&src, double *cuttOff, int iters)
{
  // this refines the cuttoff mean of the image.
  cv::Scalar meanOut;

  if (cuttOff[0] <= 0)
  {
    double mean = cv::mean(src)[1];
    cuttOff[0] = mean/2;
  }
  cv::Mat threshMask;

  for (int i(0); i < iters; i++)
  {
    cv::inRange(src, cv::Scalar(-7, cuttOff[0], 0), cv::Scalar(7, 10, 1), threshMask);

    double mean0 = cv::mean(src, threshMask < 100)[1];
    double mean1 = cv::mean(src, threshMask > 100)[1];

    double newCuttoffMean = mean0/2+mean1/2;
    cuttOff[0] = newCuttoffMean;
  }
}


cv::Point2f angleMagMean(const cv::Mat &src, const cv::Rect &srcRegion)
{
  cv::Point2f result(0.0, 0.0);

  // obtain the Region of interest, note there is no safety check.
  cv::Mat subMat(src(srcRegion));

  char* imagePtr = reinterpret_cast<char*> (subMat.data);

  int   imageStep0 =  subMat.step[0];

  int   imageStep1 =  subMat.step[1];

  int pixelCount1(subMat.cols*subMat.rows);

  for (int i = 0; i < pixelCount1; i++)
  {
    int xPos(i%subMat.cols);
    // @TODO replace the c style casting with static casting
    int yPos(static_cast<int> (floor(static_cast<float> (i)/(static_cast<float> (subMat.cols)))));

    float* pxPtr =  reinterpret_cast<float*> (imagePtr+ imageStep0*yPos + imageStep1*xPos);

    cv::Point2f newPt(pxPtr[1]*cos(pxPtr[0]), pxPtr[1]*sin(pxPtr[0]));

    float dProd(newPt.dot(result));

    if (dProd > 0)
    {
      result += newPt;
    }
    else
    {
      result -= newPt;
    }
  }

  float gain = 1 / static_cast<float>(pixelCount1);

  result *= gain;
  return result;
}
