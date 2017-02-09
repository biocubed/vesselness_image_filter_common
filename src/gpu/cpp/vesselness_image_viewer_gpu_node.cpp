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
#include "vesselness_image_filter_gpu/vesselness_viewer_kernels.h"
#include <string>

// Converts a single image into a displayable RGB format.
class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  double averageRate;

  std::string windowNodeName;

public:
  // Subscribe to input video feed and display output video feed
  // use the image callback to display the image.
  // @TODO understand how using opencv HIGHGUI with  libQT vs. libGTK makes a difference.
  ImageConverter()
    : it_(nh_),
    averageRate(0.0)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("image_thin", 1,
      &ImageConverter::imageCb, this);

    windowNodeName = ros::this_node::getName();

    cv::namedWindow(windowNodeName);
    cv::startWindowThread();
  }

  ~ImageConverter()
  {
    cv::destroyWindow(windowNodeName);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    ros::Time begin = ros::Time::now();
    cv::Mat outputImage;
    // complete the gpu conversion
    convertSegmentImageGPU(cv_ptr->image, outputImage);
    ros::Time end = ros::Time::now();

    // for validation, the average processing rate is logged.
    double startTime = static_cast<double> (begin.nsec);
    double endTime = static_cast<double> (end.nsec);
    double dt(endTime-startTime);
    if (dt < 0)
    {
        dt += 1000000000;
    }
    double rate = 1000000000/(dt);

    averageRate = rate;

    ROS_INFO("Processing Speed %f Hz \x1b[1F", averageRate);
    // Update GUI Window
    cv::imshow(windowNodeName, outputImage);
    cv::waitKey(1);
    }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_viewer");
  ImageConverter ic;
  ros::spin();
  return 0;
}
