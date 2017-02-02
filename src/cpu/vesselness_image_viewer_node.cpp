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
#include <vesselness_image_filter_cpu/vesselness_lib.h>
#include <string>

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;

  std::string windowNodeName;

public:
  ImageConverter()
    :it_(nh_)
  {
    // Subscribe to input video feed and display output video feed
    // use the image callback to display the image.
    // TODO(biocubed) understand how using opencv HIGHGUI with  libQT vs. libGTK makes a difference.
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
    if (msg->encoding.compare(std::string("32FC2")) == 0)
    {
        ROS_INFO("Converting two channel image");
        cv::Mat outputImage;
        convertSegmentImageCPU(cv_ptr->image, outputImage);

        ROS_INFO("Showing Image");
        // Update GUI Window
        cv::imshow(windowNodeName, outputImage);
        cv::waitKey(3);
    }
    else if (msg->encoding.compare(std::string("32FC1")) == 0)
    {
        ROS_INFO("Converting single channel image");
        cv::Mat outputImage;

        convertSegmentImageCPUBW(cv_ptr->image, outputImage);

        cv::imshow(windowNodeName, outputImage);
        cv::waitKey(3);
    }
    else ROS_INFO("Invalid Image");
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_viewer");
  ImageConverter ic;
  ROS_INFO("Ready to convert images");
  ros::spin();
  ROS_INFO("Quitting the image viewer...");
  return 0;
}
