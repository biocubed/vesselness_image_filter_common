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

#include "vesselness_image_filter_common/vesselness_image_filter_common.h"

using namespace cv;

//TODO brief introductory comments...

gaussParam setParameter(int side, float variance)
{
    gaussParam newParameter;
    newParameter.side = side;
    if(newParameter.side%2 == 0)
    {
        ROS_INFO("The side length must be odd, incrementing");
        newParameter.side++;

    }
    if(newParameter.side == 1)
    {
        ROS_INFO("The side length must be at least 3, setting to 3");
        newParameter.side = 3;
    }
    newParameter.variance = abs(variance);

    return newParameter;
}

VesselnessNodeBase::VesselnessNodeBase(const char* subscriptionChar,const char* publicationChar):it_(nh_)
{

    //predetermined init values. (sorta random)
    hessParam.variance = 1.5;
    hessParam.side = 5;
    betaParam = 0.1;    //  betaParamIn;
    cParam    = 0.005;     //  cParamIn;

    postProcess.variance = 2.0;
    postProcess.side = 7;

    //initialize the kernels.
    //initKernels();

    // Subscribe to input video feed and publish output video feed
    image_sub_ = it_.subscribe(subscriptionChar, 1,
        &VesselnessNodeBase::imgTopicCallback, this);

    //subscribe to the setting topic
    settings_sub_ = nh_.subscribe("/vesselness/settings", 1,
        &VesselnessNodeBase::updateFilter, this);  //imageCB is the callback f.

    //data output.
    image_pub_ = it_.advertise(publicationChar, 1);


}




//image topic callback hook
void  VesselnessNodeBase::imgTopicCallback(const sensor_msgs::ImageConstPtr& msg) {


    std::cout << "Processing an image" << std::endl;

    cv_bridge::CvImagePtr cv_ptrIn;
    cv_bridge::CvImage   cv_Out;

    //Attempt to pull the image into an opencv form.
    try
    {
        cv_ptrIn = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }

    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    std::cout << "converted image to opencv" << std::endl;
    //Actual process segmentation code:

    segmentImage(cv_ptrIn->image,outputImage);


    //The result is outputImage.
    //Publish this output
    //Fill in the headers and encoding type
    cv_Out.image = outputImage;
    //cv_Out.header =  cv_ptrIn->header;

    if(outputChannels == 1) cv_Out.encoding = std::string("32FC1");
    if(outputChannels == 2) cv_Out.encoding = std::string("32FC2");

    //publish the outputdata now.
    image_pub_.publish(cv_Out.toImageMsg());
    /*Mat outputImageDisp,preOutputImageDisp; */

    std::cout << "Finished an image" << std::endl;
}

void VesselnessNodeBase::updateFilter(const vesselness_image_filter_common::vesselness_params::ConstPtr &msg)
{
    hessParam = setParameter(msg->hessianSide,msg->hessianVariance);

    postProcess = setParameter(msg->postProcessSide,msg->postProcessVariance);

    betaParam = msg->betaParameter;    //  betaParamIn;
    cParam    = msg->cParameter;     //  cParamIn;

    initKernels();
    ROS_INFO("Updated and reinitialized the kernels");
}


VesselnessNodeBase::~VesselnessNodeBase()
{




}
