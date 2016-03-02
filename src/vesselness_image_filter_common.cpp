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


void VesselnessNodeBase::paramCallback(vesselness_image_filter_common::vesselness_params &params_, uint32_t level)
{
  ROS_INFO("Reconfigure request : %d %f %f %f %d %f",
           config.groups.hessParams.side,
           config.groups.hessParams.variance,
           config.beta,
           config.c,
           config.groups.postParams.side,
           config.groups.postParams.variance);
	
	
	gaussParam hessParam_(config.groups.hessParams.variance,config.groups.hessParams.side);
    gaussParam postProcess_(config.groups.postParams.variance,config.groups.postParams.side);

    float betaParam_ = config.beta;    //  betaParamIn;
    float cParam_    = config.c;     //  cParamIn;

	filterParameters =  segmentThinParam(hessParam_,postProcess_,betaParam_,cParam_);
	kernelReady = false;
    initKernels();
    ROS_INFO("Updated and reinitialized the kernels");
}


//TODO brief introductory comments...
VesselnessNodeBase::VesselnessNodeBase(const char* subscriptionChar,const char* publicationChar):
    it_(nh_),
	filterParameters(gaussParam(1.5,5),gaussParam(2.0,7),0.1,0.005),
	imgAllocSize(-1,-1),
	kernelReady(false),
	outputChannels(1)
{
    // Subscribe to input video feed and publish output video feed
    image_sub_ = it_.subscribe(subscriptionChar, 1,
        &VesselnessNodeBase::imgTopicCallback, this);

    //data output.
    image_pub_ = it_.advertise(publicationChar, 1);
	
	//initialize the parameter server.
    f = boost::bind(&VesselnessNodeBase::paramCallback, this, _1, _2);
    srv.setCallback(f);
	
	// initialize the kernels
	initKernels();

}

//image topic callback hook
void  VesselnessNodeBase::imgTopicCallback(const sensor_msgs::ImageConstPtr& msg) {
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

    // ROS_INFO("Converted image to opencv");
    
	if (cv_ptrIn->image.size().height =! imgAllocSize.height || cv_ptrIn->image.size().width =! imgAllocSize.width )
	{
	    ROS_INFO("Resizing the allocated matrices");
		imgAllocSize = newSize(allocateMem(cv_ptrIn->image.size());
	}
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

}





VesselnessNodeBase::~VesselnessNodeBase()
{


}
