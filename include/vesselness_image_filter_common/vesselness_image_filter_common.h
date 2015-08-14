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


#ifndef VESSELNESSNODEH
#define VESSELNESSNODEH

#include <vector>
#include <stdio.h>
#include <iostream>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vesselness_image_filter_common/vesselness_params.h>

/*
 * This file introduces the abstract base class for the vesselness_image_filter nodes.
 * The Base Class is VesselnessNodeBase.
 * The base class gets derived into two forms: a CPU and a GPU based algorithm for image processing.
 * The base class is meant to handle the ROS communication functionality as well as initialize the 
 * members and methods required by both classes.
 */


/*
 * The struct gaussParam stores the information needed to construct a simple gaussian filter kernel.
 * The members are the variance (float) and the length of a side (int). The length of the side should always be odd.
 */
struct gaussParam{

    float variance;

    int side;

};


inline float gaussFnc(float var,float x,float y){
    return 1/(3.1415*2*var)*exp(-x*x/(2*var)-y*y/(2*var));
}



/*
 * The struct segmentThinParam stores all of the variables for the vesselness segmentation
 * This includes the pre (and post) gaussian filter parameters as well as the beta,c parameters
 */
struct segmentThinParam{

    gaussParam preProcess;
    gaussParam postProcess;

    float betaParam;
    float cParam;

};

gaussParam setParameter(int, float);

/*
 * The VesselnessNodeBase abstract class defines members which are common to both the CPU and the GPU instantiation of
 * the vessel_image_filter object. The base class defines all of the public members which are common accross instantiation.
 */
class VesselnessNodeBase{

private:


    //ROS communication members.
    ros::NodeHandle nh_;

    ros::Subscriber settings_sub_;

    image_transport::ImageTransport it_;

    image_transport::Publisher image_pub_;

    image_transport::Subscriber image_sub_;


protected:

    cv::Mat outputImage;

    cv::Size imgAllocSize;

    //vesselness image filter settings
    gaussParam hessParam;
    gaussParam postProcess;

    float betaParam;
    float cParam;

    int outputChannels;

public:


    /*
     * The default constructor uses an input charactor to set who it subscribes to
     * It also then initializes the publisher.
     * Finally, the memory allocation functions are called.
     */
    VesselnessNodeBase(const char*,const char*);

    /*
     * This is the default destructor, It is not used because this class is abstract
     */
    ~VesselnessNodeBase(); 

   /*
    * Since (for speed) and efficiency, it would be useful to operate only on a small subset of the image,
    * This function (while not implemented) is reserved for future use.
    */
   //virtual void setImageMask(const Mat &) = 0;

    /*
     * imgTopicCallback. This callback hook (which activates everytime an image is received) is used to process and publish the new image.
     */
    void  imgTopicCallback(const sensor_msgs::ImageConstPtr&);

    /*
     * The abstract member  segmentImage is used by the image topic callback.
     * The function segments and returns (by reference) the output image.
     */
    virtual void segmentImage(const cv::Mat&, cv::Mat &)=0;

    /*
     * The allocateMem is  called by constructor function.
     * The memory allocation is different on the GPU and the CPU.
     */
    virtual void allocateMem(int,int) = 0;

    /*
     * The deallocateMem is  called by constructor function.
     * The memory allocation is different on the GPU and the CPU.
     */
    virtual void deallocateMem() = 0;


    /*
     * The initKernels function uses the parameters settings to initialize and set the
     * gaussian filter kernels.
     */
    virtual void initKernels()= 0;


    /*
     *
     * This callback triggers when new filter parameters are piped over.
     */
    void updateFilter(const vesselness_image_filter_common::vesselness_params::ConstPtr &);

};


#endif
