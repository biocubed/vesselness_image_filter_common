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

#ifndef IMAGESEGMENTCPUH
#define IMAGESEGMENTCPUH


#include <vesselness_image_filter_common/vesselness_image_filter_common.h>



//This class extends the basic VesselnessNode based on using a CPU to complete the actual processing.
class VesselnessNodeCPU: public VesselnessNodeBase {

private:

    //Input and output information
    cv::Mat input;
    cv::Mat output;

    //Intermediates:
    cv::Mat cXX;
    cv::Mat cXY;
    cv::Mat cYY;

    cv::Mat greyImage_xx;
    cv::Mat greyImage_xy;
    cv::Mat greyImage_yy;

    cv::Mat inputGreyG;
    cv::Mat inputFloat255G;
    cv::Mat ones;
    cv::Mat inputFloat1G;

    cv::Mat preOutput;

    cv::Mat scaled;
    cv::Mat scaledU8;
    cv::Mat dispOut;

    //Gauss kernels
    cv::Mat gaussKernel_XX;
    cv::Mat gaussKernel_XY;
    cv::Mat gaussKernel_YY;
    cv::Mat imageMask;

    cv::Mat greyFloat;
    cv::Mat greyImage;


    cv::Mat srcMats;
    cv::Mat dstMats;


    //status booleans
    bool kernelReady;
    bool allocatedKernels;


    void  setKernels();
    void  initKernels() override;
    void  updateKernels();



    //declare the memory management functions
    //void allocateMem(Size); (declared in the abstract base class)
    cv::Size allocateMem(const cv::Size&);
    void deallocateMem();

    /*TODO void VesselnessNodeGPU::findOutputCutoffs(float*,int = 10); */

    //blocking image segmentation
    void segmentImage(const cv::Mat &, cv::Mat &);

    
    //Update object parameters.
    void updateKernels(const segmentThinParam &);
	

public:
   
    //This function needs to operate at peak speed:
    VesselnessNodeCPU(const char*,const char*); //constructor
    VesselnessNodeCPU();    //default constructor
    ~VesselnessNodeCPU();   //deconstructor

};





#endif
