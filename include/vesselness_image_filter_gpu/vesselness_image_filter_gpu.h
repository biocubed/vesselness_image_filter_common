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
 *     disclaimer in the documentation and/or other cv::Materials provided
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

#ifndef IMAGESEGMENTGPUH
#define IMAGESEGMENTGPUH

#include <vesselness_image_filter_gpu/vesselness_image_filter_kernels.h>
#include <vesselness_image_filter_common/vesselness_image_filter_common.h>




//This class extends the basic VesselnessNode based on using a GPU to complete the actual processing.
class VesselnessNodeGPU: public VesselnessNodeBase {

private:
    /* private semi-static class members */

   
    cv::cuda::GpuMat inputG;
    cv::cuda::GpuMat inputGreyG;
	cv::cuda::GpuMat inputFloat255G;
	cv::cuda::GpuMat ones;
    cv::cuda::GpuMat inputFloat1G;
    cv::cuda::GpuMat cXX;
    cv::cuda::GpuMat cXY;
    cv::cuda::GpuMat cYY;
    cv::cuda::GpuMat preOutput;
	cv::cuda::GpuMat outputG;
    cv::cuda::GpuMat gaussG;

    cv::cuda::HostMem dstMatMem;
    cv::Mat dstMats;

    //cv::Mat topKernel;
    cv::Mat tempCPU_XX;
    cv::Mat tempCPU_XY;
    cv::Mat tempCPU_YY;


    cv::Size allocateMem(const cv::Size&);
	void deallocateMem();


    //inherited required functions:
    void segmentImage(const cv::Mat &, cv::Mat &);
   
   
public:


    void  initKernels() override;

    explicit VesselnessNodeGPU(const char* subscriptionChar,const char* publicationChar);

    ~VesselnessNodeGPU();   //deconstructor




    


};





#endif
