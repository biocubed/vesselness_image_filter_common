/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016 Case Western Reserve University
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



 #include "vesselness_image_filter_gpu/vesselness_viewer_kernels.h"

using namespace cv;
using namespace cv::cuda;

void convertSegmentImageGPU(const Mat&src,Mat&dst){
	
    //initialize the stream:
	cv::cuda::Stream streamInfo;

    // allocate the pagelocked memory:
    HostMem srcMatMem(src.size(), CV_32FC2);
    HostMem dstMatMem(src.size(), CV_8UC3);

    Mat srcMatLocked =srcMatMem.createMatHeader();

    src.copyTo(srcMatLocked);

   	GpuMat srcG;
   	srcG.upload(srcMatLocked,streamInfo);
    GpuMat scaledSrc,scaledSrcU;
    multiply(srcG,Scalar(255/3.14159,255.0),scaledSrc,1,CV_32FC2,streamInfo);
    
    scaledSrc.convertTo(scaledSrcU,CV_8UC2,streamInfo);

    std::vector<GpuMat> splitList;
	
    split(scaledSrcU,splitList,streamInfo);
    GpuMat onesG(src.rows,src.cols,CV_8UC1);

    splitList.push_back(onesG);

    splitList[2].setTo(127,streamInfo);

	
	GpuMat outputG,hsvG;

	merge(splitList,hsvG,streamInfo);
    
    cuda::cvtColor(hsvG,outputG,CV_HSV2BGR,0,streamInfo);

    outputG.download(dstMatMem,streamInfo);

    streamInfo.waitForCompletion();

     Mat tempDst = dstMatMem.createMatHeader();
     dst = tempDst.clone(); 

    //clean up the memory
    srcMatMem.release();
    srcG.release();
    scaledSrc.release();
    for (int i = 0; i < splitList.size(); i++)
    {
    	splitList[i].release();
    }
    outputG.release();
    hsvG.release();
    dstMatMem.release();
    // All done
}
