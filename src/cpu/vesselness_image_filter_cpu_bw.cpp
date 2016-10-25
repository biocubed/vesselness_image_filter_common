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

#include <vesselness_image_filter_cpu/vesselness_filter_node_cpu_bw.h>

using namespace cv;

void VesselnessNodeCPUBW::deallocateMem()
{


}

VesselnessNodeCPUBW::VesselnessNodeCPUBW(const char* subscriptionChar,const char* publicationChar):VesselnessNodeBase(subscriptionChar,publicationChar)
{
    // initialize the kernels
    outputChannels = 1;
    initKernels();
    setParamServer();
}


void VesselnessNodeCPUBW::initKernels(){

    double var(filterParameters.hessProcess.variance);

    //Allocate the matrices
    gaussKernel_XX =Mat(filterParameters.hessProcess.side,filterParameters.hessProcess.side,CV_32F);
    gaussKernel_XY =Mat(filterParameters.hessProcess.side,filterParameters.hessProcess.side,CV_32F);
    gaussKernel_YY =Mat(filterParameters.hessProcess.side,filterParameters.hessProcess.side,CV_32F);

    int kSizeEnd = (int) (filterParameters.hessProcess.side-1)/2;

    for(int ix = -kSizeEnd; ix < kSizeEnd+1; ix++){
        for(int iy = -kSizeEnd; iy < kSizeEnd+1; iy++){

            float ixD = (float) ix;
            float iyD = (float) iy;

            gaussKernel_XX.at<float>(iy+kSizeEnd,ix+kSizeEnd) = (ixD*ixD)/(var*var)*gaussFnc(var,ixD,iyD)-1/(var)*gaussFnc(var,ixD,iyD);

            gaussKernel_YY.at<float>(iy+kSizeEnd,ix+kSizeEnd) = (iyD*iyD)/(var*var)*gaussFnc(var,ixD,iyD)-1/(var)*gaussFnc(var,ixD,iyD);

            gaussKernel_XY.at<float>(iy+kSizeEnd,ix+kSizeEnd) = (iyD*ixD)/(var*var)*gaussFnc(var,ixD,iyD);
        }
    }
}




void  VesselnessNodeCPUBW::segmentImage(const Mat& src,Mat& dst) {

    float betaParam(filterParameters.betaParam);
    float cParam(filterParameters.cParam);
    //Actual process segmentation code:
    cvtColor(src,greyImage,CV_BGR2GRAY);
    greyImage.convertTo(greyFloat,CV_32FC1,1.0,0.0);

    float *greyFloatPtr = (float*) greyFloat.data;
    greyFloat /= 255.0;

    //Gaussian Blur filtering (XX,XY,YY);
    filter2D(greyFloat,greyImage_xx,-1,gaussKernel_XX);
    filter2D(greyFloat,greyImage_xy,-1,gaussKernel_XY);
    filter2D(greyFloat,greyImage_yy,-1,gaussKernel_YY);

    std::cout << "Blurred images" << std::endl;

    //Compute the number of total pixels
    int pixCount = greyImage_xx.rows*greyImage_xx.cols;


    //pull out the image data pointers
    float *gradPtr_xx = (float*)  greyImage_xx.data;
    float *gradPtr_yx = (float*)  greyImage_xy.data;
    float *gradPtr_xy = (float*)  greyImage_xy.data;
    float *gradPtr_yy = (float*)  greyImage_yy.data;

    preOutput.create(greyImage_xx.rows,greyImage_xx.cols,CV_32FC1);
    char* preOutputImagePtr = (char*) preOutput.data;

    int preOutputImageStep0 =  preOutput.step[0];
    int preOutputImageStep1 =  preOutput.step[1];


    char* inputMaskPtr = (char*) imageMask.data;

    int inputMaskStep0 =  imageMask.step[0];
    int inputMaskStep1 =  imageMask.step[1];


    std::cout << "at while loop" << std::endl;

    //From Frangi et al.
    //for each image, evaluate its eigen vectors, then look at the cost
	for(int i =0 ; i < pixCount; i++){
		
		int xPos =  i%greyImage_xx.cols;
		int yPos =  (int) floor(((float) i)/((float) greyImage.cols));

		//construct the output pointer
		float* prePointer =  (float*) (preOutputImagePtr+ preOutputImageStep0*yPos + preOutputImageStep1*xPos); 

		//If the mask is valid, use it to select points
		if(imageMask.rows == imageMask.rows && imageMask.cols == preOutput.cols){
			char* maskVal = (inputMaskPtr+ inputMaskStep0*yPos + inputMaskStep1*xPos); 

			if(maskVal[0] == 0)
			{
                prePointer[0] = 0.0;
				continue;
			}
		} //if(inputMask.rows == preOutput.rows && inputMask.cols == preOutput.cols)
				

		float vMag =0.0;
		float v_y  =0.0;
		float v_x  =1.0;
		float a2   =0.0;

				
		float det = gradPtr_xx[i]*gradPtr_yy[i]-gradPtr_yx[i]*gradPtr_yx[i];
		float b = -gradPtr_xx[i]-gradPtr_yy[i];
		float c =  det;
		float descriminant = sqrt(b*b-4*c);

		float eig0;
		float eig1;
		float r_Beta;

					
		//adding safety for small values of the descriminant.
		if(descriminant > 0.000000001) 
		{

			eig0 = (-b+descriminant)/(2);
			eig1 = (-b-descriminant)/(2);

			r_Beta = eig0/eig1;

			//find the dominant eigenvector:
			if(abs(r_Beta) > 1.0)  //indicates that eig0 is larger.
			{
				
				r_Beta = 1/r_Beta;		

			}
		} //if(descriminant > 0.000000001) 
		else
        {

            eig0 = eig1 = -b/2;
            r_Beta = 1.0;
            v_y = 0.00;
            v_x = 1.0;

        }

        //In this formulation, the image peak is 1.0;	
        vMag = exp(-r_Beta*r_Beta/(betaParam))*(1-exp(-(eig0*eig0+eig1*eig1)/(cParam)));


        if(!(vMag <= 1) || !(vMag >= 0))
        {
            float test = 1;
        }

        prePointer[0] = vMag;

    }
    //Once all is said and done, blur the final image using a gaussian.
    dst.create(src.size(),src.type());
    Size kernelSize(filterParameters.postProcess.side,filterParameters.postProcess.side);
    GaussianBlur(preOutput,dst,kernelSize,filterParameters.postProcess.variance,filterParameters.postProcess.variance);

    return;
}



//destructor function
VesselnessNodeCPUBW::~VesselnessNodeCPUBW(){
    //clean up the Mats and memory



}


Size VesselnessNodeCPUBW::allocateMem(const Size &sizeIn){

    imgAllocSize = sizeIn;
    outputImage.create(imgAllocSize,CV_32FC1);
    return imgAllocSize;
}
