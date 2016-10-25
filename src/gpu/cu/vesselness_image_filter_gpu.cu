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
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "vesselness_image_filter_gpu/vesselness_image_filter_kernels.h"
#include "vesselness_image_filter_gpu/vesselness_image_filter_gpu.h"
#include <vesselness_image_filter_common/vesselness_image_filter_common.h>
/*This file relies on the following external libraries:
OpenCV
Eigen
cuda
*/

//This file defines the kernel functions used by the thin segmentation cuda code.

using namespace cv::cuda;

// Replaced this function with a __device__ side function.
//#define gaussFncGPU(var,x,y) 1.0f/(3.1415f*2.0f*var)*((float) exp(-x*x/(2.0f*var)-y*y/(2.0f*var)));

__device__ float gaussFncGPU(float var, float x, float y)
{
	float result(expf(-x*x/(2.0f*var)-y*y/(2.0f*var))); 
	result /= (3.1415f*2.0f*var);
	return result;
}

__global__ void genGaussHessKernel_XX(PtrStepSzf output,float var,int offset)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int ixD = x-offset;
    int iyD = y-offset;
    if (x < output.cols && y < output.rows)
    {
        float gaussV = gaussFncGPU(var,ixD,iyD);
        float v = (ixD*ixD)/(var*var)*gaussV-1/(var)*gaussV;
        output(y, x) = v; 
    }
}

__global__ void genGaussHessKernel_XY(PtrStepSzf output,float var,int offset)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float ixD =(float) (offset-x); //offset-x;
    float iyD =(float) offset-y; //offset-y; //
    if (x < output.cols && y < output.rows)
    {
        float gaussV = gaussFncGPU(var,ixD,iyD);
        float v = (iyD*ixD)/(var*var)*gaussV;
        output(y,x) = v; 
    }
}


__global__ void genGaussHessKernel_YY(PtrStepSzf output,float var,int offset)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int ixD = x-offset;
    int iyD = y-offset;
    if (x < output.cols && y < output.rows)
    {
        float gaussV = gaussFncGPU(var,ixD,iyD);
        float v = (iyD*iyD)/(var*var)*gaussV-1/(var)*gaussV;
        output(y,x) = v; 
    }
}

__global__ void generateEigenValues(const PtrStepSzf XX,const PtrStepSzf XY,const PtrStepSzf YY,PtrStepSz<float2> output,float betaParam,float cParam)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < output.cols && y < output.rows)
    {
        float V_mag = 0.0;
        float aOut = 0.0;
        float eig0 = 0.0;
        float eig1 = 0.0;
        float det = XX(y,x)*YY(y,x)-XY(y,x)*XY(y,x);
        float b = -XX(y,x)-YY(y,x);
        float descriminant = sqrt(b*b-4*det);
        float r_Beta;
        float v_y = 0.0;
        float v_x = 1.0;
        if(descriminant > 0.000000001)
        {
            eig0 = (-b+descriminant)/(2);
            eig1 = (-b-descriminant)/(2);
            r_Beta = eig0/eig1;
            //find the dominant eigenvector:
            if(abs(r_Beta) > 1.0){  //indicates that eig0 is larger.
                r_Beta = 1/r_Beta;
                v_y = (eig0-XX(y,x))*v_x/(XY(y,x));
            }
            else v_y = (eig1-XX(y,x))*v_x/(XY(y,x));

            float a = atan2(v_y,v_x);
            if(a > 0.00)
            {
                aOut = (a); ///3.1415;
            }
            else
            {
                aOut = (a+3.1415); ///3.1415;
            }
        }
        else
        {
            eig0 = eig1 = -b/2;
            r_Beta = 1.0;
            v_x = 0.00;
            v_y = 1.0;
            aOut =0.0;
        }
        V_mag = exp(-r_Beta*r_Beta/(betaParam))*(1-exp(-(eig0*eig0+eig1*eig1)/(cParam)));

        output(y,x).x = aOut;
        output(y,x).y = V_mag;
        

        //output(y,x).x = eig0;
        //output(y,x).y = eig1;
        //output(y,x).z = aOut/(3.1415);
    }
}


//Gaussian blurring function
__global__ void gaussAngBlur(const PtrStepSz<float2> srcMat,PtrStepSz<float2> dstMat,PtrStepSzf gMat,int gaussOff)
{

    int x = threadIdx.x + blockIdx.x * blockDim.x; 
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < srcMat.cols && y < srcMat.rows)
    {
        float val = 0.0;
        float2 dirPt;
        dirPt.x = 0.0;
        dirPt.y = 0.0;

        int gaussPixCount= (gaussOff*2+1);


        for(int gx = 0; gx < gMat.cols; gx++)
            for(int gy = 0; gy < gMat.rows; gy++)
            {
                int srcXPos =x-gaussOff+gx;
                int srcYPos =y-gaussOff+gy;

                //constant corner assumption:
                if(srcXPos < 0) srcXPos = 0;
                if(srcYPos < 0) srcYPos = 0;
            
                if(srcXPos >= srcMat.cols) srcXPos = srcMat.cols-1;
                if(srcYPos >= srcMat.rows) srcYPos = srcMat.rows-1;

                float tmpVal = srcMat(srcYPos,srcXPos).y*gMat(gy,gx);
                val += tmpVal;
            
                float tmpAngle = srcMat(srcYPos,srcXPos).x; 

                float2 newDir;
                newDir.x =  tmpVal*cos(tmpAngle);
                newDir.y =  tmpVal*sin(tmpAngle);

                float tempNorm = sqrt(dirPt.x*dirPt.x+dirPt.y*dirPt.y); 
            
                //find the cos between the two vectors;
                float dotResult = (newDir.x*dirPt.x+newDir.y*dirPt.y)/(tempNorm*tmpVal);

                if(dotResult < -0.707)
                {
                    dirPt.x-=newDir.x;
                    dirPt.y-=newDir.y;
                }
                else
                {
                    dirPt.x+=newDir.x;
                    dirPt.y+=newDir.y;
                }
            }
            dstMat(y,x).y = val;  //val;
            float newAngle = atan2(dirPt.y,dirPt.x);
            if(newAngle < 0.0) dstMat(y,x).x = (newAngle+3.1415);
            else dstMat(y,x).x = (newAngle);
    }
    return;
}

//This file defines functions for segmenting the suture thread and the needle from the
//camera images using an object with GPU support.

using namespace cv;

VesselnessNodeGPU::VesselnessNodeGPU(const char* subscriptionChar,const char* publicationChar):
    VesselnessNodeBase(subscriptionChar,publicationChar)
{
    outputChannels = 2;
    initKernels();
    setParamServer();
}

void VesselnessNodeGPU::initKernels(){

	//reallocate the GpuMats
	cv::cuda::GpuMat tempGPU_XX;
    cv::cuda::GpuMat tempGPU_XY;
    cv::cuda::GpuMat tempGPU_YY;
	

ROS_INFO("Allocating");
	tempGPU_XX.create(filterParameters.hessProcess.side,filterParameters.hessProcess.side,CV_32FC1);
	tempGPU_XY.create(filterParameters.hessProcess.side,filterParameters.hessProcess.side,CV_32FC1);
	tempGPU_YY.create(filterParameters.hessProcess.side,filterParameters.hessProcess.side,CV_32FC1);

	//initialize the hessian kernels variables:
	int offset =  (int) floor((float)filterParameters.hessProcess.side/2);
ROS_INFO("Assigning");
	dim3 kBlock(1,1,1);
	dim3 kThread(filterParameters.hessProcess.side,filterParameters.hessProcess.side,1);
	genGaussHessKernel_XX<<<kBlock,kThread>>>(tempGPU_XX,filterParameters.hessProcess.variance,offset);
	genGaussHessKernel_XY<<<kBlock,kThread>>>(tempGPU_XY,filterParameters.hessProcess.variance,offset);
	genGaussHessKernel_YY<<<kBlock,kThread>>>(tempGPU_YY,filterParameters.hessProcess.variance,offset);

	
	
    

    
	ROS_INFO("XX..");
    tempGPU_XX.download(tempCPU_XX);
    ROS_INFO_STREAM(tempCPU_XX);

    ROS_INFO("XY..");
    tempGPU_XY.download(tempCPU_XY);
    ROS_INFO_STREAM(tempCPU_XY);
        

	tempGPU_YY.download(tempCPU_YY);
    ROS_INFO("YY..");
    ROS_INFO_STREAM(tempCPU_YY);


    ROS_INFO("Downloaded the GPU kernels.");
	tempGPU_XX.release();
	tempGPU_XY.release();
	tempGPU_YY.release();
	//initialize the filterParameters.postProcess Kernel:
    ROS_INFO("Creating the outer kernel");
	Mat gaussKernel = getGaussianKernel(filterParameters.postProcess.side,filterParameters.postProcess.variance,CV_32FC1);
	Mat gaussOuter  = gaussKernel*gaussKernel.t();

    ROS_INFO("uploading the outer kernel");
	gaussG.upload(gaussOuter);

	//Finished...

	ROS_INFO("Allocated the GPU post processing kernels.");
    this->kernelReady = true;
}


//This function allocates the GPU mem to save time
cv::Size VesselnessNodeGPU::allocateMem(const cv::Size& size_){

    deallocateMem();
    imgAllocSize= size_;

    //allocate the other matrices.
    preOutput.create(size_, CV_32FC2);
    outputG.create(size_, CV_32FC2);
    inputG.create(size_, CV_8UC3);
    inputGreyG.create(size_, CV_8UC1);
    inputFloat255G.create(size_, CV_32FC1);
    inputFloat1G.create(size_, CV_32FC1);
  

    ones.create(size_, CV_32FC1);
    ones.setTo(Scalar(255.0));

    //allocate the page lock memory
    dstMatMem.create(size_, CV_32FC2);
    ROS_INFO("Allocated the memory");
	return size_;
	
}

//This function allocates the GPU mem to save time
void VesselnessNodeGPU::deallocateMem(){

   ROS_INFO("Allocated the GPU post processing kernels.");
   //input data
   inputG.release();
   inputGreyG.release();
   inputFloat255G.release();
   inputFloat1G.release();

    //output data
    preOutput.release();
    outputG.release();

    ones.release();

    dstMatMem.release();

}


void VesselnessNodeGPU::segmentImage(const Mat &srcMat, Mat &dstMat)
{
    
    if(kernelReady != true)
    {
        ROS_INFO("Unable to process image");
        return;  
    } 
	
	//compute the size of the image
    int iX, iY;

    iX = srcMat.cols;
    iY = srcMat.rows;

    cv::cuda::Stream streamInfo;
    cudaStream_t cudaStream;

    //upload &  convert image to gray scale with a max of 1.0;
    inputG.upload(srcMat, streamInfo);


    cuda::cvtColor(inputG,inputGreyG,CV_BGR2GRAY,0,streamInfo);

    //perform a top hat operation.
    //cuda::morphologyEx(inputGreyG[lr],inputGreyG2[lr],MORPH_BLACKHAT,topKernel,inputBuff1[lr],inputBuff2[lr],Point(-1,-1),1,streamInfo);

    //cuda::cvtColor(inputG[lr],inputGreyG[lr],CV_BGR2GRAY,0,streamInfo[lr]);
    //streamInfo.enqueueConvert(inputGreyG[lr], inputFloat255G[lr], CV_32FC1,1.0,0.0);
    inputGreyG.convertTo(inputFloat255G, CV_32FC1,1.0,0.0,streamInfo);

    //inputGreyG[lr].convertTo(inputFloat255G[lr],CV_32FC1,1.0,0.0);
    //cuda::divide(1/255,inputFloat255G[lr],inputFloat1G[lr],CV_32F,streamInfo);

    cuda::divide(inputFloat255G,ones,inputFloat1G,1.0,CV_32F,streamInfo);


    //cuda::divide(inputFloat255G[lr],Scalar(255.0,255.0,255.0),inputFloat1G[lr]);
    cv::cuda::createLinearFilter(CV_32F,CV_32F,tempCPU_XX)->apply(inputFloat1G,cXX,streamInfo);
    cv::cuda::createLinearFilter(CV_32F,CV_32F,tempCPU_YY)->apply(inputFloat1G,cYY,streamInfo);
    cv::cuda::createLinearFilter(CV_32F,CV_32F,tempCPU_XY)->apply(inputFloat1G,cXY,streamInfo);
    //cuda::filter2D(inputFloat1G,cXX,-1,tempCPU_XX,Point(-1,-1),BORDER_DEFAULT,streamInfo);
    //cuda::filter2D(inputFloat1G,cYY,-1,tempCPU_YY,Point(-1,-1),BORDER_DEFAULT,streamInfo);
    //cuda::filter2D(inputFloat1G,cXY,-1,tempCPU_XY,Point(-1,-1),BORDER_DEFAULT,streamInfo);


    //cuda::filter2D(inputFloat1G[lr],cXX[lr],-1,tempCPU_XX);
    //cuda::filter2D(inputFloat1G[lr],cYY[lr],-1,tempCPU_YY);
    //cuda::filter2D(inputFloat1G[lr],cXY[lr],-1,tempCPU_XY);
	


    int blockX = (int) ceil((double) iX /(16.0f));
    int blockY = (int) ceil((double) iY /(16.0f));


    dim3 eigBlock(blockX,blockY,1);
    dim3 eigThread(16,16,1); 

    //What about here?
    //get the stream access first
    cudaStream = cuda::StreamAccessor::getStream(streamInfo);

    generateEigenValues<<<eigBlock,eigThread,0,cudaStream>>>(cXX,cXY,cYY,preOutput,filterParameters.betaParam,filterParameters.cParam);
        //preOutput[lr].create(iY,iX,CV_32FC3);
        //generateEigenValues<<<eigBlock,eigThread>>>(cXX[lr],cXY[lr],cYY[lr],preOutput[lr],betaParam,cParam);

        //Blur the result:
        int gaussOff = (int) floor(((float) filterParameters.postProcess.side)/2.0f);

        
        //preOutput.copyTo(outputG,streamInfo);
        gaussAngBlur<<<eigBlock,eigThread,0,cudaStream>>>(preOutput,outputG,gaussG,gaussOff);

        //compute the display output.
    /*  multiply(outputG[lr], Scalar(1/3.141,1.0,1.0),scaled[lr],255.0,-1,streamInfo);
        streamInfo.enqueueConvert(scaled[lr],scaledU8[lr],CV_8UC3,1.0,0.0);
        cuda::cvtColor(scaledU8[lr],dispOut[lr],CV_HSV2BGR,0,streamInfo);
        streamInfo.enqueueDownload(outputG[lr],dstMatMem[lr]);

        streamInfo.enqueueDownload(dispOut[lr],dispMatMem[lr]); */


	outputG.download(dstMatMem,streamInfo);
        //streamInfo.enqueueDownload(outputG,dstMatMem);

    streamInfo.waitForCompletion();

        Mat tempDst;
        tempDst = dstMatMem.createMatHeader();
        dstMat = tempDst.clone(); 
}


//destructorfunction
VesselnessNodeGPU::~VesselnessNodeGPU(){
    
	deallocateMem();


}
