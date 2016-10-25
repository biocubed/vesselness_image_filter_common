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



#include "vesselness_image_filter_gpu/vesselness_image_filter_kernels.h"


//This file defines the kernel functions used by the thin segmentation cuda code.

using namespace cv::cuda;

// Need to remove this #define, replace it with a __local__ cuda function.
// #define gaussFncGPU(var,x,y) 1.0f/(3.1415f*2.0f*var)*((float) ));

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

/*end of GPU kernel definitions*/


//image conversion functions
/*
void convertSegmentImagesGPU(const stereoImage& segIn,stereoImage& dispOut){

	for(int lr = 0; lr < 2; lr++)
	{
		convertSegmentImageGPU(segIn[lr],dispOut[lr]);
	}

}

void convertSegmentImageGPU(const Mat&src,Mat&dst){

	cv::gpu::setDevice(0);
	
	GpuMat srcGPU;
	GpuMat dstGPU;
	GpuMat scaled,u8bit;
	srcGPU.upload(src);

	multiply(srcGPU, Scalar(1/3.14159,1.0,1.0),scaled,255.0);
	scaled.convertTo(u8bit,CV_8UC3,1.0,0.0);
	
	gpu::cvtColor(u8bit,dstGPU,CV_HSV2BGR);
	dstGPU.download(dst);
}



//basic function
void segmentThinGPU(const Mat&src,Mat &dst,const segmentThinParam &inputParams)
{

	//First initialize the parameters:
	//Try GPU results
	
	cv::gpu::setDevice(0);
	GpuMat srcGPU;
	GpuMat dstGPU;

	float lBetaParam = 2*inputParams.betaParam*inputParams.betaParam;
	float lCParam = 2*inputParams.cParam*inputParams.cParam;

	//generate the template:
	GpuMat tempGPU_XX(inputParams.preProcess.side,inputParams.preProcess.side,CV_32FC1);
	GpuMat tempGPU_XY(inputParams.preProcess.side,inputParams.preProcess.side,CV_32FC1);
	GpuMat tempGPU_YY(inputParams.preProcess.side,inputParams.preProcess.side,CV_32FC1);

	int offset =  (int) floor((float)inputParams.preProcess.side/2);

	dim3 kBlock(1,1,1);
	dim3 kThread(inputParams.preProcess.side,inputParams.preProcess.side,1);
	genGaussHessKernel_XX<<<kBlock,kThread>>>(tempGPU_XX,inputParams.preProcess.variance,offset);
	genGaussHessKernel_XY<<<kBlock,kThread>>>(tempGPU_XY,inputParams.preProcess.variance,offset);
	genGaussHessKernel_YY<<<kBlock,kThread>>>(tempGPU_YY,inputParams.preProcess.variance,offset);


	GpuMat cXX(src.rows,src.cols,CV_32FC1);
	GpuMat cXY(src.rows,src.cols,CV_32FC1);
	GpuMat cYY(src.rows,src.cols,CV_32FC1);

	//PtrStep<float> test = tempGPU_XX.operator cv::gpu::PtrStepSz<float>();

	//convert image to gray scale witha max of 1.0;
	srcGPU.upload(src);
	GpuMat greySrc;
	GpuMat floatSrc255;
	GpuMat floatSrc1;
	gpu::cvtColor(srcGPU,greySrc,CV_BGR2GRAY);
	greySrc.convertTo(floatSrc255,CV_32FC1,1.0,0.0);
	
	gpu::divide(floatSrc255,Scalar(255.0,255.0,255.0),floatSrc1);
	


	//convolution offset ROI.
	Rect Roi(offset,offset,src.cols-(inputParams.preProcess.side)+1,src.rows-inputParams.preProcess.side+1);
	Rect Roi2(0,0,5,5);



	cXY = 0.0f;
	cXX = 0.0f;
	cYY = 0.0f;
	
	GpuMat cXX1 = cXX(Roi);
	GpuMat cXY1= cXY(Roi);
	GpuMat cYY1= cYY(Roi);

	gpu::ConvolveBuf cBuffer;
	cBuffer.create(floatSrc1.size(),tempGPU_XX.size()); 
	/*gpu::convolve(floatSrc1,tempGPU_XX,cXX1,true,cBuffer); //,false,cBuffer);
	gpu::convolve(floatSrc1,tempGPU_YY,cYY1,true,cBuffer); //,false,cBuffer);
	gpu::convolve(floatSrc1,tempGPU_XY,cXY1,true,cBuffer); //,false,cBuffer); * /

	Mat tempCPU_XX,tempCPU_XY,tempCPU_YY;

	tempGPU_XX.download(tempCPU_XX);
	tempGPU_XY.download(tempCPU_XY);
	tempGPU_YY.download(tempCPU_YY); 


	gpu::filter2D(floatSrc1,cXX,-1,tempCPU_XX);
	gpu::filter2D(floatSrc1,cYY,-1,tempCPU_YY);
	gpu::filter2D(floatSrc1,cXY,-1,tempCPU_XY);


#ifdef GPUOUTPUT

	Mat disp;
	srcGPU.download(disp);
	imshow("window",disp);
	cvWaitKey(0);


	greySrc.download(disp);
	imshow("window",disp);
	cvWaitKey(0);

	floatSrc255.download(disp);
	imshow("window",disp);
	cvWaitKey(0);

	floatSrc1.download(disp);
	imshow("window",disp);
	cvWaitKey(0);

	Mat kernel,tempD,tempD2;
	float test = gaussFncGPU(3.0,0,0);

	std::cout << test <<std::endl;


	tempGPU_XX.download(kernel);
	std::cout << "XX kernel:" << std::endl;
	std::cout << kernel << std::endl;

	tempGPU_XY.download(kernel);
	std::cout << "XY kernel:" << std::endl;
	std::cout << kernel << std::endl;

	tempGPU_YY.download(kernel);
	std::cout << "YY kernel:" << std::endl;
	std::cout << kernel << std::endl;
	
#endif
	
	
#ifdef GPUOUTPUT
	Mat tempOut;
	GpuMat temp = cXX(Roi2);
	
	cXX.download(tempOut);
	imshow("window",abs(tempOut));
	cvWaitKey(0);
	temp.download(tempOut);
	std::cout << "Sample XX" << std::endl;
	std::cout << tempOut << std::endl;
	cXY.download(tempOut);
	imshow("window",abs(tempOut));
	cvWaitKey(0);
	temp = cXY(Roi2);
	temp.download(tempOut);
	std::cout << "Sample XY" << std::endl;
	std::cout << tempOut << std::endl;

	cYY.download(tempOut);
	imshow("window",abs(tempOut));
	cvWaitKey(0);
	temp = cYY(Roi2);
	temp.download(tempOut);
	std::cout << "Sample YY" << std::endl;
	std::cout << tempOut << std::endl;
	cv::destroyWindow("window");
#endif 



	//cv::gpu::GaussianBlur(srcGPU,dstGPU,Size(inputParams.preProcess.side,inputParams.preProcess.side),inputParams.preProcess.variance,inputParams.preProcess.variance);
	//image size is 640x480:
	//Cleary this needs to be fixed...
	dim3 eigBlock(40,30,1);
	dim3 eigThread(16,16,1); 

	GpuMat preOutput(src.rows,src.cols,CV_32FC3);
	
	
	generateEigenValues<<<eigBlock,eigThread>>>(cXX,cXY,cYY,preOutput,lBetaParam,lCParam);

	//Blur the result:
	GpuMat output(src.rows,src.cols,CV_32FC3);
	
	Mat gaussKernel = getGaussianKernel(inputParams.postProcess.side,inputParams.postProcess.variance,CV_32FC1);
	Mat gaussKernelc = gaussKernel*gaussKernel.t();
	int gaussOff = (int) floor(((float) inputParams.postProcess.side)/2.0f);

	GpuMat gaussMat;

	gaussMat.upload(gaussKernelc);

    gaussAngBlur<<<eigBlock,eigThread,1>>>(preOutput,output,gaussMat,gaussOff);

    output.download(dst);
	//preOutput.download(dst); 
	
	//allocate the destination:
	/*dst = Mat(src.rows,src.cols,CV_32FC3);

	float betaParam = inputParams.betaParam;
	float cParam = inputParams.cParam;
	for(int ix = 0; ix < tXX.cols; ix++)
	for(int iy = 0; iy < tXX.rows; iy++)
	{

		float det = tXX.at<float>(iy,ix)*tYY.at<float>(iy,ix)-tXY.at<float>(iy,ix)*tXY.at<float>(iy,ix);
		float b = -tXX.at<float>(iy,ix)-tYY.at<float>(iy,ix);
		float descriminant = sqrt(b*b-4*det);
		float eig0 = (-b+descriminant)/(2);
		float eig1 = (-b-descriminant)/(2);

		float r_Beta = eig0/eig1;
		float v_x = 0.0;
		float v_y = 1.0;
		//find the dominant eigenvector:
		if(abs(r_Beta) > 1.0){  //indicates that eig0 is larger.
				r_Beta = 1/r_Beta;
				v_x = (eig0-tXX.at<float>(iy,ix))*v_y/(tXY.at<float>(iy,ix));
		}
		else v_x = (eig1-tXX.at<float>(iy,ix))*v_y/(tXY.at<float>(iy,ix));
		float vMag = exp(-r_Beta*r_Beta/(lBetaParam))*(1-exp(-(eig0*eig0+eig1*eig1)/(lCParam)));
		float aOut;
		float a = atan2(v_y,v_x);
		if(a > 0.00)
		{
			aOut = (a); ///3.1415;
		}
		else
		{
			aOut = (a+3.1415); ///3.1415;
		}
		Vec3f val;
		val[0] = aOut;
		val[1] = vMag;
		val[2] = 0.5;

		dst.at<Vec3f>(iy,ix) = val;
		/*Vec3f dstR = dst.at<Vec3f>(iy,ix);
		float errorA = abs(dstR[0]-aOut);
		float errorM = abs(dstR[1]-vMag);
		if(errorM > 0.001)
			printf("Large diff\n");* /
	} * /

#ifdef GPUOUTPUT
	//tempD2= dst(Roi2);
	//std::cout << tempD2 << std::endl;
#endif


} 
*/



	

