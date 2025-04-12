#ifndef _ProjectFacetToPoint
#define _ProjectFacetToPoint

#include "CudaFunctions.cuh"
#include <stdio.h>
#include <iostream>
#include <fstream>


int StartCuda()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
    }
	return 0;
}

int StopCuda()
{
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	return 0;
}

__device__ __inline__ dcomplex CmulGPU(dcomplex a, dcomplex b)
{
	dcomplex c;
	c.r=a.r*b.r-a.i*b.i;
	c.i=a.i*b.r+a.r*b.i;
	return c;
}


__device__ __inline__ dcomplex RCmulGPU(float x, dcomplex a)
{
	dcomplex c;
	c.r=x*a.r;
	c.i=x*a.i;
	return c;
}

__device__ __inline__ float dotProd(float3 a, float3 b)
{
	return a.x*b.x+a.y*b.y+a.z*b.z;
}

__device__ __inline__ float GetVectorLengthGPU(float3 v1)
{
	return sqrtf(v1.x*v1.x+v1.y*v1.y+v1.z*v1.z);
}

__device__ __inline__ float3 MakeVectorScGPU(float4 Origin, float3 Dest)
{
	float3 Vc;
	Vc.x = Dest.x-Origin.x;
	Vc.y = Dest.y-Origin.y;
	Vc.z = Dest.z-Origin.z;
	return Vc;
}




__global__ void ProjectFacetToPointGPU(dcomplex* dev_OutPressure,dcomplex* dev_Pressure,float* dev_CosInc,
												float3 ProjectToCentriod,float4* dev_Positions,dcomplex k_wave, size_t NumPoints)
{

    unsigned int i = threadIdx.x + blockIdx.x;
	if( i >= NumPoints )
		return;

	float3 vpS = MakeVectorScGPU(dev_Positions[i],ProjectToCentriod);
	float R = GetVectorLengthGPU(vpS);
	float Cos_Scat = vpS.z/R;

	dcomplex Phase;
	Phase.r = sinf(R*k_wave.i)*expf(R*k_wave.i);
	Phase.i = cosf(R*k_wave.r)*expf(R*k_wave.i);

	//*c.r=a.r*b.r-a.i*b.i;
	//*c.i=a.i*b.r+a.r*b.i;
	float a = (dev_CosInc[i]+Cos_Scat)/(4*PI*R);

	dcomplex b;
	b.r = (k_wave.i*Phase.r-R*k_wave.i*Phase.i)*a;
	b.i = (-k_wave.r*Phase.r+R*k_wave.i*Phase.i)*a;

	dev_OutPressure[i].r = b.r*dev_Pressure[i].r-b.i*dev_Pressure[i].i;
	dev_OutPressure[i].i = b.i*dev_Pressure[i].r+b.r*dev_Pressure[i].i;
}

int ProjectFacetToPoint( PointData* DestPoint, FacetData* Facet, dcomplex k_wave)
{
	dcomplex* dev_Pressure = 0;
	float4* dev_Positions = 0;
	float* dev_CosInc = 0;
	dcomplex* dev_OutPressure = 0;
	float3 ProjectToCentriod;
	cudaError_t cudaStatus;

	size_t NumPoints = Facet->NumPositions;
	dcomplex* tmp_OutPressure = new dcomplex[NumPoints];

	float3 vCp = GeoMath::MakeVectorSc(Facet->Centriod,DestPoint->PointLocation);
	ProjectToCentriod.x = GeoMath::dotProductSc(Facet->xAxis,vCp);
	ProjectToCentriod.y = GeoMath::dotProductSc(Facet->yAxis,vCp);
	ProjectToCentriod.z = GeoMath::dotProductSc(Facet->Normal,vCp);

	int threadsPerBlock = 32;
    int blocksPerGrid =((int)NumPoints + threadsPerBlock - 1) / threadsPerBlock;
    dim3 cudaBlockSize(threadsPerBlock,1,1);
    dim3 cudaGridSize(blocksPerGrid, 1, 1);


	for (auto Projection = Facet->Projections.begin(); Projection != Facet->Projections.end(); ++Projection)
    {  
		cudaStatus = cudaMalloc((void**)&dev_Pressure, NumPoints*sizeof(dcomplex));
		if (cudaStatus != cudaSuccess) {
			printf("cudaMalloc failed!");
			return 1;
		}
    
		cudaStatus = cudaMalloc((void**)&dev_OutPressure, NumPoints*sizeof(dcomplex));
		if (cudaStatus != cudaSuccess) {
			printf("cudaMalloc failed!");
			return 1;
		}

		cudaStatus = cudaMemcpy(dev_Pressure, (*Projection)->PressureValues, NumPoints*sizeof(dcomplex), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("cudaMemcpy failed!");
			return 1;
		}

		cudaStatus = cudaMalloc((void**)&dev_CosInc, NumPoints*sizeof(float));
		if (cudaStatus != cudaSuccess) {
			printf("cudaMalloc failed!");
			return 1;
		}

		cudaStatus = cudaMemcpy(dev_CosInc, (*Projection)->CosInc, NumPoints*sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("cudaMemcpy failed!");
			return 1;
		}

		cudaStatus = cudaMalloc((void**)&dev_Positions, NumPoints*sizeof(float4));
		if (cudaStatus != cudaSuccess) {
			printf("cudaMalloc failed!");
			return 1;
		}

		cudaStatus = cudaMemcpy(dev_Positions, Facet->PositionVector, NumPoints*sizeof(float4), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("cudaMemcpy failed!");
			return 1;
		}

		// Launch a kernel on the GPU with one thread for each element.
		//ProjectFacetToPointGPU<<<1, NumPoints>>>(dev_OutPressure,dev_Pressure,dev_CosInc,
		//										ProjectToCentriod, dev_Positions, k_wave, NumPoints);
		ProjectFacetToPointGPU<<<cudaBlockSize, cudaGridSize>>>(dev_OutPressure,dev_Pressure,dev_CosInc,
												ProjectToCentriod, dev_Positions, k_wave, NumPoints);
		  // Check for any errors launching the kernel
		 cudaStatus = cudaGetLastError();
		 if (cudaStatus != cudaSuccess) {
			 printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			 return 1;
		 }
    
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			return 1;
		}

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(tmp_OutPressure, dev_OutPressure, NumPoints*sizeof(dcomplex), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			printf("cudaMemcpy failed!");
			return 1;
		}

		//for(size_t i = 0; NumPoints > i; i++)
		//{
		//	std::cout << "Pixel Count   : " << i << "\n";
		//	std::cout << "Pressure Real : " << tmp_OutPressure[i].r << "\n";
		//	std::cout << "Pressure Imag : " << tmp_OutPressure[i].i << "\n";
		//}



		dcomplex& PressureOut = DestPoint->Pressure;
		for( size_t i = 0; NumPoints > i; i++)
		{
			PressureOut.r += tmp_OutPressure[i].r;
			PressureOut.i += tmp_OutPressure[i].i;
		}
		//std::cout << "Done Projection to Feild Point.\n";

	}
	
    cudaFree(dev_Pressure);
    cudaFree(dev_Positions);
	cudaFree(dev_CosInc);
	cudaFree(dev_OutPressure);
    return 0;
}
#endif