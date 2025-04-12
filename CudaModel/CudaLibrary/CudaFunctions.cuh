#ifndef _CUDAFUNCTIONS
#define _CUDAFUNCTIONS
#include "device_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define _CUDALIBRARY
#include "../CudaModel/TraceObject.h"
#include "../CudaModel/FacetData.h"
#include "../CudaModel/PointData.h"



#ifndef _DCOMPLEX
#define _DCOMPLEX
typedef struct DCOMPLEX {float r,i;} dcomplex;
#endif

//int LoadFacetData( FacetData* Facet);

int MakeFacetsOnGPU( FacetData** Facets, unsigned int NumFacets, dcomplex k_wave);
int ProjectPointToFacet( PointData* SrcPoint,  FacetData* Facet, TraceObject* TraceOb);
int ScanProjectFacetToPoint( PointData* DestPoint, FacetData* Facet, TraceObject* TraceOb,
				bool** PathMatrix, unsigned int NumSourcePoints, unsigned int NumFeildPoints);
void PrintComplexVector( dcomplex* dev_Vector, unsigned int NumPoints);
void PrintVector( float* dev_Vector, unsigned int NumPoints);
int StartCuda();
int StopCuda();
int DeleteTraceObject(TraceObject* TraceObject);
TraceObject* MakeTraceObject();
TraceObject* ScanProjectFacetToFacet(FacetData* Facet_i,FacetData* Facet_j, TraceObject* TraceObj_i);
bool** MakeCollisionDectionMatrix( PointData** SourcePoints, PointData** FeildPoints, FacetData** Facets,
								unsigned int NumSourcePoints, unsigned int NumFieldPoints, unsigned int NumFacets,float MaxTheata);
//#define POINTS_PER_THREAD 8
//#define MAX_THREADS_PER_BLOCK 2048
//#define CHUCK_SIZE 64
//#define BLOCK_UNROLLMAX 32

#define POINTS_PER_THREAD 8
#define MAX_THREADS_PER_BLOCK 1024
#define ROLL_SIZE 3
#define CHUCK_SIZE 4
#define CHUCK_SIZE2 16
#define CHUCK_SIZE3 64
#define CHUCK_SIZE4 256




// GPU Specific Parameter
#define SHARE_MEMSIZE 49152
#define THREADS_ONEDIM 1024
#define TEXTURE_ALIGNMENT 512
#define MAX_TEXTURE_DIM 4096

__constant__ dcomplex dev_k_wave[1];
__constant__ unsigned int dev_Facet_MaxIndx[1]; 
__constant__ unsigned int dev_Facet_MaxIndy[1];
__constant__ float dev_MaxTheta[1];
__constant__ unsigned int  dev_NumSourcePoints[1];
__constant__ unsigned int  dev_NumFieldPoints[1];
__constant__ unsigned int  dev_NumFacets[1];

texture<float4,3,cudaReadModeElementType> dev_Positions;
texture<float4,2,cudaReadModeElementType> dev_Projection_i;
texture<float4,2,cudaReadModeElementType> dev_Projection_j;
texture<float4,1,cudaReadModeElementType> dev_Projection;
texture<float2,2,cudaReadModeElementType> dev_FacetPressure;
texture<int,2,cudaReadModeElementType> dev_PathMatrix;
#endif