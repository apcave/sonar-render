#ifndef _ProjectPointToFacet
#define _ProjectPointToFacet

#include "CudaFunctions.cuh"
#include <stdio.h>
#include <iostream>
#include <fstream>
// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <device_functions.h>
#include <math.h>

extern TraceObject* MakeTraceObject()
{
	TraceObject* tmp_Trace = (TraceObject*)malloc(sizeof(TraceObject));
	return tmp_Trace;
}

struct GeoObject
{
	float3 Centriod;
	float3 Normal;
	float3 PointA;
	float3 PointB;
	float3 PointC;
	float3 uVab;
	float3 uVac;
	float3 uVba;
	float3 uVbc;
	float3 uVca;
	float3 uVcb;
	bool isFacet;
	float3 uVnormABthroughC;
	float3 uVnormACthroughB;
	float3 uVnormBCthroughA;
};

__global__ void CollisionDetectionGPU(bool* dev_CollisionMatrix, size_t pitch_Collision, GeoObject* dev_Objects)
{
	// There is one index (x) per block, where the indexes represent facets that is x is the facet number.
	// There is a grid made of blocks, where the x and y grid references represent the starting and finishing positions of rays.
	// For each x and y block identifier, comparison is made to all facets (x block index) to determine if the facet(s) x
	// are in the path of the ray.
	// Shared memory has block scope, hence PathFound is unique for each block, shared path stores the result for each x.
	// One a block has been evalutated all results compared and if there was a one or more collisions with a facet then there the
	// ray between the grid references is broken and this is recored in the CollisionMatrix.

	// Note CUDA with compute model 3 has an upper thread limit of 1024 threads per block and 1024 is the maximum thread index this bounds the maximum number of facets.
	// Compute model 5 (Currently requiring a Nividia Tesla or similar GPU) provides for a larger block index with a maximum block index of 16,777,216.
	// The design decision has been made to accept this upper bound given that it will rise with future revisions of the technology.


	__shared__ bool PathFound[MAX_THREADS_PER_BLOCK];

	float3 Ia;
	float3 Ib;

	// Get Intersection of Line and Plane
	float3 P0;
    float3 P1;
    float3 P2;
                

    // Matrix A - Less unused locations
    float a = 0;
    float b = 0;
    float c = 0;
    float d = 0;
    float e = 0;
    float f = 0;
    float g = 0;
    float h = 0;
    float i = 0;
          
    // Inverse Matrix A not divide by determinat
    float AIa = 0;
    float AIb = 0;
    float AIc = 0;
    float AId = 0;
    // AIe = (a*i-c*g)
    // AIf = -(a*h-b*g)
    float AIg = 0;
    // AIh = -(a*f-c*d)
    // AIi = (a*e-b*d)
                
    // Determinant of Matrix A
    float detA = 0;
                
    // Vector B
    float Ba = 0;
    float Bb = 0;
    float Bc = 0;
                
    // Cross Product of Inverse Maxtrix a and Vector B
    float x = 0;



	GeoObject ObjA;
	GeoObject ObjB;
	GeoObject ObjC;

	float3 Vab;
	Vab.x = 0;
	Vab.y = 0;
	Vab.z = 0;



	// a to b as a unit vector.
	float3 uVab;
	uVab.x = 0;
	uVab.y = 0;
	uVab.z = 0;

	float ThetaRayFacet_A = 0;
	float ThetaRayFacet_B = 0;
	float ThetaRayFacet_C = 0;

	// The source index is the object i.
	unsigned int SourceIndx;
	if(blockIdx.x < dev_NumSourcePoints[0])
		SourceIndx = blockIdx.x; // Source is a source point.
	else
		SourceIndx = blockIdx.x + dev_NumFieldPoints[0] + dev_NumSourcePoints[0] - 1; // Source is a facet.
	
	// The destination index is the object j.
	unsigned int DestIndx;
	// Source points cannot be destinations.
	DestIndx = blockIdx.y + dev_NumSourcePoints[0];

	bool* Collision_row = (bool*)((char*)dev_CollisionMatrix + SourceIndx * pitch_Collision);

	// The algorithm starts with the assumption that there is a path from i to k.
	PathFound[threadIdx.x] = true; 

	if( SourceIndx == DestIndx )
	{
	//	// The matrix is false by default, it is also symetrical by reciprocity and the diagonal is alway false.
	//	// This is added to speed up the collision detection by a factor of 2.
	//	// The processing block is abandoned.
		return;
	}

	// The obstical number is always a the object number of a facet, as facets are the only object type that are
	// capable of blocking the pressure transmitting from i to j.
	unsigned int obsticalNumber = threadIdx.x+dev_NumSourcePoints[0]+dev_NumFieldPoints[0];


	ObjA = dev_Objects[SourceIndx];
	ObjB = dev_Objects[DestIndx];
	ObjC = dev_Objects[obsticalNumber];

	Vab.x = ObjB.Centriod.x-ObjA.Centriod.x;
	Vab.y = ObjB.Centriod.y-ObjA.Centriod.y;
	Vab.z = ObjB.Centriod.z-ObjA.Centriod.z;

	// The distance between a and b or i and j.
	float LenVab = sqrtf(Vab.x*Vab.x+Vab.y*Vab.y+Vab.z*Vab.z);
	// printf("Length between A and B %e. (Object Numbers) A=%d, B=%d, Obstical Number %d\n",LenVab,SourceIndx,DestIndx,obsticalNumber);
	if(LenVab < 1e-6)
	{
		// If the source is very close to the destination there is no path.
		// This occurs with overlapping facets or field points that are very close to source points.
		// For the second case, there should not be a path and it causes numerical errors in this routine.
		return;
	}

	

	// a to b as a unit vector.
	uVab.x = Vab.x/LenVab;
	uVab.y = Vab.y/LenVab;
	uVab.z = Vab.z/LenVab;

	

	// A test, first is object a or i, a facet that is emmiting?
	if( ObjA.isFacet )
	{
		// This is the angle (in radians) between the ray from a to b and the normal of a.
		// The absolute causes the angle to alway be greater than zero, if the angle is less than zero
		// this simple respresents that the pressure feild is passing through the facet.
		ThetaRayFacet_A = acosf(abs(uVab.x*ObjA.Normal.x+uVab.y*ObjA.Normal.y+uVab.z*ObjA.Normal.z));
		//printf("Object Num :%d, Angle of Emmiting Ray to Facet (deg) %e, Max Theta Degrees %e\n", SourceIndx, (ThetaRayFacet/(2*PI))*360, (dev_MaxTheta[0]/(2*PI))*360);
	
		
		// If the angle is greater than the predetermined grazing angle cos(scatter theta) --> 0, as the adjacent component to the normal approaches zero.
		// This is a reality test,
		// the program uses a derivative of abs(cos(scatter theta)+cos(incedent theta))/2,
		// this results in spreading of pressure along parallel facets if MaxTheata >= 90 deg or the grazing angle is ignored.
		// Think of an infinte plane of facets RWBC facets and a source and feild point above.
		// If the Order of the reflections is 1, the feild properates from the source to the facets and back to feild point.eeewwww
		//printf("cos(theta_A) Degrees :%e. (Object Numbers) A=%d, B=%d, Obstical Number %d\n",ThetaRayFacet_A*180/PI,SourceIndx,DestIndx,obsticalNumber);

		if( ThetaRayFacet_A > dev_MaxTheta[0] )
		{
			//printf("cos(theta_A) Degrees :%e\n",ThetaRayFacet_A*180/PI);
			//printf("Unit Vector from a to b, x :%e, y :%e, z:%e\n",uVab.x,uVab.y,uVab.z);
			//printf("Unit Vector Normal Object A, x :%e, y :%e, z:%e\n",ObjA.Normal.x,ObjA.Normal.y,ObjA.Normal.z);
			//printf("Collision Detection, Theta is too large on source facet, Theta Ray =%e (rad), Theta Limit=%e (rad)\n", ThetaRayFacet_A, dev_MaxTheta[0]);
			PathFound[threadIdx.x] = false; // It is false that there is a path between A and B because angle of incedent is high
			goto EndDetect;  		        // Most likely facets are parallel
		}
			
	}

	if( ObjB.isFacet )
	{
		ThetaRayFacet_B = acosf(abs(uVab.x*ObjB.Normal.x+uVab.y*ObjB.Normal.y+uVab.z*ObjB.Normal.z));

		//printf("cos(theta_B) Degrees :%e. (Object Numbers) A=%d, B=%d, Obstical Number %d\n",ThetaRayFacet_B*180/PI,SourceIndx,DestIndx,obsticalNumber);
		
		if( ThetaRayFacet_B > dev_MaxTheta[0] )
		{
			//printf("cos(theta_B) Degrees :%e\n",ThetaRayFacet_B*180/PI);
			//printf("Unit Vector from a to b, x :%e, y :%e, z:%e\n",uVab.x,uVab.y,uVab.z);
			//printf("Unit Vector Normal Object B, x :%e, y :%e, z:%e\n",ObjB.Normal.x,ObjB.Normal.y,ObjB.Normal.z);
			//printf("Collision Detection, Theta is too large on destination facet, Theta Ray =%e (rad), Theta Limit=%e (rad)\n", ThetaRayFacet_B, dev_MaxTheta[0]);
			PathFound[threadIdx.x] = false; // It is false that there is a path between A and B because angle of incedent is high
			goto EndDetect;    			   // Most likely facets are parallel
		}
	}

	// The angles for the source and the destination need to be checked before the following cases.
	// If the source is the same as the obstical ignore.
	if( SourceIndx == obsticalNumber )
	{
		 PathFound[threadIdx.x] = true; 
		 goto EndDetect;
	}
	// If the destination is the same as the obstical ignore.
	if( DestIndx == obsticalNumber )
	{
		 PathFound[threadIdx.x] = true; 
		 goto EndDetect;
	}


	ThetaRayFacet_C = acosf(abs(uVab.x*ObjC.Normal.x+uVab.y*ObjC.Normal.y+uVab.z*ObjC.Normal.z));
	//printf("cos(theta_B) Degrees :%e. (Object Numbers) A=%d, B=%d, Obstical Number %d\n",ThetaRayFacet_B*180/PI,SourceIndx,DestIndx,obsticalNumber);
	if( ThetaRayFacet_C > dev_MaxTheta[0] )
	{
		// If the facet C is parallel to the ray, there can be no collision.
		// For facets that are in a plane previous testing of angles from the source, (also the destination) will exclude there
		// being a path from A to B.

		//printf("cos(theta_C) :%e\n",ThetaRayFacet_C);
		//printf("Unit Vector from a to b, x :%e, y :%e, z:%e\n",uVab.x,uVab.y,uVab.z);
		//printf("Unit Vector Normal Object C, x :%e, y :%e, z:%e\n",ObjC.Normal.x,ObjC.Normal.y,ObjC.Normal.z);
		//printf("Collision Detection, Theta is too large, Theta Ray =%e (rad), Theta Limit=%e (rad)\n", ThetaRayFacet_C, dev_MaxTheta[0]);

		PathFound[threadIdx.x] = true; // It is true that there is a path between A and B because angle of incedent is high
		goto EndDetect;                // Most likely the obstical facet is parallel to Vab
	}	

	Ia = ObjA.Centriod;
	Ib = ObjB.Centriod;

	// Get Intersection of Line and Plane
	P0 = ObjC.PointA;
    P1 = ObjC.PointB;
    P2 = ObjC.PointC;
                

    // Matrix A - Less unused locations
    a = Ia.x - Ib.x;
    b = P1.x - P0.x;
    c = P2.x - P0.x;
    d = Ia.y - Ib.y;
    e = P1.y - P0.y;
    f = P2.y - P0.y;
    g = Ia.z - Ib.z;
    h = P1.z - P0.z;
    i = P2.z - P0.z;
          
    // Inverse Matrix A not divide by determinant
    AIa = (e*i-f*h);
    AIb = -(d*i-f*g);
    AIc = (d*h-e*g);
    AId = -(b*i-c*h);
    // AIe = (a*i-c*g)
    // AIf = -(a*h-b*g)
    AIg = (b*f-c*e);
    // AIh = -(a*f-c*d)
    // AIi = (a*e-b*d)
                
    // Determinant of Matrix A
    detA = a*AIa+b*AIb+c*AIc;
                
    // Vector B
    Ba = Ia.x - P0.x;
    Bb = Ia.y - P0.y;
    Bc = Ia.z - P0.z;
                
    // Cross Product of Inverse Maxtrix a and Vector B
    x = (AIa*Ba+AId*Bb+AIg*Bc)/detA;
    // y = (AIb*Ba+AIe*Bb+AIh*Bc)/detA
    // z = (AIc*Ba+AIf*Bb+AIi*Bc)/detA
  
	//printf("Distance from point A: %e > 0 must be positive along the line.(Object Numbers) A=%d, B=%d, C=%d\n",x*LenVab,SourceIndx,DestIndx,obsticalNumber);
	if( x*LenVab <= 1e-6 )
	{
		// Intersection point is not between Point A and Point B.
		PathFound[threadIdx.x] = true;
		goto EndDetect;
	}

	//printf("Distance from point A: %e < %e must not be greater than the distance between the points..(Object Numbers) A=%d, B=%d, C=%d\n",x*LenVab,LenVab,SourceIndx,DestIndx,obsticalNumber);
	if( x*LenVab > LenVab )
	{
		// Intersection point is along Vab but past B.
		PathFound[threadIdx.x] = true;
		goto EndDetect;
	}

    float3 IntPntOnPlane;
	IntPntOnPlane.x = Ia.x + (Ib.x - Ia.x )*x;
	IntPntOnPlane.y = Ia.y + (Ib.y - Ia.y )*x;
	IntPntOnPlane.z = Ia.z + (Ib.z - Ia.z )*x;
	if(isnan(IntPntOnPlane.x) || isnan(IntPntOnPlane.y) || isnan(IntPntOnPlane.z))
	{
		printf("Unit Vector from a to b, x :%e, y :%e, z:%e\n",uVab.x,uVab.y,uVab.z);	
		printf("Intersection point on plane, x:%e, y:%e, z:%e\n", IntPntOnPlane.x,IntPntOnPlane.y,IntPntOnPlane.z);
		printf("Distance from A to Intersection Point :%e\n",x);
		printf("Distance from A to B :%e\n", LenVab);

		if( ObjA.isFacet )
		{
			printf("Object A is a Facet. Facet Num :%d\n", SourceIndx-dev_NumFieldPoints[0]-dev_NumSourcePoints[0]);
			printf("cos(theta_A) dregrees :%e\n",ThetaRayFacet_A*180/PI);
			printf("Unit Vector Normal Object A, x :%e, y :%e, z:%e\n",ObjA.Normal.x,ObjA.Normal.y,ObjA.Normal.z);
		}
		else
			printf("Object A is not a Facet.\n");

		if( ObjB.isFacet )
		{
			printf("Object B is a Facet. Facet Num :%d\n", DestIndx-dev_NumFieldPoints[0]-dev_NumSourcePoints[0]);
			printf("cos(theta_B) dregrees :%e\n",ThetaRayFacet_B*180/PI);
			printf("Unit Vector Normal Object B, x :%e, y :%e, z:%e\n",ObjB.Normal.x,ObjB.Normal.y,ObjB.Normal.z);	
		}
		else
			printf("Object B is not a Facet.\n");

		if( ObjC.isFacet )
		{
			printf("Object C is a Facet. Facet Number %d\n",threadIdx.x);
			printf("cos(theta_C) dregrees :%e\n",ThetaRayFacet_C*180/PI);
			printf("Unit Vector Normal Object C, x :%e, y :%e, z:%e\n",ObjC.Normal.x,ObjC.Normal.y,ObjC.Normal.z);	
		}
		else
			printf("Object C is not a Facet. This should never happen!!\n");	
	}
	//printf("Intersection point on plane, x:%e, y:%e, z:%e,(Object Numbers) A=%d, B=%d, C=%d\n", IntPntOnPlane.x,IntPntOnPlane.y,IntPntOnPlane.z,SourceIndx,DestIndx,obsticalNumber);

	// Is point co-interior to facet?

	// Vector from Point A on Facet C to the plane intersection point.
	float3 VaInt;
	VaInt.x = IntPntOnPlane.x - ObjC.PointA.x;
	VaInt.y = IntPntOnPlane.y - ObjC.PointA.y;
	VaInt.z = IntPntOnPlane.z - ObjC.PointA.z;

	// Project A->Int onto facet edge A->B
	float projIntOnAB = ObjC.uVab.x*VaInt.x+ObjC.uVab.y*VaInt.y+ObjC.uVab.z*VaInt.z;

	// Intersection point on AB of the perpendicular vector.
	float3 IntPointOnAB;
	IntPointOnAB.x = ObjC.PointA.x + ObjC.uVab.x*projIntOnAB;
	IntPointOnAB.y = ObjC.PointA.y + ObjC.uVab.y*projIntOnAB;
	IntPointOnAB.z = ObjC.PointA.z + ObjC.uVab.z*projIntOnAB;

	// Unit Vector from AB intersection through plane intersection.
	float3 uVperpABthoughInt;
	uVperpABthoughInt.x = IntPointOnAB.x - IntPntOnPlane.x;
	uVperpABthoughInt.y = IntPointOnAB.y - IntPntOnPlane.y;
	uVperpABthoughInt.z = IntPointOnAB.z - IntPntOnPlane.z;
	float LenPerp = sqrtf(uVperpABthoughInt.x*uVperpABthoughInt.x+uVperpABthoughInt.y*uVperpABthoughInt.y+uVperpABthoughInt.z*uVperpABthoughInt.z);
	uVperpABthoughInt.x = uVperpABthoughInt.x/LenPerp;
	uVperpABthoughInt.y = uVperpABthoughInt.y/LenPerp;
	uVperpABthoughInt.z = uVperpABthoughInt.z/LenPerp;

	if( fabs(ObjC.uVnormABthroughC.x-uVperpABthoughInt.x) > 1e-6 || fabs(ObjC.uVnormABthroughC.y-uVperpABthoughInt.y) > 1e-6 || fabs(ObjC.uVnormABthroughC.z-uVperpABthoughInt.z) > 1e-6 )
	{
		// The plane interection point is not on the same side of vAB as point C.
		PathFound[threadIdx.x] = true;
		goto EndDetect;
	}

	// Vector from Point A on Facet C to the plane intersection point.
	//float3 VaInt;
	//VaInt.x = IntPntOnPlane.x - ObjC.PointA.x;
	//VaInt.y = IntPntOnPlane.y - ObjC.PointA.y;
	//VaInt.z = IntPntOnPlane.z - ObjC.PointA.z;

	// Project A->Int onto facet edge A->C
	float projIntOnAC = ObjC.uVac.x*VaInt.x+ObjC.uVac.y*VaInt.y+ObjC.uVac.z*VaInt.z;

	// Intersection point on AC of the perpendicular vector.
	float3 IntPointOnAC;
	IntPointOnAC.x = ObjC.PointA.x + ObjC.uVac.x*projIntOnAC;
	IntPointOnAC.y = ObjC.PointA.y + ObjC.uVac.y*projIntOnAC;
	IntPointOnAC.z = ObjC.PointA.z + ObjC.uVac.z*projIntOnAC;

	// Unit Vector from AC intersection through plane intersection.
	float3 uVperpACthoughInt;
	uVperpACthoughInt.x = IntPointOnAC.x - IntPntOnPlane.x;
	uVperpACthoughInt.y = IntPointOnAC.y - IntPntOnPlane.y;
	uVperpACthoughInt.z = IntPointOnAC.z - IntPntOnPlane.z;
	LenPerp = sqrtf(uVperpACthoughInt.x*uVperpACthoughInt.x+uVperpACthoughInt.y*uVperpACthoughInt.y+uVperpACthoughInt.z*uVperpACthoughInt.z);
	uVperpACthoughInt.x = uVperpACthoughInt.x/LenPerp;
	uVperpACthoughInt.y = uVperpACthoughInt.y/LenPerp;
	uVperpACthoughInt.z = uVperpACthoughInt.z/LenPerp;

	if( fabs(ObjC.uVnormACthroughB.x-uVperpACthoughInt.x) > 1e-6 || fabs(ObjC.uVnormACthroughB.y-uVperpACthoughInt.y) > 1e-6 || fabs(ObjC.uVnormACthroughB.z-uVperpACthoughInt.z) > 1e-6 )
	{
		// The plane interection point is not on the same side of vAC as point B.
		PathFound[threadIdx.x] = true;
		goto EndDetect;
	}

	// Vector from Point B on Facet C to the plane intersection point.
	float3 VbInt;
	VbInt.x = IntPntOnPlane.x - ObjC.PointB.x;
	VbInt.y = IntPntOnPlane.y - ObjC.PointB.y;
	VbInt.z = IntPntOnPlane.z - ObjC.PointB.z;

	// Project A->Int onto facet edge A->C
	float projIntOnBC = ObjC.uVbc.x*VbInt.x+ObjC.uVbc.y*VbInt.y+ObjC.uVbc.z*VbInt.z;

	// Intersection point on AC of the perpendicular vector.
	float3 IntPointOnBC;
	IntPointOnBC.x = ObjC.PointB.x + ObjC.uVbc.x*projIntOnBC;
	IntPointOnBC.y = ObjC.PointB.y + ObjC.uVbc.y*projIntOnBC;
	IntPointOnBC.z = ObjC.PointB.z + ObjC.uVbc.z*projIntOnBC;

	// Unit Vector from AC intersection through plane intersection.
	float3 uVperpBCthoughInt;
	uVperpBCthoughInt.x = IntPointOnBC.x - IntPntOnPlane.x;
	uVperpBCthoughInt.y = IntPointOnBC.y - IntPntOnPlane.y;
	uVperpBCthoughInt.z = IntPointOnBC.z - IntPntOnPlane.z;
	LenPerp = sqrtf(uVperpBCthoughInt.x*uVperpBCthoughInt.x+uVperpBCthoughInt.y*uVperpBCthoughInt.y+uVperpBCthoughInt.z*uVperpBCthoughInt.z);
	uVperpBCthoughInt.x = uVperpBCthoughInt.x/LenPerp;
	uVperpBCthoughInt.y = uVperpBCthoughInt.y/LenPerp;
	uVperpBCthoughInt.z = uVperpBCthoughInt.z/LenPerp;

	if( fabs(ObjC.uVnormBCthroughA.x-uVperpBCthoughInt.x) > 1e-6 || fabs(ObjC.uVnormBCthroughA.y-uVperpBCthoughInt.y) > 1e-6 || fabs(ObjC.uVnormBCthroughA.z-uVperpBCthoughInt.z) > 1e-6 )
	{
		// The plane interection point is not on the same side of vBC as point A.
		PathFound[threadIdx.x] = true;
		goto EndDetect;
	}

	// The point is interior to the Facet C.
	// printf("Collision Detection, Ray intersects Facet k.\n");
	PathFound[threadIdx.x] = false;

EndDetect:
	
	// All facets must not collide with path in order for the path to be clear.
	//if( PathFound[threadIdx.x] )
	//	printf("PathFound[%d] = true,  post EndDetect. (Object Numbers) A=%d, B=%d, Obstical Number %d\n",threadIdx.x,SourceIndx,DestIndx,obsticalNumber);
	//else
	//	printf("PathFound[%d] = false, post EndDetect. (Object Numbers) A=%d, B=%d, Obstical Number %d\n",threadIdx.x,SourceIndx,DestIndx,obsticalNumber);

	__syncthreads();
	//PathFound[threadIdx.x] = true;

	unsigned int jj;
	if(threadIdx.x%CHUCK_SIZE == 0)
	{
		
		#pragma unroll ROLL_SIZE
		for (jj = threadIdx.x+1; threadIdx.x+CHUCK_SIZE > jj && blockDim.x > jj; jj++)
		{
			PathFound[threadIdx.x] = PathFound[threadIdx.x] && PathFound[jj]; 
		}
		
	}
	
	if(threadIdx.x%CHUCK_SIZE2 == 0)
	{
		__syncthreads();
		#pragma unroll ROLL_SIZE
		for (jj = threadIdx.x+CHUCK_SIZE; threadIdx.x+CHUCK_SIZE2 > jj && blockDim.x > jj; jj += CHUCK_SIZE)
		{
			PathFound[threadIdx.x] = PathFound[threadIdx.x] && PathFound[jj];
		}
	}
	
	if(threadIdx.x%CHUCK_SIZE3 == 0)
	{
		__syncthreads();
		#pragma unroll ROLL_SIZE
		for (jj = threadIdx.x+CHUCK_SIZE2; threadIdx.x+CHUCK_SIZE3 > jj && blockDim.x > jj; jj += CHUCK_SIZE2)
		{
			PathFound[threadIdx.x] = PathFound[threadIdx.x] && PathFound[jj];
		}
	}

	if(threadIdx.x%CHUCK_SIZE4 == 0)
	{
		__syncthreads();
		#pragma unroll ROLL_SIZE
		for (jj = threadIdx.x+CHUCK_SIZE3; threadIdx.x+CHUCK_SIZE4 > jj && blockDim.x > jj; jj += CHUCK_SIZE3)
		{
			PathFound[threadIdx.x] = PathFound[threadIdx.x] && PathFound[jj];
		}
	}

	if( threadIdx.x == 0 )
	{
		#pragma unroll CHUCK_SIZE
		// Note the block dimension should never exceed the value of the macro MAX_THREADS_PER_BLOCK.
		for (jj = 0; blockDim.x > jj; jj += CHUCK_SIZE4)
		{
			PathFound[0] = PathFound[0] && PathFound[jj];
		}

		// Matrix is sysmetrical about the zero value diagonal.
		// This comes from graph theory, and resiprosity is there is a pressure path from i to j,
		// then there is a pressure path j to i, by definition there is no pressure path from i to i.
		// Sources or projectors emmit but to not recieve it is possible for them the recieve from the Path Matrix,
		// although other parts of the program exclude this possibility.
		// Feild points, are absobers they accumulate pressure although they do not transit, in concept they are observation points
		// that do not affect the pressure feild.
		// Facets are the third kind of object, they accumulate and absorb the pressure feild and then transimit, reflect or do both
		// to the pressure feild.

		if( PathFound[0] )
			printf("Collision Detection     Path Detected between, (Object Numbers) i=%d, j=%d\n", SourceIndx, DestIndx);
		else
			printf("Collision Detection Path Not Detected between, (Object Numbers) i=%d, j=%d\n", SourceIndx, DestIndx);


		Collision_row = (bool*)((char*)dev_CollisionMatrix + SourceIndx * pitch_Collision);
		Collision_row[DestIndx] = PathFound[0];
		// Collision_row[DestIndx] = true;

		Collision_row = (bool*)((char*)dev_CollisionMatrix + DestIndx * pitch_Collision);
		Collision_row[SourceIndx] = PathFound[0];
		// Collision_row[SourceIndx] = true;
	}
}

__global__ void ScanProjectFacetToFacetGPU(dcomplex* dev_LastPixelPressure_j,unsigned int FacetNum_i, unsigned int FacetNum_j, unsigned int pitch_Pixels)
{
	__shared__ dcomplex WorkingPressure[MAX_THREADS_PER_BLOCK];

	unsigned int Start_i = threadIdx.x*POINTS_PER_THREAD;
	unsigned int Start_j = threadIdx.y*POINTS_PER_THREAD;

	dcomplex SubPres;
	SubPres.r = 0;
	SubPres.i = 0;

	unsigned int ii = threadIdx.x+blockDim.x*threadIdx.y;
	unsigned int Indx_j = blockIdx.x;
	unsigned int Indy_j = blockIdx.y;

	if( Indx_j < dev_Facet_MaxIndx[0] && Indy_j < dev_Facet_MaxIndy[0] )
	{

		// Projecting from All on Facet i to one point on Facet j....

		float4 FacetPoint_j = text3D(dev_Positions,FacetNum_j,Indx_j,Indy_j);
		float4 ProjectToCentriod_i = tex2D(dev_Projection_i,Indx_j,Indy_j);

		#pragma unroll POINTS_PER_THREAD
		for (unsigned int i = 0; i < POINTS_PER_THREAD; i++)
		{
			unsigned int Indx = Start_i+i;
			if( Indx < dev_Facet_MaxIndx[0] )
			{
				#pragma unroll POINTS_PER_THREAD
				for (unsigned int j = 0; j < POINTS_PER_THREAD; j++)
				{
					unsigned int Indy = Start_j+j;
					if( Indy < dev_Facet_MaxIndy[0] )
					{
						float4 FacetPoint_i = tex3D(dev_Positions,FacetNum_i,Indx,Indy);
						
						if( FacetPoint_i.w > 0 && FacetPoint_j.w > 0 && !isnan(ProjectToCentriod_i.w) && !isnan(FacetPoint_j.w) )
						{
							float3 vpS;
							vpS.x	       = ProjectToCentriod_i.x - FacetPoint_i.x;
							vpS.y	       = ProjectToCentriod_i.y - FacetPoint_i.y;
							vpS.z	       = ProjectToCentriod_i.z - FacetPoint_i.z;
							float R        = sqrtf(vpS.x*vpS.x+vpS.y*vpS.y+vpS.z*vpS.z);
							

							// NB: Formula is exp(1i*k_wave*R)

							dcomplex Phase;
							Phase.r = cosf(R*dev_k_wave[0].r)*expf(-R*dev_k_wave[0].i);
							Phase.i = sinf(R*dev_k_wave[0].r)*expf(-R*dev_k_wave[0].i);
							
									//*c.r=a.r*b.r-a.i*b.i;
									//*c.i=a.i*b.r+a.r*b.i;
	
							float a = -ProjectToCentriod_i.w*FacetPoint_j.w;
		

							dcomplex b;
							b.r =  (dev_k_wave[0].r*Phase.r-dev_k_wave[0].i*Phase.i)*a;
							b.i =  (dev_k_wave[0].i*Phase.r+dev_k_wave[0].r*Phase.i)*a;
							

							// Facet Pressure on i, referenced to pixels x, y
							float2 FacetPress = tex2D(dev_FacetPressure,Indx,Indy);
							SubPres.r += b.r*FacetPress.x-b.i*FacetPress.y;
							SubPres.i += b.i*FacetPress.x+b.r*FacetPress.y;
							

							if( isnan(SubPres.r) || isnan(SubPres.i) )
							{
								printf("Facet 2 Facet, ProjectToCentriod_i.w: %e\n",ProjectToCentriod_i.w);
								printf("Facet 2 Facet, FacetPoint_j.w: %e\n",FacetPoint_j.w);
								printf("Facet 2 Facet, R: %e\n",R);
								printf("Facet 2 Facet, Phase Real: %e, Imag :%e\n",Phase.r,Phase.i);
								printf("Facet 2 Facet, a:%e\n",a);
								printf("Facet 2 Facet, Phase Real: %e, Imag :%e\n",Phase.r,Phase.i);
								printf("Facet 2 Facet, SubPres Real: %e, Imag :%e\n",SubPres.r,SubPres.i);
							}

						}
					}
				}
			}
		}
	}
	if(ii >= MAX_THREADS_PER_BLOCK)
		printf("****************Errror Allocating Shared Memory!!!!!!!!!!!!!!!\n");
	WorkingPressure[ii].r = SubPres.r;
	WorkingPressure[ii].i = SubPres.i;

	__syncthreads();

	unsigned int jj;
	dcomplex PressSum;

	if(ii%CHUCK_SIZE == 0)
	{
		PressSum.r = 0;
		PressSum.i = 0;

		#pragma unroll ROLL_SIZE
		for (jj = ii+1; ii+CHUCK_SIZE > jj; jj++)
		{
			PressSum.r += WorkingPressure[jj].r;
			PressSum.i += WorkingPressure[jj].i;
		}
		WorkingPressure[ii].r += PressSum.r;
		WorkingPressure[ii].i += PressSum.i;
		
	}
	
	if(ii%CHUCK_SIZE2 == 0)
	{
		__syncthreads();
		PressSum.r = 0;
		PressSum.i = 0;

		#pragma unroll ROLL_SIZE
		for (jj = ii+CHUCK_SIZE; ii+CHUCK_SIZE2 > jj; jj += CHUCK_SIZE)
		{
			PressSum.r += WorkingPressure[jj].r;
			PressSum.i += WorkingPressure[jj].i;
		}
		WorkingPressure[ii].r += PressSum.r;
		WorkingPressure[ii].i += PressSum.i;
		
	}
	
	if(ii%CHUCK_SIZE3 == 0)
	{
		__syncthreads();
		PressSum.r = 0;
		PressSum.i = 0;

		#pragma unroll ROLL_SIZE
		for (jj = ii+CHUCK_SIZE2; ii+CHUCK_SIZE3 > jj; jj += CHUCK_SIZE2)
		{
			PressSum.r += WorkingPressure[jj].r;
			PressSum.i += WorkingPressure[jj].i;
		}
		WorkingPressure[ii].r += PressSum.r;
		WorkingPressure[ii].i += PressSum.i;
		
	}

	if(ii%CHUCK_SIZE4 == 0)
	{
		__syncthreads();
		PressSum.r = 0;
		PressSum.i = 0;

		#pragma unroll ROLL_SIZE
		for (jj = ii+CHUCK_SIZE3; ii+CHUCK_SIZE4 > jj; jj += CHUCK_SIZE3)
		{
			PressSum.r += WorkingPressure[jj].r;
			PressSum.i += WorkingPressure[jj].i;
		}
		WorkingPressure[ii].r += PressSum.r;
		WorkingPressure[ii].i += PressSum.i;
		
	}

	if( ii == 0 )
	{
		PressSum.r = 0;
		PressSum.i = 0;

		#pragma unroll CHUCK_SIZE
		for (jj = 0; blockDim.x*blockDim.y > jj; jj += CHUCK_SIZE4)
		{
			PressSum.r += WorkingPressure[jj].r;
			PressSum.i += WorkingPressure[jj].i;
		}
		dcomplex* pressure_row = (dcomplex*)((char*)dev_LastPixelPressure_j + Indy_j * pitch_Pixels);
		pressure_row[Indx_j].r = PressSum.r;
		pressure_row[Indx_j].i = PressSum.i;

		if( isnan(PressSum.r) | isnan(PressSum.i) )
			printf("Facet 2 Facet, block projection Pressure Value, Real: %e, Imag: %e\n", PressSum.r, PressSum.i);
	}
}

__global__ void ScanProjectFacetToPointGPU(dcomplex* dev_OutPressure, unsigned int FacetNum)
{
	__shared__ dcomplex WorkingPressure[MAX_THREADS_PER_BLOCK];

	unsigned int Start_i = threadIdx.x*POINTS_PER_THREAD;
	unsigned int Start_j = threadIdx.y*POINTS_PER_THREAD;

	//printf("************ FacetNum :%d\n", FacetNum); 
	//printf("threadIdx.x :%d, threadIdx.y :%d, threadIdx.z :%d\n",threadIdx.x,threadIdx.y,threadIdx.z);
	//printf("Facet Limits Start_i:%d, Start_j:%d\n",dev_Facet_MaxIndx[0],dev_Facet_MaxIndy[0]);
	//printf("Start_i:%d, Start_j:%d\n",Start_i,Start_j);
	
	dcomplex SubPres;
	SubPres.r = 0;
	SubPres.i = 0;

	unsigned int ii = threadIdx.x+blockDim.x*threadIdx.y;
	//printf("BlockDim x:%d, y :%d, ThreadIndx x:%d, y:%d\n",blockDim.x,blockDim.y,threadIdx.x,threadIdx.y);
	//printf("Max Facet Indexes x:%d, y :%d\n",dev_Facet_MaxIndx[0],dev_Facet_MaxIndy[0]);
	//printf("Max Linear Block Index:%d\n", ii);
	//printf("Max Facet Min Index x:%d y:%d\n", Start_i,Start_j);

	int isPathToFeild = tex2D(dev_PathMatrix,FacetNum+dev_NumSourcePoints[0]+dev_NumFieldPoints[0], blockIdx.x+dev_NumSourcePoints[0]);

	//if(isPathToFeild == 1)
	//	printf("There is a path to the field Point. Facet Num :%d, Feild Point Num %d, Facet Object Num  %d, Feild Point Num %d\n",FacetNum,blockIdx.x,FacetNum+dev_NumSourcePoints[0]+dev_NumFieldPoints[0], blockIdx.x+dev_NumSourcePoints[0]);
	//else
	//	printf("No path to the field Point.         Facet Num :%d, Feild Point Num %d, Facet Object Num  %d, Feild Point Num %d\n",FacetNum,blockIdx.x,FacetNum+dev_NumSourcePoints[0]+dev_NumFieldPoints[0], blockIdx.x+dev_NumSourcePoints[0]);
	

	if(isPathToFeild == 1)
	{
	#pragma unroll POINTS_PER_THREAD
	for (unsigned int i = 0; i < POINTS_PER_THREAD; i++)
	{
		unsigned int Indx = Start_i+i;
		if( Indx < dev_Facet_MaxIndx[0] )
		{
			#pragma unroll POINTS_PER_THREAD
			for (unsigned int j = 0; j < POINTS_PER_THREAD; j++)
			{
				unsigned int Indy = Start_j+j;
				if( Indy < dev_Facet_MaxIndy[0] )
				{
					float4 ProjectToCentriod = tex1D(dev_Projection,blockIdx.x);
					float4 FacetPoint = tex3D(dev_Positions,FacetNum,Indx,Indy);
					//if( FacetPoint.w > 0 && !isnan(ProjectToCentriod.w) )
					if( FacetPoint.w > 0 )
					{
						float3 vpS;
						vpS.x	=  FacetPoint.x-ProjectToCentriod.x;
						vpS.y	=  FacetPoint.y-ProjectToCentriod.y;
						vpS.z	=  FacetPoint.z-ProjectToCentriod.z;
						float R = sqrtf(vpS.x*vpS.x+vpS.y*vpS.y+vpS.z*vpS.z);

						// NB: Formula is exp(1i*k_wave*R)
						dcomplex Phase;
						Phase.r = cosf(R*dev_k_wave[0].r)*expf(-R*dev_k_wave[0].i);
						Phase.i = sinf(R*dev_k_wave[0].r)*expf(-R*dev_k_wave[0].i);

								//*c.r=a.r*b.r-a.i*b.i;
								//*c.i=a.i*b.r+a.r*b.i;
						//float Cos_Inc = tex2D(dev_CosInc,Indx,Indy);
						//printf("Cos Inc :%f\n",Cos_Inc);
						//float a = -((abs(Cos_Inc)+abs(Cos_Scat)))/(4*PI*R);

						float a = -ProjectToCentriod.w;

						dcomplex b;
						b.r =  (dev_k_wave[0].r*Phase.r-dev_k_wave[0].i*Phase.i)*a;
						b.i =  (dev_k_wave[0].i*Phase.r+dev_k_wave[0].r*Phase.i)*a;
	
						float2 FacetPress = tex2D(dev_FacetPressure,Indx,Indy);

						//printf("Pressure, Real:%f, Imag:%f\n",FacetPress.x,FacetPress.y);

						SubPres.r += b.r*FacetPress.x-b.i*FacetPress.y;
						SubPres.i += b.i*FacetPress.x+b.r*FacetPress.y;

						//printf("Ind Max :%d, :%d, Ind Val :%d, :%d, Point Val:%f, y :%f, z :%f, w :%f\n",dev_Facet_MaxIndx[0],dev_Facet_MaxIndy[0],Indx, Indy, FacetPoint.x,FacetPoint.y,FacetPoint.z,FacetPoint.w);
						
						if( isnan(SubPres.r) || isnan(SubPres.i) )
						{
							printf("Facet Indexes x:%d, y :%d\n",Indx,Indy);
							printf("Facet Point x             :%f, y :%f, z :%f, w :%f\n",FacetPoint.x,FacetPoint.y,FacetPoint.z,FacetPoint.w);
							printf("ProjectToCentriod Point x :%f, y :%f, z :%f, w :%f\n",ProjectToCentriod.x,ProjectToCentriod.y,ProjectToCentriod.z,ProjectToCentriod.w);
							printf("FacetPress Pressure, Real :%f, Imag :%f\n",FacetPress.x,FacetPress.y);
						}
					}
				}
			}
		}
	}
	}
	//if(ii >= MAX_THREADS_PER_BLOCK)
	//	printf("****************Errror Allocating Shared Memory!!!!!!!!!!!!!!!\n");
	WorkingPressure[ii].r = SubPres.r;
	WorkingPressure[ii].i = SubPres.i;

	if( isnan(SubPres.r) || isnan(SubPres.i) )
		printf("********Facet 2 Point, block mesh Pressure Value, Real: %e, Imag: %e\n", SubPres.r, SubPres.i);


	__syncthreads();

	unsigned int jj;
	dcomplex PressSum;

	if(ii%CHUCK_SIZE == 0)
	{
		PressSum.r = 0;
		PressSum.i = 0;

		#pragma unroll ROLL_SIZE
		for (jj = ii+1; ii+CHUCK_SIZE > jj; jj++)
		{
			PressSum.r += WorkingPressure[jj].r;
			PressSum.i += WorkingPressure[jj].i;
		}
		WorkingPressure[ii].r += PressSum.r;
		WorkingPressure[ii].i += PressSum.i;
	}
	
	if(ii%CHUCK_SIZE2 == 0)
	{
		__syncthreads();
		PressSum.r = 0;
		PressSum.i = 0;

		#pragma unroll ROLL_SIZE
		for (jj = ii+CHUCK_SIZE; ii+CHUCK_SIZE2 > jj; jj += CHUCK_SIZE)
		{
			PressSum.r += WorkingPressure[jj].r;
			PressSum.i += WorkingPressure[jj].i;
		}
		WorkingPressure[ii].r += PressSum.r;
		WorkingPressure[ii].i += PressSum.i;
		
	}
	
	if(ii%CHUCK_SIZE3 == 0)
	{
		__syncthreads();
		//printf("Thread %d, made it to Chuck Size 3 loop.\n",ii);
		PressSum.r = 0;
		PressSum.i = 0;

		#pragma unroll ROLL_SIZE
		for (jj = ii+CHUCK_SIZE2; ii+CHUCK_SIZE3 > jj; jj += CHUCK_SIZE2)
		{
			PressSum.r += WorkingPressure[jj].r;
			PressSum.i += WorkingPressure[jj].i;
		}
		WorkingPressure[ii].r += PressSum.r;
		WorkingPressure[ii].i += PressSum.i;
		
	}

	if(ii%CHUCK_SIZE4 == 0)
	{
		__syncthreads();
		//printf("Thread %d, made it to Chuck Size 4 loop.\n",ii);
		PressSum.r = 0;
		PressSum.i = 0;

		#pragma unroll ROLL_SIZE
		for (jj = ii+CHUCK_SIZE3; ii+CHUCK_SIZE4 > jj; jj += CHUCK_SIZE3)
		{
			PressSum.r += WorkingPressure[jj].r;
			PressSum.i += WorkingPressure[jj].i;
		}
		WorkingPressure[ii].r += PressSum.r;
		WorkingPressure[ii].i += PressSum.i;
	}

	if( ii == 0 )
	{
		//printf("Thread %d, made it to the final loop.\n",ii);
		PressSum.r = 0;
		PressSum.i = 0;

		#pragma unroll CHUCK_SIZE
		for (jj = 0; blockDim.x*blockDim.y > jj; jj += CHUCK_SIZE4)
		{
			PressSum.r += WorkingPressure[jj].r;
			PressSum.i += WorkingPressure[jj].i;
		}

		dev_OutPressure[blockIdx.x].r += PressSum.r;
		dev_OutPressure[blockIdx.x].i += PressSum.i;
		
		if( isnan(dev_OutPressure[blockIdx.x].r) | isnan(dev_OutPressure[blockIdx.x].i) )
			printf("Facet 2 Point, block projection Pressure Value, Real: %e, Imag: %e\n", PressSum.r, PressSum.i);
	}
}


bool** MakeCollisionDectionMatrix( PointData** SourcePoints, PointData** FeildPoints, FacetData** Facets,
								unsigned int NumSourcePoints, unsigned int NumFieldPoints, unsigned int NumFacets,float MaxTheata)
{
	
	cudaMemcpyToSymbol(dev_MaxTheta,&MaxTheata,sizeof(float));
	cudaMemcpyToSymbol(dev_NumSourcePoints,&NumSourcePoints,sizeof(unsigned int));
	cudaMemcpyToSymbol(dev_NumFieldPoints,&NumFieldPoints,sizeof(unsigned int));
	cudaMemcpyToSymbol(dev_NumFacets,&NumFacets,sizeof(unsigned int));

	printf("Making Symbols in MakeCollisionDectionMatrix: %s\n", cudaGetErrorString(cudaGetLastError()));



	unsigned int TotalNumObjects = NumSourcePoints+NumFieldPoints+NumFacets;

	if( NumFacets > MAX_THREADS_PER_BLOCK )
	{
		printf("More facets in model than can be managed by current CUDA compute mode Limit:%d, Number of Facets :%d\n",MAX_THREADS_PER_BLOCK,NumFacets);
		return NULL;
	}

	dim3 cudaBlockSize;
	cudaBlockSize.x = NumFacets; // Objects that can break or collide with the rays
	cudaBlockSize.y = 1;
	cudaBlockSize.z = 1;

    dim3 cudaGridSize;
	cudaGridSize.x = NumSourcePoints+NumFacets; // Source points of rays to be tested.
	cudaGridSize.y = NumFieldPoints+NumFacets;  // Destination points of rays to be tested.
	cudaGridSize.z = 1;

	printf("Collision Detection Total Number of Rays to Trace :%d\n",cudaGridSize.x*cudaGridSize.y);

	GeoObject* GeoObjects = (GeoObject*)malloc(sizeof(GeoObject)*TotalNumObjects);

	for(unsigned int i = 0; TotalNumObjects > i; i++)
	{
		
		if(i < NumSourcePoints)
		{
			// Object is a source point
			GeoObjects[i].Centriod = SourcePoints[i]->PointLocation;
			GeoObjects[i].isFacet = false;
		}
		else
		{
			if(i < NumSourcePoints+NumFieldPoints)
			{
				// Object is a Field Point.
				GeoObjects[i].Centriod = FeildPoints[i-NumSourcePoints]->PointLocation;
				GeoObjects[i].isFacet = false;
			}
			else
			{
				FacetData* t_Fac = Facets[i-NumSourcePoints-NumFieldPoints];
				// Object is a facet.
				GeoObjects[i].Centriod = t_Fac->Centriod;
				GeoObjects[i].Normal   = t_Fac->Normal;
				GeoObjects[i].PointA   = t_Fac->Point_A;
				GeoObjects[i].PointB   = t_Fac->Point_B;
				GeoObjects[i].PointC   = t_Fac->Point_C;
				GeoObjects[i].uVab	   = GeoMath::MakeUnitVectorSc(t_Fac->Point_A,t_Fac->Point_B);
				GeoObjects[i].uVac     = GeoMath::MakeUnitVectorSc(t_Fac->Point_A,t_Fac->Point_C);
				GeoObjects[i].uVba     = GeoMath::MakeUnitVectorSc(t_Fac->Point_B,t_Fac->Point_A);
				GeoObjects[i].uVbc     = GeoMath::MakeUnitVectorSc(t_Fac->Point_B,t_Fac->Point_C);
				GeoObjects[i].uVca     = GeoMath::MakeUnitVectorSc(t_Fac->Point_C,t_Fac->Point_A);
				GeoObjects[i].uVcb     = GeoMath::MakeUnitVectorSc(t_Fac->Point_C,t_Fac->Point_B);
				GeoObjects[i].uVnormABthroughC = GeoMath::MakeUnitNormSc(GeoObjects[i].uVab,GeoObjects[i].PointA,GeoObjects[i].PointC);
				GeoObjects[i].uVnormACthroughB = GeoMath::MakeUnitNormSc(GeoObjects[i].uVac,GeoObjects[i].PointA,GeoObjects[i].PointB);
				GeoObjects[i].uVnormBCthroughA = GeoMath::MakeUnitNormSc(GeoObjects[i].uVbc,GeoObjects[i].PointB,GeoObjects[i].PointA);


				GeoObjects[i].isFacet = true;
			}
		}
	}

	cudaError_t cudaStatus;
	GeoObject* dev_Objects;
	cudaStatus = cudaMalloc((void **)&dev_Objects, (size_t)(TotalNumObjects*sizeof(GeoObject)));
	if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!\n");
        return NULL;
    }

	cudaStatus = cudaMemcpy(dev_Objects,GeoObjects,TotalNumObjects*sizeof(GeoObject),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy2D Collision Matrix to Device Failed!\n");
        return NULL;
    }

	bool* dev_CollisionMatrix;
	size_t* pitch_Collision = new size_t;
	cudaStatus = cudaMallocPitch((void **)&dev_CollisionMatrix, pitch_Collision,
									(size_t)(TotalNumObjects*sizeof(bool)),(size_t)(TotalNumObjects));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!\n");
        return NULL;
    }

	// This data only host copy h_CollisionMatrix is used to initialize the device copy and
	// store the collision values for use on the host.
	// This data could be syncronized using CUDA shared memory (host, device scoped version) 
	// this has not been done as it is initalized once and then read from many times.
	bool* h_CollisionMatrix = (bool*)malloc(sizeof(bool)*TotalNumObjects*TotalNumObjects);
	for(unsigned int i = 0; TotalNumObjects*TotalNumObjects > i; i++)
	{
		h_CollisionMatrix[i] = false;
	}

	cudaStatus = cudaMemcpy2D(dev_CollisionMatrix,pitch_Collision[0],h_CollisionMatrix,
				TotalNumObjects*sizeof(bool),TotalNumObjects*sizeof(bool),TotalNumObjects,cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy2D Collision Matrix to Device Failed!\n");
        return NULL;
    }



	unsigned int SharedMemorySize = MAX_THREADS_PER_BLOCK*sizeof(bool);
	if( SHARE_MEMSIZE < SharedMemorySize )
		printf("Total Shared Memory Size Per Block Exceeds Maxium Specification");
																		
	CollisionDetectionGPU<<<cudaGridSize, cudaBlockSize, SharedMemorySize>>>(dev_CollisionMatrix,pitch_Collision[0],dev_Objects);
									  

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("CollisionDetectionGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return NULL;
	}
    
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("CollisionDetectionGPU returned error code %d after launching!\n", cudaStatus);
		return NULL;
	}


	cudaStatus = cudaMemcpy2D(h_CollisionMatrix,TotalNumObjects*sizeof(bool),dev_CollisionMatrix,
				pitch_Collision[0],TotalNumObjects*sizeof(bool),TotalNumObjects,cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy2D Collision Matrix to Device Failed!\n");
        return NULL;
    }

	// PathMatrix is persistant and unique, it has been implemented with 2 mallocs, PathMatrix for pointers and h_CollisionMatrix
	// for data. freeing h_CollisionMatrix will result in the removal of the data.
	// PathMatrix is not cleaned up as it persists to the end of the process.
	bool** PathMatrix = (bool**)malloc(sizeof(bool*)*TotalNumObjects);
	for(unsigned int i = 0; TotalNumObjects > i; i++)
	{
		PathMatrix[i] = (bool*)((char*)h_CollisionMatrix + i*TotalNumObjects*sizeof(bool));
	}

	//printf("Printing PathMatrix *****************************\n");
	//for(unsigned int i = 0; TotalNumObjects > i; i++)
	//{
	//	for(unsigned int j = 0; TotalNumObjects > j; j++)
	//	{
	//		if( PathMatrix[i][j] )
	//			printf("PathMatrix[%d][%d] = true\n",i,j);

	////		else
	////			printf("PathMatrix[%d][%d] = false\n",i,j);
	//	}
	//}

	//printf("Printing h_CollisionMatrix *****************************\n");
	//for(unsigned int i = 0; TotalNumObjects > i; i++)
	//{
	//	bool* CollisionColumn = (bool*)((char*)h_CollisionMatrix + i*TotalNumObjects*sizeof(bool));
	//	for(unsigned int j = 0; TotalNumObjects > j; j++)
	//	{
	//		if( CollisionColumn[j] )
	//			printf("h_CollisionMatrix[%d][%d] = true\n",i,j);

	////		else
	////			printf("PathMatrix[%d][%d] = false\n",i,j);
	//	}
	//}
	int** t_Path = (int**)malloc(sizeof(int)*TotalNumObjects*TotalNumObjects);
	for(unsigned int i = 0; TotalNumObjects > i; i++)
	{
		bool* CollisionColumn = (bool*)((char*)h_CollisionMatrix + i*TotalNumObjects*sizeof(bool));
		int* t_Path_Column = (int*)((char*)t_Path + i*TotalNumObjects*sizeof(int));
		for(unsigned int j = 0; TotalNumObjects > j; j++)
		{	
			if( CollisionColumn[j] )
				t_Path_Column[j] = 1;
			else
				t_Path_Column[j] = 0;
		}
	}

	// The collision detection matrix is stored as a texture for later use.
	// Texture are supposed to be numbers and so the data is cast as an unsigned int.
	// To retrive a bool from the texture on the device to requires casting again.
	cudaArray* d_PathArray; 
	cudaChannelFormatDesc d_Pathchannel;
	d_Pathchannel=cudaCreateChannelDesc<int>();  
	cudaMallocArray(&d_PathArray,&d_Pathchannel,TotalNumObjects,TotalNumObjects); 
	cudaMemcpy2DToArray(d_PathArray,0,0,t_Path,TotalNumObjects*sizeof(int),
								TotalNumObjects*sizeof(int),TotalNumObjects,cudaMemcpyHostToDevice);
	dev_PathMatrix.filterMode=cudaFilterModePoint;
	dev_PathMatrix.addressMode[0]=cudaAddressModeClamp;
	dev_PathMatrix.addressMode[1]=cudaAddressModeClamp;
	dev_PathMatrix.normalized = false;    // access with integer texture coordinates
	cudaBindTextureToArray(dev_PathMatrix,d_PathArray);
	printf("Making Path Matrix Texture, Error: %s\n", cudaGetErrorString(cudaGetLastError()));


	cudaFree(dev_Objects);
	//cudaFree(dev_CollisionMatrix);
	free(GeoObjects);

	return(PathMatrix);
}

int ScanProjectFacetToPoint( PointData** DestPoints, FacetData* Facet, TraceObject* TraceOb, bool** PathMatrix, 
													unsigned int NumSourcePoints, unsigned int NumFeildPoints)
{
	dcomplex* dev_PressureOut = 0;
	float4* ProjectToCentriod;
	cudaError_t cudaStatus;
	PointData* DestPoint;

	cudaArray* d_PressArray; 
	cudaChannelFormatDesc d_Presschannel;
	d_Presschannel=cudaCreateChannelDesc<float2>();  
	cudaMallocArray(&d_PressArray,&d_Presschannel,Facet->NumXpnts,Facet->NumYpnts); 
	cudaMemcpy2DToArray(d_PressArray,0,0,TraceOb->dev_LastPixelPressure,TraceOb->pitch_Pixels,
								Facet->NumXpnts*sizeof(float2),Facet->NumYpnts,cudaMemcpyDeviceToDevice);
	dev_FacetPressure.filterMode=cudaFilterModePoint;
	dev_FacetPressure.addressMode[0]=cudaAddressModeClamp;
	dev_FacetPressure.addressMode[1]=cudaAddressModeClamp;
    dev_FacetPressure.normalized = false;    // access with integer texture coordinates
	cudaBindTextureToArray(dev_FacetPressure,d_PressArray);
	//printf("Making Facet Pressure Texture, Error: %s\n", cudaGetErrorString(cudaGetLastError()));

	
	cudaStatus = cudaMalloc((void**)&dev_PressureOut, NumFeildPoints*sizeof(dcomplex));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return 1;
	}

    dim3 cudaBlockSize;
	cudaBlockSize.x = (Facet->NumXpnts+POINTS_PER_THREAD-1)/POINTS_PER_THREAD;
	cudaBlockSize.y = (Facet->NumYpnts+POINTS_PER_THREAD-1)/POINTS_PER_THREAD;
	cudaBlockSize.z = 1;

	//printf("Threads Allocated Per Block, x: %d, y: %d, z: %d\n",cudaBlockSize.x,cudaBlockSize.y,cudaBlockSize.z);

	if(cudaBlockSize.x*cudaBlockSize.y >= MAX_THREADS_PER_BLOCK)
	{
		printf("Not Enough Threads Per Block, Consider Increasing Positions Per Thread or Reducing Maxium Facet Size. Number of Threads :%d",cudaBlockSize.x*cudaBlockSize.y);
		return 1;
	}

	if(cudaBlockSize.x > THREADS_ONEDIM)
	{
		printf("Too Many Threads Allocated Per Block X-Axis, Consider Increasing Positions Per Thread.");
		return 1;
	}


	if(cudaBlockSize.y > THREADS_ONEDIM)
	{
		printf("Too Many Threads Allocated Per Block Y-Axis, Consider Increasing Positions Per Thread.");
		return 1;
	}


    dim3 cudaGridSize;
	cudaGridSize.x = NumFeildPoints;
	cudaGridSize.y = 1;
	cudaGridSize.z = 1;

	ProjectToCentriod = new float4[NumFeildPoints];
	for(unsigned int i = 0; NumFeildPoints > i; i++)
	{
		DestPoint = DestPoints[i];
		float3 vCp = GeoMath::MakeVectorSc(Facet->Centriod,DestPoint->PointLocation);
		ProjectToCentriod[i].x = GeoMath::dotProductSc(Facet->xAxis,vCp);
		ProjectToCentriod[i].y = GeoMath::dotProductSc(Facet->yAxis,vCp);
		ProjectToCentriod[i].z = GeoMath::dotProductSc(Facet->Normal,vCp);

		float Rc_jk = GeoMath::GetVectorLength(vCp);
		float CosScat = ProjectToCentriod[i].z / Rc_jk;

		if(TraceOb->CosInc >= 0 && CosScat >= 0)
			ProjectToCentriod[i].w = (abs(TraceOb->CosInc)+abs(CosScat))/(4*PI*Rc_jk);

		if(TraceOb->CosInc < 0 && CosScat < 0)
			ProjectToCentriod[i].w = (abs(TraceOb->CosInc)+abs(CosScat))/(4*PI*Rc_jk);

		if(TraceOb->CosInc > 0 && CosScat < 0)
		{
			// printf("LP Not Implemented Yet.\n");
			ProjectToCentriod[i].w = 0;
		}

		if(TraceOb->CosInc < 0 && CosScat > 0)
		{
			// printf("LP Not Implemented Yet.\n");
			ProjectToCentriod[i].w = 0;
		}
		//if(isnan(ProjectToCentriod[i].w))
		//{
		//	printf("Facet 2 Point Error, a: %e\n",ProjectToCentriod[i].w);
		//printf("Facet 2 Point Error, TraceOb->CosInc: %e\n",TraceOb->CosInc);
		//printf("Facet 2 Point Error, CosScat: %e\n",CosScat);
		// printf("Facet 2 Point Error, Rc_jk: %e\n",Rc_jk);
		//}
	}

	cudaArray* d_ProjArray; 
	cudaChannelFormatDesc d_Projchannel;
	d_Projchannel=cudaCreateChannelDesc<float4>();  
	cudaMallocArray(&d_ProjArray,&d_Projchannel,(size_t)NumFeildPoints); 
	cudaMemcpyToArray(d_ProjArray,0,0,ProjectToCentriod,NumFeildPoints*sizeof(float4),cudaMemcpyHostToDevice); 
	dev_Projection.filterMode=cudaFilterModePoint;
	dev_Projection.addressMode[0]=cudaAddressModeClamp;
    dev_Projection.normalized = false;    // access with integer texture coordinated
	cudaBindTextureToArray(dev_Projection,d_ProjArray);


	unsigned int SharedMemorySize = MAX_THREADS_PER_BLOCK*sizeof(dcomplex);
	if( SHARE_MEMSIZE < SharedMemorySize )
		printf("Total Shared Memory Size Per Block Exceeds Maxium Specification");
	ScanProjectFacetToPointGPU<<<cudaGridSize, cudaBlockSize, SharedMemorySize>>>(dev_PressureOut, Facet->FacetNum);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Facet to Point launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}
    
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching ScanProjectFacetToPointGPU!\n", cudaStatus);
		return 1;
	}


	// Copy output vector from GPU buffer to host memory.
	dcomplex* tmp_Pressure = new dcomplex[NumFeildPoints];
	cudaStatus = cudaMemcpy(tmp_Pressure, dev_PressureOut, NumFeildPoints*sizeof(dcomplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("Copying Feild Point Pressures from Device failed!");
		return 1;
	}

	for(unsigned int i = 0; NumFeildPoints > i; i++)
	{
		DestPoint = DestPoints[i];
		DestPoint->Pressure.r += tmp_Pressure[i].r;
		DestPoint->Pressure.i += tmp_Pressure[i].i;
		//printf("Facet 2 Feild Point Pressure, Real :%e, Imag :%e\n",DestPoint->Pressure.r,DestPoint->Pressure.i);
	}

	cudaUnbindTexture(dev_Projection);
	cudaUnbindTexture(dev_FacetPressure);
	cudaFreeArray(d_ProjArray); 
	cudaFreeArray(d_PressArray);
	cudaFree(dev_PressureOut);
	delete ProjectToCentriod;
	delete tmp_Pressure;
    return 0;
}
							
__global__ void ProjectPointToFacetGPU(dcomplex PressInc,dcomplex* dev_Pressure,unsigned int FacetNum, float3 ProjectToCentriod,unsigned int pitch_Pixels)
{
	unsigned int Start_i;
	unsigned int Start_j;
	unsigned int Indx;
	unsigned int Indy;
	unsigned int i;
	unsigned int j;
	float R;
	float a; 
	float4 FacetPoint;
	float3 vpS;
	dcomplex Phase;
	float2* Pixel_row;

	Start_i = threadIdx.x*POINTS_PER_THREAD;
	Start_j = threadIdx.y*POINTS_PER_THREAD;

	//printf("************ FacetNum :%d\n", FacetNum); 
	//printf("threadIdx.x :%d, threadIdx.y :%d, threadIdx.z :%d\n",threadIdx.x,threadIdx.y,threadIdx.z);
	//printf("Facet Limits Start_i:%d, Start_j:%d\n",dev_Facet_MaxIndx[0],dev_Facet_MaxIndy[0]);
	//printf("Start_i:%d, Start_j:%d\n",Start_i,Start_j);
	//return;

	#pragma unroll POINTS_PER_THREAD
    for(i = 0; i < POINTS_PER_THREAD; i++)
    {
		Indx = Start_i+i;
		if( Indx < dev_Facet_MaxIndx[0] )
		{
			#pragma unroll POINTS_PER_THREAD
			for (j = 0; j < POINTS_PER_THREAD; j++)
			{
				Indy = Start_j+j;
				if( Indy < dev_Facet_MaxIndy[0] )
				{
					
					FacetPoint = tex3D(dev_Positions,FacetNum,Indx,Indy);
					//printf("Ind Max :%d, :%d, Ind Val :%d, :%d, Point Val:%f, y :%f, z :%f, w :%f\n",dev_Facet_MaxIndx[0],dev_Facet_MaxIndy[0],Indx, Indy, FacetPoint.x,FacetPoint.y,FacetPoint.z,FacetPoint.w);
						
					if( FacetPoint.w > 0 )
					{
					//
					//printf("Point x :%f, y :%f, z :%f, w :%f\n",FacetPoint.x,FacetPoint.y,FacetPoint.z,FacetPoint.w);

						
						vpS.x	=  FacetPoint.x-ProjectToCentriod.x;
						vpS.y	=  FacetPoint.y-ProjectToCentriod.y;
						vpS.z	=  FacetPoint.z-ProjectToCentriod.z;
						R = sqrtf(vpS.x*vpS.x+vpS.y*vpS.y+vpS.z*vpS.z);


						// NB: Formula is exp(1i*k_wave*R)	
						a = FacetPoint.w/(2*PI*R);

						//printf("Radius %e\n",R);
						
						Phase.r = cosf(R*dev_k_wave[0].r)*expf(-R*dev_k_wave[0].i)*a;
						Phase.i = sinf(R*dev_k_wave[0].r)*expf(-R*dev_k_wave[0].i)*a;


						//*c.r=a.r*b.r-a.i*b.i;
						//*c.i=a.i*b.r+a.r*b.i;
						Pixel_row = (float2*)(((char*)dev_Pressure) + Indy * pitch_Pixels);
						Pixel_row[Indx].x = Phase.r*PressInc.r-Phase.i*PressInc.i;
						Pixel_row[Indx].y = Phase.i*PressInc.r+Phase.r*PressInc.i;
	

						//printf("Point x :%f, y :%f, z :%f, w :%f\n",FacetPoint.x,FacetPoint.y,FacetPoint.z,FacetPoint.w);
						//printf("Indx:%d, Indy:%d\n",Indx,Indy);

					}
					else
					{
						Pixel_row = (float2*)((char*)dev_Pressure + Indy * pitch_Pixels);
						Pixel_row[Indx].x = 0;
						Pixel_row[Indx].y = 0;
					}
					//printf("R Radius :%e\n",R);
					//printf("Phase Pressure :%e, Imag :%e\n",Phase.r, Phase.i);
					//printf("Incendent Pressure :%e, Imag :%e\n",PressInc.r, PressInc.i);
					//printf("K Wave Real :%e, Imag :%e\n",dev_k_wave[0].r, dev_k_wave[0].i);
					//printf("Pressure Real :%e, Imag :%e\n",Pixel_row[Indx].x , Pixel_row[Indx].y);
				}
			}
		}
	}
}



int MakeFacetsOnGPU( FacetData** Facets, unsigned int NumFacets, dcomplex k_wave)
{


	FacetData* CurFacet;

	unsigned int MaxNumPosX = 0;
	unsigned int MaxNumPosY = 0;


	for (unsigned int i = 0; NumFacets > i; i++)
	{
		CurFacet = Facets[i];
		if( MaxNumPosX < CurFacet->NumXpnts )
			MaxNumPosX = CurFacet->NumXpnts;

		if( MaxNumPosY < CurFacet->NumYpnts )
			MaxNumPosY = CurFacet->NumYpnts;
	}

	if(NumFacets > MAX_TEXTURE_DIM)
	{
		printf("Error, Maximum Number of Facets Exceeded\n");
		return 1;
	}

	if(MaxNumPosX > MAX_TEXTURE_DIM)
	{
		printf("Error, Maximum Number of Points X-Axis Exceeded\n");
		return 1;
	}

	if(MaxNumPosY > MAX_TEXTURE_DIM)
	{
		printf("Error, Maximum Number of Points Y-Axis Exceeded\n");
		return 1;
	}

	cudaMemcpyToSymbol(dev_k_wave,&k_wave,1*sizeof(dcomplex));
	printf("Making Symbol dev_k_wave: %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpyToSymbol(dev_Facet_MaxIndx,&MaxNumPosX,1*sizeof(unsigned int));
	printf("Making Symbol dev_Facet_MaxIndx, Value %d, Error: %s\n", MaxNumPosX, cudaGetErrorString(cudaGetLastError()));

	cudaMemcpyToSymbol(dev_Facet_MaxIndy,&MaxNumPosY,1*sizeof(unsigned int));
	printf("Making Symbol dev_Facet_MaxIndy, Value %d, Error: %s\n", MaxNumPosY, cudaGetErrorString(cudaGetLastError()));

	cudaExtent extentPos = make_cudaExtent(NumFacets, MaxNumPosX, MaxNumPosY);
	//unsigned int pitch = ((unsigned int)extentPos.width*sizeof(float4)+TEXTURE_ALIGNMENT-1)/TEXTURE_ALIGNMENT;
	unsigned int pitch = (unsigned int)extentPos.width*sizeof(float4);
    unsigned int slicePitch = pitch * (unsigned int)extentPos.height;
	float4* h_Pos = (float4*)malloc(pitch*extentPos.height*extentPos.depth);

	for (unsigned int z = 0; z < extentPos.depth; ++z) {
        char* slice = (char*)h_Pos + z * slicePitch;
        for (unsigned int y = 0; y < extentPos.height; ++y) {
            float4* row = (float4*)(slice + y * pitch);
            for (unsigned int x = 0; x < extentPos.width; ++x) {
				CurFacet = Facets[x];
                row[x].x = CurFacet->PositionVector[z*MaxNumPosX+y].x;
				row[x].y = CurFacet->PositionVector[z*MaxNumPosX+y].y;
				row[x].z = CurFacet->PositionVector[z*MaxNumPosX+y].z;
				row[x].w = CurFacet->PositionVector[z*MaxNumPosX+y].w;				
            }
        }
    }




	cudaArray* tmpArray; 
	cudaChannelFormatDesc channel;
	channel=cudaCreateChannelDesc<float4>();  
		
	cudaPitchedPtr srcPtr = make_cudaPitchedPtr((void*)&h_Pos[0], pitch, extentPos.width, extentPos.height);
	//printf("Making Pitched Pointer: %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMalloc3DArray(&tmpArray,&channel,extentPos);
	printf("cudaMalloc3DArray Position Texture: %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy3DParms params = {0};
	params.srcPtr = srcPtr;
	params.dstArray = tmpArray;
	params.extent = extentPos;
	params.kind = cudaMemcpyHostToDevice;

	cudaMemcpy3D(&params);
	printf("cudaMemcpy3D Position Texture: %s\n", cudaGetErrorString(cudaGetLastError()));

	dev_Positions.filterMode=cudaFilterModePoint;
	dev_Positions.addressMode[0]=cudaAddressModeClamp;
	dev_Positions.addressMode[1]=cudaAddressModeClamp;
	dev_Positions.addressMode[2]=cudaAddressModeClamp;
    dev_Positions.normalized = false;    // access with integer texture coordinates
	cudaBindTextureToArray(dev_Positions,tmpArray);

	printf("Position Texture: %s\n", cudaGetErrorString(cudaGetLastError()));

	//delete h_Pos;
	//free(h_Pos);
	//cudaFreeArray(tmpArray);

	return 0;
}




int ProjectPointToFacet( PointData* ScrPoint, FacetData* Facet, TraceObject* TraceOb)
{
	float3 ProjectToCentriod;
	cudaError_t cudaStatus;

	float3 vCp = GeoMath::MakeVectorSc(Facet->Centriod,ScrPoint->PointLocation);
	ProjectToCentriod.x = GeoMath::dotProductSc(Facet->xAxis,vCp);
	ProjectToCentriod.y = GeoMath::dotProductSc(Facet->yAxis,vCp);
	ProjectToCentriod.z = GeoMath::dotProductSc(Facet->Normal,vCp);

	//printf("Facet :%d, Normal x :%e, y:%e, z:%e\n", Facet->FacetNum, Facet->Normal.x, Facet->Normal.y, Facet->Normal.z);

    dim3 cudaBlockSize;
	cudaBlockSize.x = (unsigned int)((Facet->NumXpnts+POINTS_PER_THREAD-1)/POINTS_PER_THREAD);
	cudaBlockSize.y = (unsigned int)((Facet->NumYpnts+POINTS_PER_THREAD-1)/POINTS_PER_THREAD);
	cudaBlockSize.z = 1;

	//printf("Block Size Information. *******************\n");
	//printf("Number of points x-axis %d\n",Facet->NumXpnts);
	//printf("Number of points y-axis %d\n",Facet->NumYpnts);
	//printf("cudaBlockSize.x %d\n",cudaBlockSize.x);
	//printf("cudaBlockSize.y %d\n",cudaBlockSize.y);
	//printf("Threads per block %d, Max value %d\n",cudaBlockSize.x*cudaBlockSize.y,MAX_THREADS_PER_BLOCK);

	if(cudaBlockSize.x*cudaBlockSize.y >= MAX_THREADS_PER_BLOCK)
	{
		printf("Memory Allocation Error, not enough shared memory, Consider Increasing Positions Per Thread.\n");
		return 1;
	}

	if(cudaBlockSize.x > THREADS_ONEDIM)
	{
		printf("Too Many Threads Allocated Per Block X-Axis, Consider Increasing Positions Per Thread.\n");
		return 1;
	}


	if(cudaBlockSize.y > THREADS_ONEDIM)
	{
		printf("Too Many Threads Allocated Per Block Y-Axis, Consider Increasing Positions Per Thread.\n");
		return 1;
	}

    dim3 cudaGridSize;
	cudaGridSize.x = 1;
	cudaGridSize.y = 1;
	cudaGridSize.z = 1;

	float3 Vcs = GeoMath::MakeVectorSc(Facet->Centriod,ScrPoint->PointLocation);
	float Dist_R = GeoMath::GetVectorLength(Vcs);

	TraceOb->CosInc = ProjectToCentriod.z/Dist_R;
	TraceOb->FacetNum_i = 0;
	TraceOb->FacetNum_j = Facet->FacetNum;

	//printf("Facet :%d, Z Projection :%e, Radius :%e, CosInc :%e\n", Facet->FacetNum, ProjectToCentriod.z, Dist_R, TraceOb->CosInc);

	cudaStatus = cudaMallocPitch((void **)&TraceOb->dev_LastPixelPressure, (size_t*)(&(TraceOb->pitch_Pixels)),
									(size_t)(Facet->NumXpnts * sizeof(float2)), (size_t)(Facet->NumYpnts));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!\n");
        return 1;
    }
	
    ProjectPointToFacetGPU<<<cudaGridSize, cudaBlockSize>>>(ScrPoint->Pressure,TraceOb->dev_LastPixelPressure,
											Facet->FacetNum, ProjectToCentriod, TraceOb->pitch_Pixels);



      // Check for any errors launching the kernel
     cudaStatus = cudaGetLastError();
	 if (cudaStatus != cudaSuccess) {
         printf("ProjectPointToFacetGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
         return 1;
	 }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
     if (cudaStatus != cudaSuccess) {
        printf("ProjectPointToFacetGPU Synchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return 1;
    }
	//TraceOb->h_LastPixelPressure = (dcomplex*)malloc(sizeof(dcomplex)*TraceOb->pitch_Pixels*Facet->NumYpnts);
	//cudaMemcpy2D(TraceOb->h_LastPixelPressure,TraceOb->pitch_Pixels,TraceOb->dev_LastPixelPressure,
	//			TraceOb->pitch_Pixels,Facet->NumXpnts*sizeof(dcomplex),Facet->NumYpnts,cudaMemcpyDeviceToHost);
	//printf("Making Pixel Pressure Pointer Copy, Error: %s\n", cudaGetErrorString(cudaGetLastError()));

	//TraceOb->h_CosInc = (float*)malloc(sizeof(float)*TraceOb->pitch_Angles*Facet->NumYpnts);
	//cudaMemcpy2D(TraceOb->h_CosInc,TraceOb->pitch_Angles,TraceOb->dev_CosInc,
	//			TraceOb->pitch_Angles,Facet->NumXpnts*sizeof(float),Facet->NumYpnts,cudaMemcpyDeviceToHost);
	//printf("Making Angle Array Pointer, Error: %s\n", cudaGetErrorString(cudaGetLastError()));

    return 0;
}

TraceObject* ScanProjectFacetToFacet(FacetData* Facet_i,FacetData* Facet_j, TraceObject* TraceObj_i)
{
	cudaError_t cudaStatus;

	TraceObject* TraceObj_j = MakeTraceObject();
	cudaStatus = cudaMallocPitch((void **)&TraceObj_j->dev_LastPixelPressure, (size_t*)(&(TraceObj_j->pitch_Pixels)),
									(size_t)(Facet_j->NumXpnts * sizeof(float2)), (size_t)(Facet_j->NumYpnts));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!\n");
        return TraceObj_j;
    }

	cudaArray* d_PressArray; 
	cudaChannelFormatDesc d_Presschannel;
	d_Presschannel=cudaCreateChannelDesc<float2>();  
	cudaMallocArray(&d_PressArray,&d_Presschannel,Facet_i->NumXpnts,Facet_i->NumYpnts); 
	cudaMemcpy2DToArray(d_PressArray,0,0,TraceObj_i->dev_LastPixelPressure,TraceObj_i->pitch_Pixels,
								Facet_i->NumXpnts*sizeof(float2),Facet_i->NumYpnts,cudaMemcpyDeviceToDevice);
	dev_FacetPressure.filterMode=cudaFilterModePoint;
	dev_FacetPressure.addressMode[0]=cudaAddressModeClamp;
	dev_FacetPressure.addressMode[1]=cudaAddressModeClamp;
    dev_FacetPressure.normalized = false;    // access with integer texture coordinates
	cudaBindTextureToArray(dev_FacetPressure,d_PressArray);
	//printf("Making Facet Pressure Texture, Error: %s\n", cudaGetErrorString(cudaGetLastError()));


	dim3 cudaBlockSize;
	cudaBlockSize.x = (Facet_i->NumXpnts+POINTS_PER_THREAD-1)/POINTS_PER_THREAD;
	cudaBlockSize.y = (Facet_i->NumYpnts+POINTS_PER_THREAD-1)/POINTS_PER_THREAD;
	cudaBlockSize.z = 1;

	if(cudaBlockSize.x*cudaBlockSize.y >= MAX_THREADS_PER_BLOCK)
	{
		printf("Memory Allocation Error, not enough shared memory, Consider Increasing Positions Per Thread.\n");
		return TraceObj_j;
	}

	if(cudaBlockSize.x > THREADS_ONEDIM)
	{
		printf("Too Many Threads Allocated Per Block X-Axis, Consider Increasing Positions Per Thread.\n");
		return TraceObj_j;
	}


	if(cudaBlockSize.y > THREADS_ONEDIM)
	{
		printf("Too Many Threads Allocated Per Block Y-Axis, Consider Increasing Positions Per Thread.\n");
		return TraceObj_j;
	}

    dim3 cudaGridSize;
	cudaGridSize.x = Facet_j->NumXpnts;
	cudaGridSize.y = Facet_j->NumYpnts;
	cudaGridSize.z = 1;

	float3 Vjk = GeoMath::MakeVectorSc(Facet_i->Centriod,Facet_j->Centriod);
	float R_cjk = GeoMath::GetVectorLength(Vjk);
	if(R_cjk <= 0)
		printf("************Error Facet 2 Facet, facets overlap or i=j, i :%d, j :%d\n", Facet_i->FacetNum,Facet_j->FacetNum);

	float Z_cjk = GeoMath::dotProductSc(Vjk,Facet_i->Normal);
	float CosScat = Z_cjk/R_cjk;

	float a_Factor = 0;
	if( TraceObj_i->CosInc >= 0 && CosScat >= 0)
		a_Factor = (TraceObj_i->CosInc+CosScat)/(4*PI*R_cjk);
	
	if( TraceObj_i->CosInc < 0 && CosScat < 0)
		a_Factor = (TraceObj_i->CosInc+CosScat)/(4*PI*R_cjk);

	//printf("*************************************************************************\n");
	printf("Facet :%d to Facet :%d, R %e, CosInc %e, CosScat %e\n",Facet_i->FacetNum,Facet_j->FacetNum, R_cjk,TraceObj_i->CosInc*180/PI,CosScat*180/PI);
	printf("Facet :%d to Facet :%d, Z %e, a_Factor %e\n",Facet_i->FacetNum,Facet_j->FacetNum,Z_cjk, a_Factor);

	// Relative co-ordinates from a point on j to the centeriod of i.
	float4* ProjectToCentriod_i = new float4[Facet_j->NumXpnts*Facet_j->NumYpnts];
	for(unsigned int i = 0; Facet_j->NumXpnts > i; i++)
	{
		for(unsigned int j = 0; Facet_j->NumYpnts > j; j++)
		{
			// Local Point on Facet j
			float4 PointOnFacet_j = Facet_j->PositionVector[j*Facet_i->NumXpnts+i];
			float3 GPointOnFacet_j;

			// Global Point on Facet j
			GPointOnFacet_j.x = Facet_j->Centriod.x+Facet_j->xAxis.x*PointOnFacet_j.x+Facet_j->yAxis.x*PointOnFacet_j.y+Facet_j->Normal.x*PointOnFacet_j.z;
			GPointOnFacet_j.y = Facet_j->Centriod.y+Facet_j->xAxis.y*PointOnFacet_j.x+Facet_j->yAxis.y*PointOnFacet_j.y+Facet_j->Normal.y*PointOnFacet_j.z;
			GPointOnFacet_j.z = Facet_j->Centriod.z+Facet_j->xAxis.z*PointOnFacet_j.x+Facet_j->yAxis.z*PointOnFacet_j.y+Facet_j->Normal.z*PointOnFacet_j.z;

			// Relative local co-ordinates of the centriod of facet i to the point on Facet j
			float3 vCp = GeoMath::MakeVectorSc(Facet_i->Centriod,GPointOnFacet_j);
			ProjectToCentriod_i[j*Facet_j->NumXpnts+i].x = GeoMath::dotProductSc(Facet_i->xAxis,vCp);
			ProjectToCentriod_i[j*Facet_j->NumXpnts+i].y = GeoMath::dotProductSc(Facet_i->yAxis,vCp);
			ProjectToCentriod_i[j*Facet_j->NumXpnts+i].z = GeoMath::dotProductSc(Facet_i->Normal,vCp);

			// Note there is approximation here that may be improved in future versions.
			ProjectToCentriod_i[j*Facet_j->NumXpnts+i].w = a_Factor;

		}
	}

	cudaArray* d_ProjectionArray_i; 
	cudaChannelFormatDesc d_Projectionchannel_i;
	d_Projectionchannel_i=cudaCreateChannelDesc<float4>();
	cudaMallocArray(&d_ProjectionArray_i,&d_Projectionchannel_i,Facet_i->NumXpnts,Facet_i->NumYpnts); 
	cudaMemcpy2DToArray(d_ProjectionArray_i,0,0,ProjectToCentriod_i,Facet_i->NumXpnts*sizeof(float4),
									Facet_i->NumXpnts*sizeof(float4),Facet_i->NumYpnts,cudaMemcpyHostToDevice);
	//printf("Making Projection_i Array Copy, Error: %s\n", cudaGetErrorString(cudaGetLastError()));
	dev_Projection_i.filterMode=cudaFilterModePoint;
	dev_Projection_i.addressMode[0]=cudaAddressModeClamp;
	dev_Projection_i.addressMode[1]=cudaAddressModeClamp;
    dev_Projection_i.normalized = false;    // access with integer texture coordinated
	cudaBindTextureToArray(dev_Projection_i,d_ProjectionArray_i);


	unsigned int SharedMemorySize = MAX_THREADS_PER_BLOCK*sizeof(dcomplex);
	ScanProjectFacetToFacetGPU<<<cudaGridSize, cudaBlockSize,SharedMemorySize>>>(TraceObj_j->dev_LastPixelPressure,
													Facet_i->FacetNum, Facet_j->FacetNum, TraceObj_j->pitch_Pixels);

	TraceObj_j->FacetNum_i = Facet_i->FacetNum;
	TraceObj_j->FacetNum_j = Facet_j->FacetNum;

	float3 Vji = GeoMath::MakeVectorSc(Facet_j->Centriod,Facet_i->Centriod);
	float R_cji = GeoMath::GetVectorLength(Vji);
	float Z_cji = GeoMath::dotProductSc(Vji,Facet_j->Normal);
	TraceObj_j->CosInc = Z_cji/R_cji;

	if(R_cji <= 0)
		printf("************Error Facet 2 Facet, facets overlap or j=k, i :%d, j :%d\n", Facet_i->FacetNum,Facet_j->FacetNum);


     // Check for any errors launching the kernel
     cudaStatus = cudaGetLastError();
	 if (cudaStatus != cudaSuccess) {
         printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
         return TraceObj_j;
	 }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
     if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return TraceObj_j;
    }

	cudaUnbindTexture(dev_FacetPressure);
	cudaUnbindTexture(dev_Projection_i);
	cudaUnbindTexture(dev_Projection_j);
	cudaFreeArray(d_ProjectionArray_i);
	cudaFreeArray(d_PressArray); 
	delete ProjectToCentriod_i;
	return TraceObj_j;
}



int StartCuda()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		return 1;
    }
	return 0;
}

int StopCuda()
{
	cudaError_t cudaStatus = cudaUnbindTexture(dev_Positions);
    if (cudaStatus != cudaSuccess) {
        printf("Unbinding of Positions Texture Failed!\n");
		return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        return 1;
    }
	return 0;
}




int DeleteTraceObject(TraceObject* TraceObject)
{
	cudaFree(TraceObject->dev_LastPixelPressure);
	delete TraceObject;
	return 0;
}

void PrintComplexVector( dcomplex* dev_Vector, unsigned int NumPoints)
{
	dcomplex* h_Vector = new dcomplex[NumPoints];

	cudaError_t cudaStatus = cudaMemcpy(h_Vector, dev_Vector, NumPoints*sizeof(dcomplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!\n");
		return;
	}

	for(unsigned int i = 0; NumPoints > i; i++)
	{
		std::cout << "Value Real : " << h_Vector[i].r << "\n";
		std::cout << "Value Imag : " << h_Vector[i].i << "\n";
	}
	delete h_Vector;
}


void PrintVector( float* dev_Vector, unsigned int NumPoints)
{
	float* h_Vector = new float[NumPoints];

	cudaError_t cudaStatus = cudaMemcpy(h_Vector, dev_Vector, NumPoints*sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!\n");
		return;
	}

	for(unsigned int i = 0; NumPoints > i; i++)
	{
		std::cout << "Value : " << h_Vector[i] << "\n";
	}
	delete h_Vector;
}

#endif