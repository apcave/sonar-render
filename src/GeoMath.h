#ifndef _GEOMATH
#define _GEOMATH
#include <math.h>

#define PI 3.1415926535897932384626433832795f

#include <cuda_runtime.h>

class GeoMath
{
public:
	static float LengthBetweenPoints(float3 P1, float3 P2)
	{
		return sqrtf((P1.x - P2.x) * (P1.x - P2.x) + (P1.y - P2.y) * (P1.y - P2.y) + (P1.z - P2.z) * (P1.z - P2.z));
	}

	static unsigned int GetMaxIndex(float *Vals, unsigned int NumVals)
	{
		float MaxVal = Vals[0];
		unsigned int MaxInd = 0;

		for (unsigned int i = 1; NumVals > i; i++)
		{
			if (Vals[i] > MaxVal)
			{
				MaxVal = Vals[i];
				MaxInd = i;
			}
		}
		return MaxInd;
	}

	static float3 MakeVectorSc(float3 Origin, float3 Dest)
	{
		float3 Vc;
		Vc.x = Dest.x - Origin.x;
		Vc.y = Dest.y - Origin.y;
		Vc.z = Dest.z - Origin.z;
		return Vc;
	}

	static float3 CrossProductSc(float3 v1, float3 v2)
	{
		float3 crossProduct;
		crossProduct.x = v1.y * v2.z - v1.z * v2.y;
		crossProduct.y = v1.z * v2.x - v1.x * v2.z;
		crossProduct.z = v1.x * v2.y - v1.y * v2.x;
		return crossProduct;
	}

	static float3 MakeVectorThroughPointPerpToVector(float3 vectPnt, float3 vect, float3 perpPnt, float3 &pntOnLine)
	{
		float3 perpVect;
		float3 vectPointPerp = MakeVectorSc(vectPnt, perpPnt);
		float3 unitVect = MakeUnitVectorSc(vect);
		float projLength = dotProductSc(unitVect, vectPointPerp);

		pntOnLine.x = vectPnt.x + unitVect.x * projLength;
		pntOnLine.y = vectPnt.y + unitVect.y * projLength;
		pntOnLine.z = vectPnt.z + unitVect.z * projLength;
		perpVect = MakeVectorSc(pntOnLine, perpPnt);
		return perpVect;
	}

	static float3 MakeUnitNormSc(float3 uV, float3 refPoint, float3 PerpPoint)
	{
		float3 tmpVrefPerp = MakeVectorSc(refPoint, PerpPoint);
		float dist_uV = dotProductSc(uV, tmpVrefPerp);
		float3 uV_IntPoint;
		uV_IntPoint.x = refPoint.x + uV.x * dist_uV;
		uV_IntPoint.y = refPoint.y + uV.y * dist_uV;
		uV_IntPoint.z = refPoint.z + uV.z * dist_uV;
		return MakeUnitVectorSc(uV_IntPoint, PerpPoint);
	}

	static float3 MakeUnitVectorSc(float3 Origin, float3 Dest)
	{
		return MakeUnitVectorSc(MakeVectorSc(Origin, Dest));
	}

	static float3 MakeUnitVectorSc(float3 vect)
	{
		float Lenght = sqrtf(vect.x * vect.x + vect.y * vect.y + vect.z * vect.z);
		float3 unitVect;
		unitVect.x = vect.x / Lenght;
		unitVect.y = vect.y / Lenght;
		unitVect.z = vect.z / Lenght;
		return unitVect;
	}

	static float dotProductSc(float3 v1, float3 v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	static float GetVectorLength(float3 v1)
	{
		return sqrtf(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
	}
};
#endif