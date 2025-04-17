#ifndef _FacetData
#define _FacetData

#define ANSI
#include <iostream>
#include "GeoMath.h"
#include "dcomplex.h"

class Facet
{
public:
	float3 Centroid;
	float3 Normal;
	float3 xAxis;
	float3 yAxis;
	float3 pointOnBase;

	float3 v1;
	float3 v2;
	float3 v3;

	int Material;

	float BaseLength;
	float Height;
	float BaseLengthNeg;
	float BaseLengthPos;
	float Area;

	float *PixelArea;
	int NumXpnts;
	int NumYpnts;
	int NumXpntsNegative;
	float *vY;
	float *vX;
	float delta;

	dcomplex *PressureValues;

public:
	Facet(float3 t_v1, float3 t_v2, float3 t_v3);
	~Facet()
	{
		delete[] PixelArea;
		delete[] vY;
		delete[] vX;
	};

public:
	void PrintMatrix();

	void CompressPixels();
	void MakePixelData(float pixel_length);

private:
	void GenerateFacetLimits();
	void CalculateCentroid();
};
#endif