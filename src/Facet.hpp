#ifndef _FacetData
#define _FacetData

#include "FacetCuda.hpp"
#include "Globals.h"

#include <iostream>
#include "GeoMath.h"
#include "dcomplex.h"

class Facet : public FacetCuda
{

public:
	float3 v1;
	float3 v2;
	float3 v3;
	float texCoords[6]; // Texture coordinates for the vertices
	float3 Centroid;

private:
	float3 Normal;
	float3 xAxis;
	float3 yAxis;
	float3 pointOnBase;

	int Material;

	float BaseLength;
	float Height;
	float BaseLengthNeg;
	float BaseLengthPos;
	float Area;

	float *fragArea = 0;

	float *vY = 0;
	float *vX = 0;
	float delta;

public:
	Facet(float3 t_v1, float3 t_v2, float3 t_v3, ObjectType type, float t_resolution);
	~Facet();

public:
	void PrintAreaMatrix();
	void MakeCuda();

	void CompressPixels();
	void MakeFragmentData();

private:
	void GenerateFacetLimits();
	void CalculateCentroid();
};
#endif