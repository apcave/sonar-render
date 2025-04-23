#ifndef _FacetData
#define _FacetData

#include "FacetCuda.hpp"

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

private:
	float3 Centroid;
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

	float *fragArea = nullptr;

	float *vY = nullptr;
	float *vX = nullptr;
	float delta;

public:
	Facet(float3 t_v1, float3 t_v2, float3 t_v3);
	~Facet();

public:
	void PrintMatrix();
	void MakeCuda();

	void CompressPixels();
	void MakeFragmentData(float frag_length);

private:
	void GenerateFacetLimits();
	void CalculateCentroid();
};
#endif