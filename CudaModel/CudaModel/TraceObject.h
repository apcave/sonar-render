#ifndef _TraceObject
#define _TraceObject

#include "GeoMath.h"
#include <stdio.h>
#include <math.h>
#include <vector>

struct TraceObject
{
	unsigned int FacetNum_i;
	unsigned int FacetNum_j;


	unsigned int pitch_Pixels;
	dcomplex* dev_LastPixelPressure;
	dcomplex* h_LastPixelPressure;

	float CosInc;
};
#endif