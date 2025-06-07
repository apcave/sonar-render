#ifndef MODEL_SHARED_H
#define MODEL_SHARED_H
#include "Globals.h"
#include "dcomplex.h"
#include <optix.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

/**
 * @brief OptiX parameters for rendering kernel.
 *
 * This is a global structure that contains all the CUDA for a run in an organized way.
 */

enum CalcType
{
    SOURCE_POINTS = 0,
    FIELD_POINTS,
    FACET_SELF,
    FACET_RESP,
    FACET_NO_RESP,
};

typedef struct
{
    int numPnts;
    float3 *position;
    dcomplex *pressure;
} dev_points;

typedef struct
{
    float3 normal;
    float3 base_point;
    float3 xAxis;
    float3 yAxis;
    int3 frag_points;
    float *frag_area;
    dcomplex *P; // These buffers are swapped for the application.
    dcomplex *P_in;
    dcomplex *P_out;

} dev_facet;

typedef struct
{
    ObjectType objectType;
    int numFacets;
    dev_facet *facets;
} dev_object;

typedef struct
{
    OptixTraversableHandle handle; // Acceleration structure handle

    dcomplex k_wave;  // Wave number
    float frag_delta; // Fragment length

    dev_object srcObject;
    dev_object dstObject;

    dev_points srcPoints;
    dev_points dstPoints;

    CalcType calcType;

} globalParams;

#endif