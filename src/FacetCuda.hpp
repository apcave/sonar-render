#ifndef _FACET_CUDA
#define _FACET_CUDA

#include "FacetGl.hpp"

#include <cuda_runtime.h>

struct dev_facet
{
    float3 normal;
    float3 base_point;
    float3 xAxis;
    float3 yAxis;
    int3 frag_points;
};

class FacetCuda : public FacetGl
{
public:
    FacetCuda();
    void AllocateCuda(float3 &normal,
                      float3 &base_point,
                      float3 &xAxis,
                      float3 &yAxis,
                      float *frag_area);

    void WriteSurface(float *dev_frag_stats);
    void GetSurfaceScalers(float *dev_frag_stats);

private:
    // Area of the fragment.
    float *dev_frag_area;
    dev_facet *dev_data;
    int3 frag_points;

private:
    // Presure on the surface.
    double *dev_Pr;
    double *dev_Pi;

    // Working buffers for facet to facet calculations.
    double *dev_Pr_initial;
    double *dev_Pi_initial;

    double *dev_Pr_result;
    double *dev_Pi_result;

    friend class ModelCuda;
};
#endif