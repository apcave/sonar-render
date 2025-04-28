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
    ~FacetCuda();
    void AllocateCuda(float3 &normal,
                      float3 &base_point,
                      float3 &xAxis,
                      float3 &yAxis,
                      float *frag_area);

    void WriteSurface(float *dev_frag_stats);
    void GetSurfaceScalers(float *dev_frag_stats);
    void PrintMatrix();

private:
    // Area of the fragment.
    float *dev_frag_area = 0;
    dev_facet *dev_data = 0;
    int3 frag_points;

private:
    // Presure on the surface.
    double *dev_Pr = 0;
    double *dev_Pi = 0;

    // Working buffers for facet to facet calculations.
    double *dev_Pr_initial = 0;
    double *dev_Pi_initial = 0;

    double *dev_Pr_result = 0;
    double *dev_Pi_result = 0;

    friend class ModelCuda;
};
#endif