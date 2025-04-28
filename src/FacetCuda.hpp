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
    float *dev_frag_area = nullptr;
    dev_facet *dev_data = nullptr;
    int3 frag_points;

private:
    // Presure on the surface.
    double *dev_Pr = nullptr;
    double *dev_Pi = nullptr;

    // Working buffers for facet to facet calculations.
    double *dev_Pr_initial = nullptr;
    double *dev_Pi_initial = nullptr;

    double *dev_Pr_result = nullptr;
    double *dev_Pi_result = nullptr;

    friend class ModelCuda;
};
#endif