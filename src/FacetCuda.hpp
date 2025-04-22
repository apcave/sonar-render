#ifndef _FACET_CUDA
#define _FACET_CUDA

#include "FacetGl.hpp"

#include <cuda_runtime.h>

class FacetCuda : public FacetGl
{
public:
    FacetCuda();
    void allocateCuda(float3 &normal,
                      float3 &base_point,
                      float3 &xAxis,
                      float3 &yAxis,
                      float *frag_area);

private:
    // Presure on the surface.
    double *dev_Pr;
    double *dev_Pi;

    // Area of the fragment.
    double *dev_frag_area;

    int3 *dev_frag_points;
    float3 *dev_facet_normal;
    float3 *dev_base_point;
    float3 *dev_xAxis;
    float3 *dev_yAxis;
    int3 frag_points;

protected:
    // Working buffers for facet to facet calculations.
    double *dev_Pr_initial;
    double *dev_Pi_initial;

    double *dev_Pr_result;
    double *dev_Pi_result;

    friend class CudaModelTes;
}
#endif