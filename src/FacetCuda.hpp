#ifndef _FACET_CUDA
#define _FACET_CUDA

#include "FacetGl.hpp"
#include "ModelShared.h"

#include <cuda_runtime.h>

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

    void WriteSurface(float min_dB, float max_dB, bool render_phase);
    void PrintMatrix();

    dev_facet MakeOptixStruct();
    void AccumulatePressure();
    void PrimeReflections();
    void SwapOutputToInputPressure();    

private:
    // Area of the fragment.
    float *dev_frag_area = 0;

    dev_facet host_facet;

    int3 frag_points;

private:
    // Presure on the surface.
    dcomplex *dev_P = 0;

    // Working buffers for facet to facet calculations.
    dcomplex *dev_P_out = 0;
    dcomplex *dev_P_in = 0;

    friend class ModelCuda;
};
#endif