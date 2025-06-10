#ifndef _OBJECT_CUDA
#define _OBJECT_CUDA

#include "ObjectGl.hpp"
#include "ModelShared.h"

class ObjectCuda : public ObjectGl
{
public:
    ObjectCuda();
    ~ObjectCuda();
    void MakeCudaObjects();
    void WriteSurfaceToGlTexture(float min_dB, float max_dB, bool render_phase);
    void PrintSurfacePressure();
    dev_object MakeOptixStructArray();
    void AccumulatePressure();
    void PrimeReflections();
    void SwapOutputToInputPressure();

protected:
    float resolution = 0.0f;

private:
    dev_object h_obj = {};

};

#endif