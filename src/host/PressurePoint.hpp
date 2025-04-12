#ifndef _PressurePoint
#define _PressurePoint
#include "dcomplex.h"
#include <cuda_runtime.h>

class PressurePoint
{   
    public:
    float3 position;
    dcomplex pressureValue;
    
    public:
    PressurePoint(float3 p1)
    {
        position = p1;
        pressureValue.r = 0;
        pressureValue.i = 0;
    }

};
#endif