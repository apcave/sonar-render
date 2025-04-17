#ifndef _PressurePoint
#define _PressurePoint
#include "dcomplex.h"
#include <cuda_runtime.h>

class PressurePoint
{
public:
    float3 position;
    dcomplex pressure;

public:
    PressurePoint(float3 p1)
    {
        position = p1;
        pressure.r = 0;
        pressure.i = 0;
    }
};
#endif