#include "CudaUtils.cuh"

/**
 * @Brief Mutex style lock for double precision floating point numbers.
 *        This is a workaround for the lack of atomicAdd for double precision
 *       floating point numbers in CUDA on older devices.
 */
__device__ double atomicAddDouble(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ float atomicMaxFloat(float *address, float val)
{
    int *address_as_int = (int *)address; // Treat the float as an int
    int old = *address_as_int, assumed;

    do
    {
        assumed = old;
        float old_val = __int_as_float(assumed);
        if (old_val >= val)
        {
            break; // No need to update if the current value is already greater or equal
        }
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);

    return __int_as_float(old);
}

__device__ float atomicMinFloat(float *address, float val)
{
    int *address_as_int = (int *)address; // Treat the float as an int
    int old = *address_as_int, assumed;

    do
    {
        assumed = old;
        float old_val = __int_as_float(assumed);
        if (old_val <= val)
        {
            break; // No need to update if the current value is already less or equal
        }
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);

    return __int_as_float(old);
}

__device__ float3 MakeVector(float3 Origin, float3 Dest)
{
    float3 Vc;
    Vc.x = Dest.x - Origin.x;
    Vc.y = Dest.y - Origin.y;
    Vc.z = Dest.z - Origin.z;
    return Vc;
}

__device__ float3 DivideVector(float3 vect, float variable)
{
    float3 dividedVect;
    dividedVect.x = vect.x / variable;
    dividedVect.y = vect.y / variable;
    dividedVect.z = vect.z / variable;
    return dividedVect;
}

__device__ float DotProduct(float3 v1, float3 v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ float GetVectorLength(float3 v1)
{
    return sqrtf(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
}