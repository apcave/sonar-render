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