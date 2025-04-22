#include "FacetGl.h"
#include <cuda_runtime.h>

__global__ void MakeSurface(double *P, cudaSurfaceObject_t surface, int maxXpnt, float *stats)
{
    int xPnt = threadIdx.x;
    int yPnt = blockIdx.x;

    int index = yPnt * maxXpnt + xPnt;

    float value = 0.5 + ((float)P[index] / stats[0]);

    // value = (float)yPnt / (float)maxXpnt;
    surf2Dwrite(value, surface, xPnt * sizeof(float), yPnt);

    // printf("Write surface: %f, %f, %f\n", value, stats[0], stats[1]);
}

void FacetGl::WriteSurface(double *dev_Pr, double *dev_Pi, float *dev_frag_stats)
{
    MapToCuda();
    // Copy the data to the device.
    // Make the texture out of the real values.
    dim3 threadsPerBlock(numXpnts, 1);
    dim3 numBlocks(numYpnts, 1);
    MakeSurface<<<numBlocks, threadsPerBlock>>>(dev_Pr, surface, numXpnts, stats);

    cudaGraphicsUnmapResources(1, &cudaResource, 0);
    readyToRender = true;
}