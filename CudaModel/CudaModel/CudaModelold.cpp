// CudaModel.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <tchar.h>


extern int addWithCuda(int *c, const int *a, const int *b, unsigned int size);
extern int cudaReset();

int _tmain(int argc, _TCHAR* argv[])
{
	const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    int cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != 0) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaReset();
    if (cudaStatus != 0) {
        printf("cudaDeviceReset failed!");
        return 1;
    }

    return 0;


	return 0;
}

