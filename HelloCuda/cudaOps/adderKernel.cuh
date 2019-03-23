#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gpuAdd(int* d_a, int* d_b, int* d_c);

int gpuAdd(int const a, int const b);
