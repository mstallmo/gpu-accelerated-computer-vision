#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>

__global__ void gpuAddKernel(int const* const d_a, int const* const d_b, int* d_c);

void gpuAdd(int const* const h_a, int const* const h_b, int* h_c);