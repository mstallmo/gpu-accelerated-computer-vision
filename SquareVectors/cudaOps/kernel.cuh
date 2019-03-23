#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void gpuSquareKernel(float* d_in, float* d_out);

void gpuSquare(float* h_in, float* h_out);