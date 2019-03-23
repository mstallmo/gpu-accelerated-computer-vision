#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int subtractGpu(int h_a, int h_b);

int multiplyGpu(int h_a, int h_b);

void parallel1();
void parallel2();
void parallel3();

cudaDeviceProp getDeviceVersion();

void gpuCube(int *h_in, int *h_out);
