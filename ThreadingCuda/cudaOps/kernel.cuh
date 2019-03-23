#include <cuda.h>
#include <cuda_runtime.h>

__global__ void firstKernel(void);

void runKernel();
void getDeviceProperties();
void printDeviceName(int const deviceIndex);
void printDriverRuntimeVersion();
void printMemoryData(int const deviceIndex);
void printCpuData(int const deviceIndex);
