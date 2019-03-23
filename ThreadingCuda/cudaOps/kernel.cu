#include "kernel.cuh"
#include <stdio.h>

__global__ void firstKernel(void)
{
    printf("Hello I'm thread in block: %d\n", blockIdx.x);
}

void runKernel()
{
    getDeviceProperties();
    firstKernel << <16, 1 >> > ();
}

void getDeviceProperties()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0)
    {
        printf("There are no available devices that support CUDA");
    }
    else
    {
        printf("Detected %d CUDA capable device(s)\n", device_count);

        printDeviceName(0);
        printDriverRuntimeVersion();
        printMemoryData(0);
        printCpuData(0);
    }
}

void printDeviceName(int const deviceIndex)
{
    cudaDeviceProp device_property;
    cudaGetDeviceProperties(&device_property, deviceIndex);
    printf("\nDevice %d: \"%s\" \n", 0, device_property.name);
}

void printDriverRuntimeVersion()
{
    int driver_version, runtime_version;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    printf(" CUDA Driver Version / Runtime Version: %d.%d / %d.%d\n", driver_version / 1000, (driver_version % 100) / 10, runtime_version / 1000, (runtime_version % 100) / 10);
}

void printMemoryData(int const deviceIndex)
{
    cudaDeviceProp device_property;
    cudaGetDeviceProperties(&device_property, deviceIndex);
    
    printf(" Total amount of global memeory %.0f MBytes (%llu bytes)\n", (float)device_property.totalGlobalMem / 1048576.0f, (unsigned long long) device_property.totalGlobalMem);
    printf(" Memory Clock rate: %.0f Mhz\n", device_property.memoryClockRate);
    
    if (device_property.l2CacheSize)
    {
        printf(" L2 Cache Size: %d bytes\n", device_property.l2CacheSize);
    }

    printf(" Total amount of constant memory: %lu bytes\n", device_property.totalConstMem);
    printf(" Total amount of shared memeory per block: %lu bytes\n", device_property.sharedMemPerBlock);
    printf(" Total amount of regisers available per block: %d\n", device_property.regsPerBlock);
}

void printCpuData(int const deviceIndex)
{
    cudaDeviceProp device_property;
    cudaGetDeviceProperties(&device_property, deviceIndex);
    
    printf(" (%2d) Multiprocessors", device_property.multiProcessorCount);
    printf(" GPU Max Clock rate: %.0f MHz (%0.2f GHz)\n", device_property.clockRate * 1e-3f, device_property.clockRate * 1e-6f);

    printf(" Maximum number of threads per multiprocessor: %d\n", device_property.maxThreadsPerMultiProcessor);
    printf(" Maximum number of threads per block: %d\n", device_property.maxThreadsPerBlock);
    printf(" Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n", device_property.maxThreadsDim[0], device_property.maxThreadsDim[1], device_property.maxThreadsDim[2]);
    printf(" Max dimension size of a grid size (x, y, z): (%d, %d, %d)\n", device_property.maxGridSize[0], device_property.maxGridSize[1], device_property.maxGridSize[2]);
}
