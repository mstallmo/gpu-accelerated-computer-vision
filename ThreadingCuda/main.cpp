#include "cudaOps/kernel.cuh"
#include <iostream>

int main(void) 
{
    runKernel();
    cudaDeviceSynchronize();

    std::cout << "All threads are finished" << std::endl;
    return 0;
}
