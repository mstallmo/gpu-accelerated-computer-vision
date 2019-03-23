#include <iostream>
#include "cudaOps/kernel.cuh"

#define N 100

void cpuAdd(int const* h_a, int const* h_b, int* h_c)
{
    int tid = 0;
    while (tid < N)
    {
        h_c[tid] = h_a[tid] + h_b[tid];
        tid += 1;
    }
}

int main() {
    int h_a[N], h_b[N], h_c[N];

    for (int i = 0; i < N; i++)
    {
        h_a[i] = 2 * i*i;
        h_b[i] = i;
    }

    cpuAdd(h_a, h_b, h_c);

    std::cout << "Vector addition on CPU" << std::endl;
    for (int i = 0; i < N; i++)
    {
        std::cout << "The sum of " << i << " element is " << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    gpuAdd(h_a, h_b, h_c);

    std::cout << "Vector addition on GPU" << std::endl;
    for (int i = 0; i < N; i++)
    {
        std::cout << "The sum of " << i << " element is " << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    return 0;
}
